import pathlib

import lightning as L
import seaborn as sns
import torch
from deepspeed.ops.adam import FusedAdam
from matplotlib import pyplot as plt
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

import wandb
from models.alignment_loss import AlignmentLoss


class BaseModel(L.LightningModule):
    """
    Base class for BrepCadcoder and other custom models.
    Shared: training_step, validation_step, configure_optimizers
    Require override for forward(batch).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.multi_gpu = config.deepspeed.multi_gpu
        self.lr = config.optimizer.learning_rate

        self.align_loss = AlignmentLoss(
            mode="best_match",  # or "centroid"
            weight=config.loss.alignment_loss_weight,
            sample_tokens=128,
            detach_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.cadcoder.model_name,
            trust_remote_code=True,  # device_map="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if config.cadcoder.quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        if self.config.cadcoder.use_pretrained:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.cadcoder.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                quantization_config=bnb_config,
                attn_implementation=config.cadcoder.attn_implementation,
                use_safetensors=True,
            )
        else:
            cfg = AutoConfig.from_pretrained(
                self.config.cadcoder.model_name,
                trust_remote_code=True,
            )
            base_model = AutoModelForCausalLM.from_config(
                cfg,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.config.cadcoder.attn_implementation,
            )
        if config.cadcoder.use_lora:
            self.peft_config = LoraConfig(
                r=config.cadcoder.lora_r,
                lora_alpha=config.cadcoder.lora_r,
                target_modules=self.config.cadcoder.lora_target_modules,
                lora_dropout=config.decoder.dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            base_model = get_peft_model(base_model, self.peft_config)
        base_model.config.use_cache = False
        if config.optimizer.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        for name, param in base_model.named_parameters():
            if "lora" in name.lower() and not param.requires_grad:
                param.requires_grad = True
        self.model = base_model

    def forward(
        self,
        prefix_embeds,
        input_ids,
        attention_mask,
        labels,
        batch,
        batch_idx,
        mode="train",
    ):
        # Alignment Loss
        alignment_loss = torch.tensor(0.0, device=self.device)
        cosine_similarity = torch.tensor(0.0, device=self.device)
        if self.config.loss.use_alignment_loss:
            alignment_loss, cosine_similarity = self.align_loss(
                prefix_embeds=prefix_embeds,  # [B, T_pref, D]
                input_ids=input_ids,  # [B, T_seq]
                embed_fn=self.model.get_input_embeddings(),  # or lambda ids: emb(ids)
                pad_id=self.tokenizer.pad_token_id,
                eos_id=self.tokenizer.eos_token_id,
                ignore_id=0,
            )
            assert not torch.isnan(alignment_loss).any(), "Alignment loss is NaN"

        # 4. Get text embeddings
        text_embeds = self.model.get_input_embeddings()(input_ids)

        # 5. Concatenate prefix + text
        if prefix_embeds.dim() > 1:
            text_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
        # text_embeds[:, : self.config.cadcoder.num_virtual_tokens, :] = prefix_embeds

        # Clear intermediate tensors
        del prefix_embeds

        outputs = self.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_attentions=self.config.logging.log_attention_scores,
        )

        # Clear large tensors immediately
        del attention_mask

        # Combine losses
        if self.config.loss.use_alignment_loss:
            total_loss = outputs.loss + alignment_loss
        else:
            total_loss = outputs.loss
        return_dict = {
            "loss": total_loss,
        }
        if self.trainer.global_step % self.config.logging.log_every_n_steps == 0:
            return_dict.update(
                {
                    "generation_loss": outputs.loss.detach().cpu().item(),
                    "alignment_loss": alignment_loss.detach().cpu().item(),
                    "cosine_similarity": cosine_similarity.detach().cpu().item(),
                }
            )
        if batch_idx < 3 and self.config.logging.log_one_sample:
            return_dict.update(
                {
                    "logits": outputs.logits.detach().cpu(),
                    "labels": labels.detach().cpu(),
                }
            )
        return return_dict

    def get_prefix_embeds(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        prefix_embeds = self.get_prefix_embeds(batch)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        return_dict = self.forward(
            prefix_embeds, input_ids, attention_mask, labels, batch, batch_idx
        )
        self.log(
            "train/total_loss",
            return_dict["loss"].item(),
            on_step=True,
            sync_dist=False,
            rank_zero_only=True,
            batch_size=batch["input_ids"].shape[0],
        )
        if "generation_loss" in return_dict:
            self.log(
                "train/generation_loss",
                return_dict["generation_loss"],
                rank_zero_only=True,
                batch_size=batch["input_ids"].shape[0],
                on_step=True,
            )
            self.log(
                "train/alignment_loss",
                return_dict["alignment_loss"],
                rank_zero_only=True,
                batch_size=batch["input_ids"].shape[0],
                on_step=True,
            )
            self.log(
                "train/cosine_similarity",
                return_dict["cosine_similarity"],
                rank_zero_only=True,
                batch_size=batch["input_ids"].shape[0],
                on_step=True,
            )
        return return_dict

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            prefix_embeds = self.get_prefix_embeds(batch)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            return_dict = self.forward(
                prefix_embeds,
                input_ids,
                attention_mask,
                labels,
                batch,
                batch_idx,
                mode="validation",
            )
            self.log(
                "val/total_loss",
                return_dict["loss"].item(),
                sync_dist=False,
                rank_zero_only=True,
                batch_size=batch["input_ids"].shape[0],
            )
            if "generation_loss" in return_dict:
                self.log(
                    "val/generation_loss",
                    return_dict["generation_loss"],
                    rank_zero_only=True,
                    batch_size=batch["input_ids"].shape[0],
                )
                self.log(
                    "val/alignment_loss",
                    return_dict["alignment_loss"],
                    rank_zero_only=True,
                    batch_size=batch["input_ids"].shape[0],
                )
                self.log(
                    "val/cosine_similarity",
                    return_dict["cosine_similarity"],
                    rank_zero_only=True,
                    batch_size=batch["input_ids"].shape[0],
                )
            return return_dict

    def configure_optimizers(self):
        if self.config.deepspeed.multi_gpu:
            # optimizer = DeepSpeedCPUAdam(
            #     [
            #         {"params": self.prefix_encoder.parameters(), "lr": self.config.optimizer.learning_rate},
            #         {"params": self.model.parameters(), "lr": self.config.optimizer.learning_rate},
            #     ],
            #     lr=self.config.optimizer.learning_rate,  # optional global default
            #     weight_decay=self.config.loss.weight_decay,
            #     betas=(0.9, 0.999),
            # )
            optimizer = FusedAdam(
                [
                    {
                        "params": self.prefix_encoder.parameters(),
                        "lr": self.config.optimizer.learning_rate,
                    },
                    {
                        "params": self.model.parameters(),
                        "lr": self.config.optimizer.learning_rate,
                    },
                ],
                lr=self.config.optimizer.learning_rate,  # optional global default
                weight_decay=self.config.loss.weight_decay,
                betas=(0.9, 0.999),
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self.prefix_encoder.parameters(),
                        "lr": self.config.optimizer.learning_rate * 5,
                    },
                    {
                        "params": self.model.parameters(),
                        "lr": self.config.optimizer.learning_rate,
                    },
                ],
                lr=self.config.optimizer.learning_rate,  # optional global default
                weight_decay=self.config.loss.weight_decay,
                betas=(0.9, 0.999),
            )
        if self.config.optimizer.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.optimizer.step_size,
                T_mult=2,
                eta_min=1e-7,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        elif self.config.optimizer.scheduler == "CosineWithWarmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.optimizer.warmup_steps,
                num_training_steps=50,  # e.g., steps_per_epoch * num_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif self.config.optimizer.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.optimizer.step_size,
                gamma=self.config.optimizer.gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Step every epoch
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def attention_visualization(self, batch, attentions, max_tokens: int = 20):
        if (
            not attentions
            or not hasattr(self, "logger")
            or not hasattr(self.logger, "experiment")
        ):
            return

        try:
            num_virtual_tokens = self.config.cadcoder.num_virtual_tokens

            for sample_idx in range(self.config.training.batch_size):
                # Take the single sample from each layer: attn[0] → [heads, seq, seq]
                sample_attn = torch.stack(
                    [attn[0] for attn in attentions]
                )  # [layers, heads, seq, seq]
                avg_attn = (
                    sample_attn.mean(dim=(0, 1)).detach().float().cpu().numpy()
                )  # [seq, seq]

                # Truncate to first max_tokens
                truncated_attn = avg_attn[:max_tokens, :max_tokens]

                # Tokens: <prefix_0>, <prefix_1>, ... + decoded code
                token_ids = batch["input_ids"][sample_idx]
                token_ids = token_ids[token_ids >= 0]  # filter out -100
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                # Add prefix tokens
                prefix_tokens = [f"<prefix_{i}>" for i in range(num_virtual_tokens)]
                tokens = prefix_tokens + tokens
                tokens = tokens[:max_tokens]  # truncate tokens to match attention

                # Plot
                fig, ax = plt.subplots(figsize=(14, 14))
                sns.heatmap(
                    truncated_attn,
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap="viridis",
                    ax=ax,
                    annot=True,
                    fmt=".2f",
                )
                ax.set_title(f"Attention Heatmap - Sample {sample_idx}")
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()

                self.logger.experiment.log(
                    {f"attention_heatmap_sample_{sample_idx}": wandb.Image(fig)}
                )
                # plt.show()
                plt.close(fig)

        except Exception as e:
            print(f"Error in attention visualization: {e}")

    def on_train_start(self):
        if self.config.training.profiler:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs"),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
                profile_memory=True,
            )
            self.profiler.start()
            torch.cuda.memory._record_memory_history()

    def on_train_epoch_start(self):
        if self.current_epoch >= self.config.encoder.freeze_encoder_epoch:
            for p in self.prefix_encoder.parameters():
                p.requires_grad = False

    def on_train_batch_end(self, outputs: dict, batch, batch_idx):
        if self.config.training.profiler:
            self.profiler.step()
        if (
            self.config.logging.log_one_sample
            and batch_idx < 3
            and not self.trainer.sanity_checking
        ):
            self.outputs_ids = batch["ids"]
            self.trainer.callbacks[0]._log_generated_sample(
                trainer=self.trainer,
                pl_module=self,
                logits=outputs["logits"],
                true_labels=batch["labels"].cpu(),
                phase="train",
                ids=batch["ids"],
                batch_idx=batch_idx,
            )

        if (
            self.config.logging.log_attention_scores
            and self.current_epoch == self.config.training.max_epochs - 1
            and batch_idx == 0
        ):
            self.attention_visualization(
                batch,
                outputs.attentions,
                max_tokens=self.config.cadcoder.num_virtual_tokens * 2,
            )
        if (
            self.config.memory.empty_cache_freq > 0
            and batch_idx % self.config.memory.empty_cache_freq == 0
        ):
            torch.cuda.empty_cache()

    def on_validation_batch_end(self, outputs: dict, batch, batch_idx) -> None:
        if (
            self.config.memory.empty_cache_freq > 0
            and batch_idx % self.config.memory.empty_cache_freq == 0
        ):
            torch.cuda.empty_cache()
        if (
            self.config.logging.log_one_sample
            and batch_idx < 3
            and not self.trainer.sanity_checking
        ):
            self.outputs_ids = batch["ids"]
            self.trainer.callbacks[0]._log_generated_sample(
                trainer=self.trainer,
                pl_module=self,
                logits=outputs["logits"],
                true_labels=batch["labels"],
                phase="val",
                ids=batch["ids"],
                batch_idx=batch_idx,
            )

    def on_train_end(self):
        if self.config.training.profiler:
            self.profiler.stop()
            print(
                self.profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=10
                )
            )
            print(
                self.profiler.key_averages().table(
                    sort_by="cuda_memory_usage", row_limit=10
                )
            )

    def on_save_checkpoint(self, checkpoint):
        ckpt_path = str(
            pathlib.Path(__file__).parent.parent.parent
            / "checkpoints"
            / self.config.general.model
        )
        print("Saved Checkpoint to ", ckpt_path)
        self.model.save_pretrained(ckpt_path)
        state = checkpoint["state_dict"]
        filtered_state = {}
        for name, param in state.items():
            if "lora" in name:
                filtered_state[name] = state[name]
        return checkpoint
