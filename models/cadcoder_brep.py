import pathlib
import warnings
from datetime import datetime

import dgl
import psutil
import torch
from peft import PeftModel
from torch import nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import wandb
from models import BaseModel
from models.config import Config
from models.encoder_brep import BrepEncoder
from modules.cad_evaluator import CadEvaluator

# Suppress DGL autocast deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dgl")


def print_memory_usage(stage=""):
    """Print current memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(
                f"{stage} - GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

    # RAM usage
    ram = psutil.virtual_memory()
    print(
        f"{stage} - RAM: {ram.used/1024**3:.2f}GB / {ram.total/1024**3:.2f}GB ({ram.percent:.1f}%)"
    )


class BrepCadcoder(BaseModel):
    def __init__(self, config: Config, peft_model=None):
        super().__init__(config)
        self.config = config

        # Prefix encoder (already has built-in alignment layers)
        self.prefix_encoder = BrepConditionedPrefixEncoder(config).to(
            dtype=torch.bfloat16
        )

    def get_prefix_embeds(self, batch, batch_idx=None):
        dgl_graphs = batch["graphs"]
        prefix_embeds = self.prefix_encoder(dgl_graphs)
        return prefix_embeds

    def generate(
        self,
        graph,
        context: str | None = None,
        temperature=None,
        top_p=None,
        top_k=None,
    ):
        self.model.eval()

        # Defaults
        temperature = (
            self.config.inference.temperature
            if hasattr(self.config.inference, "temperature")
            else 0.7
            if temperature is None
            else temperature
        )
        # top_p = self.config.inference.top_p if top_p is None else top_p
        # top_k = self.config.inference.top_k if top_k is None else top_k

        # Some tokenizers don't define a pad token; make sure we have one for safety.
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with torch.inference_mode():
            # Prefix embeddings from the graph encoder
            prefix_embeds = self.prefix_encoder(graph.to(self.device))  # [1, T_pref, D]
            input_embeds = prefix_embeds
            attention_mask = torch.ones(
                (1, input_embeds.shape[1]), dtype=torch.long, device=self.device
            )

            # ---- Optional: system prompt + context (raw text) ----
            pieces = []
            if context:
                pieces.append(context)

            if pieces:
                tok = self.tokenizer(
                    pieces,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=False,
                )
                # Concatenate pieces along sequence dim
                input_ids = tok["input_ids"].to(self.device)
                token_embeds = self.model.get_input_embeddings()(input_ids)  # [1, L, D]

                input_embeds = torch.cat(
                    [input_embeds, token_embeds], dim=1
                )  # [1, T_pref+L, D]
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, token_embeds.shape[1]),
                            device=self.device,
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
                )
                generated_ids = input_ids.clone()  # start with given tokens
            else:
                # Start “empty” (only prefix) and grow
                generated_ids = torch.empty(
                    (1, 0), dtype=torch.long, device=self.device
                )

            # ---- Autoregressive sampling loop ----
            max_new = self.config.inference.max_new_tokens
            for _ in tqdm(range(max_new), total=max_new):
                out = self.model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                # Temperature
                logits = out.logits[:, -1, :] / max(temperature, 1e-6)

                # Top-k filter
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    topk_vals, topk_idx = torch.topk(logits, k)
                    filt = torch.full_like(logits, float("-inf"))
                    logits = filt.scatter(1, topk_idx, topk_vals)

                # Top-p (nucleus) filter
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(probs, dim=-1)
                    # mask tokens past the nucleus
                    mask = cumprobs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = 0
                    # Scatter back to original order
                    logits[
                        torch.gather(
                            torch.arange(
                                logits.size(-1), device=logits.device
                            ).unsqueeze(0),
                            1,
                            sorted_idx,
                        )
                    ] = sorted_logits.masked_fill(mask, float("-inf"))

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

                # IDs
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Embeds + mask
                next_embed = self.model.get_input_embeddings()(next_token)  # [1,1,D]
                input_embeds = torch.cat([input_embeds, next_embed], dim=1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=torch.long),
                    ],
                    dim=1,
                )

                # Early stop
                if (
                    self.tokenizer.eos_token_id is not None
                    and next_token.item() == self.tokenizer.eos_token_id
                ):
                    break

        # Decode only the *text* portion (i.e., not the prefix-embeds). Since `generated_ids`
        # contains only tokenizer-produced tokens (system/context + generated), decoding is simple:
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def test_step(self, batch, batch_idx):
        predictions = []
        i = 0
        for cad_id, cad_code_gt, graph, code_path in zip(
            batch["id"], batch["code"], dgl.unbatch(batch["graph"]), batch["code_path"]
        ):
            generated_code = self.generate(graph=graph, context=cad_code_gt[:50])

            pred = {
                "code_path": code_path,
                "prediction": generated_code,
                "ground_truth": cad_code_gt,
                "timestamp": datetime.now().isoformat(),
                "id": cad_id,
            }
            predictions.append(pred)
            i += 1
            if i == 5:
                break

        self._save_test_results(predictions)

    def _save_test_results(self, predictions, batch_idx=0):
        """Save test results to a JSON file"""
        import json
        from datetime import datetime

        # Create results directory if it doesn't exist
        results_dir = pathlib.Path(__file__).parent / "test_results"
        results_dir.mkdir(exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(self.logger.experiment.name, str):
            run_name = self.logger.experiment.name + "_" + str(batch_idx)
        else:
            run_name = f"{timestamp}_{batch_idx}"
        filename = f"cadcoder_test_results_{run_name}.json"
        filepath = results_dir / filename

        # Prepare summary statistics
        summary = {
            "total_samples": len(predictions),
        }

        valid_counter = 0
        invalid_counter = 0
        sum_chamfer_distances = 0.0
        sum_f_scores = 0.0
        sum_nc_scores = 0.0
        sum_code_bleu_scores = 0.0
        invalid_reasons = []
        for prediction in predictions:
            generated_code = prediction["prediction"]
            cad_code_gt = prediction["ground_truth"]
            try:
                eval = CadEvaluator(sampler_points=500, threshold=0.05)
                result = eval.evaluate(code_gt=cad_code_gt, code_pred=generated_code)
                chamfer = result.get("chamfer")
                f_score = result.get("fscore")
                nc = result.get("normal_consistency")
                codebleu = result.get("codebleu")
                invalid_reason = result.get("invalid_reason")

                if invalid_reason is None and chamfer is not None:
                    valid_counter += 1
                    sum_chamfer_distances += chamfer
                    sum_f_scores += f_score
                    sum_nc_scores += nc
                else:
                    invalid_counter += 1
                    invalid_reasons.append(
                        {
                            "reason": invalid_reason,
                            "code": generated_code,
                            "code_gt": cad_code_gt,
                        }
                    )
                sum_code_bleu_scores += codebleu
            except AttributeError:
                continue

        invalid_ratio = (invalid_counter + 1) / (valid_counter + invalid_counter + 1)
        summary["metrics/mean_chamfer_distance"] = (
            sum_chamfer_distances / valid_counter if valid_counter > 0 else None
        )
        summary["metrics/mean_f_score"] = (
            sum_f_scores / valid_counter if valid_counter > 0 else None
        )
        summary["metrics/mean_normal_consistency"] = (
            sum_nc_scores / valid_counter if valid_counter > 0 else None
        )
        summary["metrics/mean_codebleu"] = (
            sum_code_bleu_scores / (valid_counter + invalid_counter)
            if (valid_counter + invalid_counter) > 0
            else None
        )
        summary["metrics/invalid_ratio"] = invalid_ratio
        self.logger.log_metrics(
            summary,
        )

        # Combine summary and detailed results
        results = {
            "summary": summary,
            "detailed_results": predictions,
        }

        if invalid_reasons:
            invalid_reason_table = wandb.Table(columns=["reason", "code", "code_gt"])
            for reason in invalid_reasons:
                invalid_reason_table.add_data(
                    reason["reason"], reason["code"], reason["code_gt"]
                )
            wandb.log({"invalid_reason": invalid_reason_table})

        # Save to file
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Test results saved to: {filepath}")

        # Save to wandb
        if self.config.logging.logger == "wandb":
            artifact = wandb.Artifact(
                filename,
                type="evaluation",
            )
            artifact.add_file(str(filepath))
            self.logger.experiment.log_artifact(artifact)

        print(f"Test results saved to: {filepath}")
        print(summary)

    # def on_train_end(self) -> None:
    #     checkpoint_path = (
    #         pathlib.Path(__file__).parent.parent.parent
    #         / "checkpoints"
    #         / self.config.model
    #     )
    #     if self.config.mode == "train" and self.trainer.is_global_zero:
    #         print(f"Saving checkpoint to: {checkpoint_path}")
    #         # self.trainer.save_checkpoint(checkpoint_path, weights_only=True)
    #     #  print(f"Saved Lightning checkpoint to {checkpoint_path}")
    #
    #     # run_name = self.logger.experiment.name
    #     # self.model.save_pretrained(checkpoint_path / run_name)
    #     # print(
    #     #     f"Model and tokenizer saved to brep_cadcoder_model_{date_time_str}/ and brep_cadcoder_tokenizer/"
    #     # )

    @staticmethod
    def load_from_checkpoint_2(ckpt_path, config):
        base_model = AutoModelForCausalLM.from_pretrained(
            config.cadcoder.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )

        checkpoint_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "checkpoints"
            / config.general.model
            / "brep_cadcoder_model"
        )

        # Load PEFT config and adapter weights
        print(f"Checkpoint path: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        return BrepCadcoder(config, peft_model=model)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.generate(graph=batch["graph"])


class BrepConditionedPrefixEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = BrepEncoder(config)  # outputs (batch_size, dim_latent)
        self.config = config

        self.graph_to_prefix = nn.Sequential(
            nn.Linear(self.config.decoder.dim_latent, self.config.decoder.dim_hidden),
            nn.ReLU(),
            nn.Dropout(self.config.decoder.dropout),
            nn.Linear(
                config.decoder.dim_hidden,
                config.cadcoder.num_virtual_tokens * self.config.decoder.dim_model,
            ),
            nn.Tanh(),
        )
        self.config = config

    def forward(self, graph_input):
        batch_size = graph_input.batch_size

        # This should return (B, dim_latent)
        _, graph_embedding = self.encoder(
            graph_input
        )  # optionally unpack if encoder returns tuple

        # Map to soft token embeddings
        prefix = self.graph_to_prefix(graph_embedding)  # shape: [B, V * D]
        prefix = prefix.view(
            batch_size,
            self.config.cadcoder.num_virtual_tokens,
            self.config.decoder.dim_model,
        )  # shape: [B, V, D]

        return prefix
