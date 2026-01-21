import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

import wandb
from models import BaseModel
from models.encoder_text import TextEmbedder
from modules.cad_evaluator import CadEvaluator


class TextCadcoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.invalid_reason_table = wandb.Table(columns=["reason", "code", "code_gt"])
        self.intermediate_outputs = wandb.Table(
            columns=["batch_idx", "text", "code", "generated_code", "epoch", "loss"]
        )
        self.validation_outputs = wandb.Table(columns=["text", "code", "code_gt"])
        self.prefix_embeds_pca = wandb.Table(columns=["text", "x", "y", "epoch"])
        self.prefix_encoder = TextEmbedder(config)

    def get_prefix_embeds(self, batch):
        text = batch["text"]
        prefix_embeds = self.prefix_encoder(text)
        return prefix_embeds

    def test_step(self, batch, batch_idx):
        # Store batch index for reference in results
        self._current_batch_idx = batch_idx
        return self._evaluate_batch(batch, batch_idx, timed=True)

    def predict_step(self, batch, batch_idx):
        return self._evaluate_batch(batch, timed=True)

    def _evaluate_batch(self, batch, batch_idx, timed=False):
        import time
        from datetime import datetime

        predictions = []
        self.model.eval()

        for id, cad_code_gt, text, code_path in zip(
            batch["id"], batch["code"], batch["text"], batch["code_path"]
        ):
            with torch.inference_mode():
                # add batch dimension
                prefix_embeds = self.prefix_encoder(text.unsqueeze(0))
                bos_embedding = self.model.get_input_embeddings().weight[
                    self.tokenizer.bos_token_id
                ]
                if self.config.cadcoder.use_bos_token:
                    prefix_embeds = torch.cat(
                        [prefix_embeds, bos_embedding.unsqueeze(0).unsqueeze(0)], dim=1
                    )
                attention_mask = torch.ones(
                    (1, prefix_embeds.shape[1]),
                    device=prefix_embeds.device,
                    dtype=torch.long,
                )
                start_time = time.time() if timed else None
                output_ids = self.model.generate(
                    inputs_embeds=prefix_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.inference.max_new_tokens,
                    do_sample=True,
                    top_k=10,
                    temperature=0.2,
                )
                duration = time.time() - start_time if timed else None
                tokens_per_sec = output_ids.shape[1] / duration
                self.log("ptl/tokens_per_sec", tokens_per_sec, sync_dist=True)

            # Decode only the generated part (skip the prompt)
            generated_code = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            valid_counter = 0
            invalid_counter = 0
            invalid_reason = []
            decoded_text = self.text_tokenizer.decode(text, skip_special_tokens=True)
            self.validation_outputs.add_data(
                decoded_text,
                generated_code,
                self.tokenizer.decode(
                    cad_code_gt[cad_code_gt != 1], skip_special_tokens=True
                ),
            )
            try:
                eval = CadEvaluator(sampler_points=500, threshold=0.05)
                chamfer, f_score, nc = eval.evaluate(
                    code_gt=self.tokenizer.decode(cad_code_gt[cad_code_gt != 1][1:]),
                    code_pred=generated_code,
                ).values()
                valid_counter += 1
            except AttributeError as e:
                invalid_reason.append(
                    {"reason": str(e), "code": generated_code, "code_gt": cad_code_gt}
                )
                chamfer, f_score, nc = None, None, None
            except ValueError as e:
                chamfer, f_score, nc = None, None, None
                invalid_counter += 1
                invalid_reason.append(
                    {"reason": str(e), "code": generated_code, "code_gt": cad_code_gt}
                )

            pred = {
                "text": self.text_tokenizer.decode(text[0]),
                "code_path": code_path,
                "cad_code_pred": generated_code,
                "cad_code_gt": self.tokenizer.decode(
                    cad_code_gt[cad_code_gt != 1][1:], skip_special_tokens=True
                ),
                "chamfer_distance": chamfer,
                "f_score": f_score,
                "normal_consistency": nc,
                "timestamp": datetime.now().isoformat(),
                "id": id,
            }
            if timed:
                pred["generation_time_sec"] = duration

            predictions.append(pred)

        # Save results to file (only on rank 0)
        if self.trainer.is_global_zero:
            if valid_counter + invalid_counter == 0:
                invalid_ratio = 1
            else:
                invalid_ratio = invalid_counter / (valid_counter + invalid_counter)
            self._save_test_results(
                predictions, invalid_ratio, invalid_reason, batch_idx
            )
        return predictions

    def _save_test_results(
        self, predictions, invalid_ratio, invalid_reason, batch_idx=0
    ):
        """Save test results to a JSON file"""
        import json
        from datetime import datetime

        # Create results directory if it doesn't exist
        results_dir = pathlib.Path("test_results")
        results_dir.mkdir(exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cadcoder_test_results_{timestamp}.json"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(self.logger.experiment.name, str):
            run_name = self.logger.experiment.name + "_" + str(batch_idx)
        else:
            run_name = f"{timestamp}_{batch_idx}"
        filename = f"cadcoder_test_results_{run_name}.json"
        filepath = results_dir / filename

        # Prepare summary statistics
        valid_predictions = [
            p for p in predictions if p["chamfer_distance"] is not None
        ]
        summary = {
            "total_samples": len(predictions),
            "valid_evaluations": len(valid_predictions),
            "failed_evaluations": len(predictions) - len(valid_predictions),
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "epoch": getattr(self.trainer, "current_epoch", 0),
                "model_name": "BrepCadcoder",
            },
        }

        if valid_predictions:
            chamfer_distances = [p["chamfer_distance"] for p in valid_predictions]
            f_scores = [p["f_score"] for p in valid_predictions]
            nc_scores = [p["normal_consistency"] for p in valid_predictions]

            summary["metrics"] = {
                "chamfer_distance": {
                    "mean": sum(chamfer_distances) / len(chamfer_distances),
                    "min": min(chamfer_distances),
                    "max": max(chamfer_distances),
                },
                "f_score": {
                    "mean": sum(f_scores) / len(f_scores),
                    "min": min(f_scores),
                    "max": max(f_scores),
                },
                "normal_consistency": {
                    "mean": sum(nc_scores) / len(nc_scores),
                    "min": min(nc_scores),
                    "max": max(nc_scores),
                },
            }
            self.logger.log_metrics(
                {
                    "mean_chamfer_distance": summary["metrics"]["chamfer_distance"][
                        "mean"
                    ],
                    "mean_f_score": summary["metrics"]["f_score"]["mean"],
                    "mean_normal_consistency": summary["metrics"]["normal_consistency"][
                        "mean"
                    ],
                    "invalid_ratio": invalid_ratio,
                }
            )

        # Combine summary and detailed results
        results = {
            "summary": summary,
            "invalid_ratio": invalid_ratio,
            "detailed_results": predictions,
        }

        if invalid_reason:
            for reason in invalid_reason:
                self.invalid_reason_table.add_data(
                    reason["reason"], reason["code"], reason["code_gt"]
                )

        # Save to file
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Test results saved to: {filepath}")
        print(
            f"Summary: {summary['valid_evaluations']}/{summary['total_samples']} successful evaluations"
        )

        if valid_predictions:
            avg_chamfer = summary["metrics"]["chamfer_distance"]["mean"]
            avg_f_score = summary["metrics"]["f_score"]["mean"]
            avg_nc = summary["metrics"]["normal_consistency"]["mean"]
            print(
                f"Average metrics - Chamfer: {avg_chamfer:.4f}, F-Score: {avg_f_score:.4f}, NC: {avg_nc:.4f}"
            )

    def on_train_end(self) -> None:
        # Existing logging
        wandb.log({"invalid_reason": self.invalid_reason_table})
        wandb.log({"detailed_results": self.intermediate_outputs})

        # try:
        #     import numpy as np
        #     from tqdm import tqdm
        #
        #     # Get the dataloader
        #     dataloader = self.trainer.train_dataloader
        #     all_embeddings = []
        #     text_prompts = []
        #
        #     # Set model to eval mode
        #     was_training = self.training
        #     self.eval()
        #
        #     # Process batches
        #     with torch.no_grad():
        #         for batch in tqdm(dataloader, desc="Generating embeddings"):
        #             if isinstance(batch, dict) and 'text' in batch:
        #                 # Get text inputs
        #                 texts = batch['text']
        #                 if isinstance(texts, torch.Tensor):
        #                     texts = self.text_tokenizer.batch_decode(texts, skip_special_tokens=True)
        #
        #                 # Generate embeddings
        #                 embeddings = self.prefix_encoder(batch['text'].to(device="cuda"))
        #                 all_embeddings.append(embeddings.cpu())
        #                 text_prompts.extend(texts)
        #
        #                 # Limit number of samples for visualization
        #                 if len(text_prompts) >= 1000:  # Adjust this number as needed
        #                     break
        #     if not all_embeddings:
        #         return
        #
        #     # Concatenate all embeddings
        #     all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        #
        #     # Perform PCA
        #     pca = PCA(n_components=2)
        #     reduced = pca.fit_transform(all_embeddings)
        #
        #     # Create and log the table
        #     table = wandb.Table(columns=["x", "y", "label", "text"])
        #     for i in range(len(reduced)):
        #         label = f"sample_{i}"
        #         text = text_prompts[i] if i < len(text_prompts) else ""
        #         table.add_data(float(reduced[i, 0]), float(reduced[i, 1]), label, text)
        #
        #     # Log the plot
        #     wandb.log({
        #         "prefix_embed_pca": wandb.plot.scatter(
        #             table,
        #             "x",
        #             "y",
        #             title="Prefix Embedding PCA",
        #         )
        #     })
        #
        # except Exception as e:
        #     import traceback
        #     print(f"Error in embedding visualization: {e}")
        #     traceback.print_exc()
        #
        # finally:
        #     # Restore training mode
        #     if was_training:
        #         self.train()

    def on_test_end(self) -> None:
        # Log the final validation results
        if hasattr(self, "validation_outputs") and hasattr(self, "logger"):
            wandb.log({"validation_results": self.validation_outputs})
        # )
        # if self.config.mode == "train":
        #     shutil.rmtree(checkpoint_path / "text_cadcoder_model/", ignore_errors=True)
        #     shutil.rmtree(
        #         checkpoint_path / "text_cadcoder_tokenizer/", ignore_errors=True
        #     )
        #     self.model.save_pretrained(checkpoint_path / "text_cadcoder_model/")
        #     self.tokenizer.save_pretrained(checkpoint_path / "text_cadcoder_tokenizer/")
        #     print(f"Model and tokenizer saved to {checkpoint_path}")

    @staticmethod
    def load_from_checkpoint(ckpt_path, config):
        base_model = AutoModelForCausalLM.from_pretrained(
            config.cadcoder.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )

        # Load PEFT config and adapter weights
        checkpoint_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "checkpoints"
            / config.general.model
            / "text_cadcoder_model"
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        return TextCadcoder(config, peft_model=model)
