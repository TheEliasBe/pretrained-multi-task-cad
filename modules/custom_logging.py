import json
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

import wandb
from modules.cad_evaluator import CadEvaluator

if TYPE_CHECKING:
    import pytorch_lightning as pl

logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def visualize_logits(param):
    pass


class LoggingCallbackWandb(Callback):
    def __init__(self, config):
        self.config = config
        self.per_epoch_example = []
        self.console = Console()
        if self.config.logging.log_one_sample:
            self.log_one_sample_table = wandb.Table(
                columns=[
                    "epoch",
                    "batch",
                    "is_valid",
                    "id",
                    "generated_code",
                    "true_code",
                ]
            )
            self.log_one_sample_table_val = wandb.Table(
                columns=[
                    "epoch",
                    "batch",
                    "is_valid",
                    "id",
                    "generated_code",
                    "true_code",
                ]
            )
            self.log_one_sample_data = []
            self.log_one_sample_data_val = []

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        ...
        # metrics = {
        #     "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
        # }
        #
        # # Dynamically extract all callback metrics
        # for key, value in trainer.callback_metrics.items():
        #     if isinstance(value, torch.Tensor):
        #         metrics[key] = value.item()
        #
        # trainer.logger.log_metrics(metrics=metrics)

    def _log_generated_sample(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        logits: torch.Tensor,
        true_labels: torch.Tensor,
        phase: str,  # either "train" or "val"
        ids: torch.Tensor = None,
        batch_idx: int = 0,
    ):
        valid_sum = 0
        invalid_sum = 0
        for i in range(logits.shape[0]):
            true_labels_i = true_labels[
                i, self.config.cadcoder.num_virtual_tokens :
            ].cpu()
            true_labels_i = true_labels_i[true_labels_i != -100]
            generated_ids = torch.argmax(logits, dim=-1).cpu()
            generated_ids = generated_ids[
                i,
                self.config.cadcoder.num_virtual_tokens : len(true_labels_i)
                + self.config.cadcoder.num_virtual_tokens,
            ]

            # add beginning token from true code
            if (
                self.config.general.model == "cadcoder-seqcompl"
                or self.config.general.model == "cadcoder-image"
            ):
                generated_ids = torch.cat(
                    (
                        true_labels[i, self.config.cadcoder.num_virtual_tokens]
                        .unsqueeze(dim=0)
                        .cpu(),
                        generated_ids.cpu(),
                    ),
                    dim=0,
                )

            generated_code = pl_module.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            true_code = pl_module.tokenizer.batch_decode(
                true_labels_i, skip_special_tokens=True
            )
            del true_labels_i, generated_ids

            generated_code = (
                "".join(generated_code) if generated_code else "INVALID CODE"
            )

            true_code = "".join(true_code) if true_code else "INVALID CODE"
            if i == 0:
                print("")  # new line for better readability
                self.console.print(
                    Panel(
                        Syntax(generated_code[:1000], "python", line_numbers=True),
                        title=f"Generated Code Sample ({phase})",
                        border_style="blue",
                    )
                )

                self.console.print(
                    Panel(
                        Syntax(true_code[:1000], "python", line_numbers=True),
                        title=f"True Code Sample ({phase})",
                        border_style="green",
                    )
                )

            # evaluate metrics
            eval = CadEvaluator(sampler_points=500, threshold=0.05)
            result = eval.evaluate_cadquery(code_gt=true_code, code_pred=generated_code)

            if result["invalid_reason"] is not None:
                invalid_sum += 1
                is_valid = False
            else:
                valid_sum += 1
                is_valid = True

            prefix = f"{phase}/"
            invalid_ratio = invalid_sum / (valid_sum + invalid_sum)
            id = ids[i]
            pl_module.log(f"{prefix}invalid_ratio", invalid_ratio, on_epoch=True)
            for key in [
                "chamfer",
                "fscore",
                "normal_consistency",
                "codebleu",
                "intersection_over_union",
            ]:
                if result.get(key) is not None:
                    pl_module.log(
                        f"{prefix}{key if key != 'normal_consistency' else 'nc'}",
                        result[key],
                        rank_zero_only=True,
                        on_epoch=True,
                    )
            logged_data = {
                "epoch": trainer.current_epoch,
                "batch": i,
                "generated_code": generated_code,
                "true_code": true_code,
                "chamfer_distance": result["chamfer"],
                "fscore": result["fscore"],
                "codebleu": result["codebleu"],
                "intersection_over_union": result["intersection_over_union"],
                "normal_consistency": result["normal_consistency"],
                "is_valid": is_valid,
                "id": id,
                "invalid_reason": result.get("invalid_reason", "None"),
            }
            if phase == "train":
                self.log_one_sample_table.add_data(
                    trainer.current_epoch, i, is_valid, id, generated_code, true_code
                )
                self.log_one_sample_data.append(logged_data)
            elif phase == "val":
                self.log_one_sample_table_val.add_data(
                    trainer.current_epoch, i, is_valid, id, generated_code, true_code
                )
                self.log_one_sample_data_val.append(logged_data)
        # Save intermediate to JSON file
        results_data = {
            "train_samples": self.log_one_sample_data,
            "val_samples": self.log_one_sample_data_val,
        }
        run_name = getattr(trainer.logger.experiment, "name", "experiment")
        base_dir = (
            pathlib.Path(__file__).parent.parent
            / "results"
            / self.config.general.model
            / run_name
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        json_file = base_dir / f"epoch_{trainer.current_epoch}_batch_{batch_idx}.json"
        with open(json_file, "w") as f:
            json.dump(results_data, f, indent=2)
            print(f"Saved intermediate results to {json_file}")

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.is_global_zero:
            console = Console()
            table = Table(show_lines=True)

            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")

            # Add metrics from callback_metrics
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    table.add_row(key, f"{value.item():.4f}")

            # Add learning rate
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            table.add_row("trainer/learning_rate", f"{lr:.4e}")
            print("")
            console.print(table)

            # Log learning rate to logger
            trainer.logger.log_metrics({"trainer/learning_rate": lr})

            # Log one sample to visualize training progress
            # Decode logits for debugging
            if (
                getattr(pl_module, "output_logits", None) is not None
                and trainer.current_epoch % 3 == 0
                and trainer.is_global_zero
            ):
                logits = pl_module.output_logits
                true_labels = pl_module.outputs_ground_truth
                self._log_generated_sample(
                    trainer,
                    pl_module,
                    phase="train",
                    logits=logits,
                    true_labels=true_labels,
                    ids=pl_module.outputs_ids,
                )
                del pl_module.output_logits, true_labels, logits

                # Get experiment run name from wandb or config
                run_name = getattr(trainer.logger.experiment, "name", "experiment")

                # Save to JSON file
                results_data = {
                    "train_samples": self.log_one_sample_data,
                    "val_samples": self.log_one_sample_data_val,
                }
                base_dir = (
                    pathlib.Path(__file__).parent.parent
                    / "results"
                    / self.config.general.model
                )
                base_dir.mkdir(parents=True, exist_ok=True)
                json_file = base_dir / f"{run_name}_epoch_{trainer.current_epoch}.json"
                with open(json_file, "w") as f:
                    json.dump(results_data, f, indent=2)
                    print(f"Saved intermediate results to {json_file}")

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if (
            getattr(pl_module, "output_logits_val", None) is not None
            and trainer.current_epoch % 3 == 0
        ):
            logits = pl_module.output_logits_val
            true_labels = pl_module.outputs_ground_truth_val
            self._log_generated_sample(
                trainer,
                pl_module,
                phase="val",
                logits=logits,
                true_labels=true_labels,
                ids=pl_module.outputs_ids,
            )
            del pl_module.output_logits_val, true_labels, logits

    def on_train_start(self, trainer, pl_module):
        print(
            f"Train Dataset Size {len(trainer.train_dataloader.dataset)}, Val Dataset Size {len(trainer.val_dataloaders.dataset)}"
        )
        print(
            f"Number of train batches {len(trainer.train_dataloader)}, Number of val batches {len(trainer.val_dataloaders)}"
        )

    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        if isinstance(exception, KeyboardInterrupt) or isinstance(exception, OSError):
            self.on_train_end(trainer, pl_module)
        else:
            # dont sync with wandb
            wandb.finish(exit_code=255)

    def on_train_end(self, trainer, pl_module):
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                trainer.logger.log_metrics(metrics={key: value.item()})

        if hasattr(trainer.model, "token_loss_buffer"):
            if isinstance(trainer.model.token_loss_buffer, dict):
                for i, token_loss in trainer.model.token_loss_buffer.items():
                    token_ids = token_loss[0]
                    losses = token_loss[1]

                    unique_ids = torch.unique(token_ids)
                    loss_per_token_id = {
                        int(token_id): losses[token_ids == token_id].mean().item()
                        for token_id in unique_ids
                    }

                    wandb.log(
                        {
                            f"loss_per_token/{token}": loss
                            for token, loss in loss_per_token_id.items()
                        }
                    )

                    wandb.log(
                        {
                            f"loss_per_token_{i}": wandb.plot.bar(
                                wandb.Table(
                                    data=[[k, v] for k, v in loss_per_token_id.items()],
                                    columns=["token_id", "avg_loss"],
                                ),
                                "token_id",
                                "avg_loss",
                                title=f"Loss per Token ID for Projection Layer {i}",
                            )
                        }
                    )
            else:
                token_ids = trainer.model.token_loss_buffer[0]
                losses = trainer.model.token_loss_buffer[1]

                unique_ids = torch.unique(token_ids)
                loss_per_token_id = {
                    int(token_id): losses[token_ids == token_id].mean().item()
                    for token_id in unique_ids
                }

                trainer.logger.log(
                    {
                        "loss_per_token": wandb.plot.bar(
                            wandb.Table(
                                data=[[k, v] for k, v in loss_per_token_id.items()],
                                columns=["token_id", "avg_loss"],
                            ),
                            "token_id",
                            "avg_loss",
                            title="Loss per Token ID",
                        )
                    }
                )

                trainer.logger.save()

        if self.config.logging.log_one_sample:
            trainer.logger.experiment.log({"generated_code": self.log_one_sample_table})
            trainer.logger.experiment.log(
                {"generated_code_val": self.log_one_sample_table_val}
            )
            # also log to file system
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
