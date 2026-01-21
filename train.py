import argparse
import dataclasses
import os
import pathlib

# Disable tokenizer parallelism warning when using with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from data_loader.brep_to_rapidcadpy_dataset import BrepToPyDataset
from modules.custom_logging import LoggingCallbackWandb


class BaseTrainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        tags = [self.config.mode]
        if config.fast_dev_run:
            tags.append("debug")
        if config.logger_tag:
            tags.append(config.logger_tag)
        if config.logger == "wandb":
            os.environ["WANDB_MODE"] = config.logger_mode
            os.environ["WANDB_DISABLE_CODE"] = "true"
            os.environ["WANDB_DISABLE_PIP"] = "1"  # no need for PIP scanning
            self.logger = WandbLogger(
                project=config.model,
                log_model=False,
                tags=tags,
                config=dataclasses.asdict(config),
            )
            try:
                self.logger.log_hyperparams(dataclasses.asdict(config))
            except Exception:
                pass
        elif config.logger == "mlflow":
            self.logger = MLFlowLogger(
                tracking_uri=str(pathlib.Path(__file__).parent / "logs" / "mlruns"),
                experiment_name="cadcoder",
                run_name="run-1",
            )
        elif config.logger == "tensorboard":
            self.logger = TensorBoardLogger(
                save_dir=str(pathlib.Path(__file__).parent / "logs" / "tensorboard"),
                name=config.model,
            )
            try:
                self.logger.experiment.add_hparams(
                    dataclasses.asdict(config),
                    {},
                )
            except AttributeError:
                # not available in dummy run
                pass

        # 1) Initialize the model and dataloader
        if config.model == "cadcoder-brep":
            from models import BrepCadcoder

            self.model = BrepCadcoder(config)

            self.train_dataloader, self.val_dataloader, self.test_dataloader = (
                BrepToPyDataset.create_splits(
                    config,
                )
            )
        elif config.model == "cadcoder-text":
            from models import TextCadcoder

            self.model = TextCadcoder(config)
            from data_loader.text_to_rapidcadpy_dataset import TextToPyDataset

            self.train_dataloader, self.val_dataloader, self.test_dataloader = (
                TextToPyDataset.create_splits(config)
            )
        elif config.model == "cadcoder-image":
            from models import CadcoderImage

            self.model = CadcoderImage(config)
            from data_loader.image_to_rapidcadpy_dataset import ImageToPyDataset

            self.train_dataloader, self.val_dataloader, self.test_dataloader = (
                ImageToPyDataset.create_splits(config)
            )
        elif config.model == "cadcoder-seqcompl":
            from models import CadcoderSeqCompletion

            self.model = CadcoderSeqCompletion(config)
            from data_loader.shaft_dataset import RapidcadpyShaftDataset

            self.train_dataloader, self.val_dataloader, self.test_dataloader = (
                RapidcadpyShaftDataset.create_splits(config)
            )
        elif config.model == "cadcoder-deepseek":
            from models.cadcoder_shaft import DeepSeekCodeCompletion

            self.model = DeepSeekCodeCompletion(config)
            from data_loader.prompt_completion_dataset import PromptCompletionDataset

            self.train_dataloader, self.val_dataloader, self.test_dataloader = (
                PromptCompletionDataset.create_splits(config)
            )
        else:
            raise NotImplementedError("Model not implemented")
        if config.logger == "wandb":
            ...
            # self.logger.watch(self.model.prefix_encoder, log="all", log_graph=False)

        # 2) Load the checkpoint if provided
        if config.checkpoint:
            self.checkpoint_path = str(
                pathlib.Path(__file__).parent
                / "checkpoints"
                / config.model
                / config.checkpoint
            )
        else:
            self.checkpoint_path = None

        # 4) enable tensor core usage
        torch.set_float32_matmul_precision("medium")

        callbacks: list = [
            LoggingCallbackWandb(config),
            EarlyStopping(
                monitor="train/total_loss",
                patience=config.patience,
                mode="min",
                verbose=True,
                min_delta=config.min_delta,
            ),
        ]
        if config.checkpointing:
            file_name = f"model-{self.logger.experiment.name}"
            callbacks.append(
                ModelCheckpoint(
                    dirpath=f"checkpoints/{config.model}",
                    filename=file_name,
                    every_n_epochs=self.config.training.checkpoint_freq,
                    save_top_k=1,
                    verbose=True,
                ),
            )

        if config.multi_gpu:
            self.trainer = L.Trainer(
                accelerator="cuda",
                devices=config.num_gpus,
                num_nodes=config.num_nodes,
                strategy="ddp_find_unused_parameters_true",
                max_epochs=self.config.max_epochs,
                logger=[self.logger],
                log_every_n_steps=self.config.log_every_n_steps,
                callbacks=callbacks,
                val_check_interval=1.0,
                accumulate_grad_batches=self.config.accumulate_grad_batches,
                fast_dev_run=self.config.fast_dev_run,
                precision="bf16-mixed",
                enable_checkpointing=self.config.checkpointing,
            )
        else:
            self.trainer = L.Trainer(
                accelerator="cuda",
                devices=[0],
                max_epochs=self.config.max_epochs,
                logger=[self.logger],
                log_every_n_steps=self.config.log_every_n_steps,
                callbacks=callbacks,
                gradient_clip_val=self.config.gradient_clip_val,
                accumulate_grad_batches=self.config.accumulate_grad_batches,
                fast_dev_run=self.config.fast_dev_run,
                precision="bf16-mixed",
                enable_checkpointing=self.config.checkpointing,
                num_sanity_val_steps=1,
                val_check_interval=1.0,
                check_val_every_n_epoch=1,
            )

    def train(self) -> None:
        if self.checkpoint_path:
            self.model = self.model.__class__.load_from_checkpoint(
                self.checkpoint_path, strict=False, config=self.config
            )
        self.trainer.fit(
            self.model,
            self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
        # self.trainer.test(
        #     model=self.model,
        #     dataloaders=self.test_dataloader,
        # )

    def evaluate(self) -> None:
        if self.config.checkpoint is None:
            ckpt_path = None
        else:
            ckpt_path = str(
                pathlib.Path(__file__).parent
                / "checkpoints"
                / self.config.model
                / self.config.checkpoint
            )
        self.model = self.model.__class__.load_from_checkpoint(
            ckpt_path, config=self.config
        )
        self.trainer.test(
            model=self.model,
            dataloaders=self.test_dataloader,
        )

    def tune(self, trial):
        # Suggest hyperparameters
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        dropout = trial.suggest_uniform("dropout", 0.1, 0.2)

        # Build config
        self.config.learning_rate = lr
        self.config.dropout = dropout
        tuned_model = self.model.__class__(self.config)

        # Train and validate
        self.trainer.fit(tuned_model, self.train_dataloader, self.val_dataloader)
        try:
            loss = self.trainer.callback_metrics["ptl/val_loss"].item()
        except KeyError:
            loss = torch.tensor([10.0], requires_grad=True).to(self.config.device)
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--brep2cad",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--text2cad",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--cadcoderbrep",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--cadcodertext",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--cadcoderimage",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--cadcoderseq",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--cadcodershaft",
        default=None,
        action="store_true",
    )
    args = parser.parse_args()
    base_config = pathlib.Path(__file__).parent / "config.yaml"
    if args.cadcoderbrep is not None:
        config_file = pathlib.Path(__file__).parent / "config" / "cadcoderbrep.yaml"
    elif args.cadcodertext is not None:
        config_file = pathlib.Path(__file__).parent / "config" / "cadcodertext.yaml"
    elif args.cadcoderimage is not None:
        config_file = pathlib.Path(__file__).parent / "config" / "cadcoderimage.yaml"
    elif args.cadcoderseq is not None:
        config_file = pathlib.Path(__file__).parent / "config" / "cadcoderseqcompl.yaml"
    elif args.cadcodershaft is not None:
        config_file = pathlib.Path(__file__).parent / "config" / "cadcoder-shaft.yaml"
    else:
        raise ValueError("No config file specified")
    base_cfg = OmegaConf.load(base_config)
    override_cfg = OmegaConf.load(config_file)

    merged_cfg = OmegaConf.merge(base_cfg, override_cfg)

    # Convert to nested dictionaries for the hierarchical config
    config_dict = OmegaConf.to_object(merged_cfg)

    # Create the hierarchical config object
    from models.config import (
        BrepEncoderConfig,
        CadcoderConfig,
        Config,
        DataConfig,
        DecoderConfig,
        DeepspeedConfig,
        EncoderConfig,
        GeneralConfig,
        InferenceConfig,
        LoggingConfig,
        LossConfig,
        MemoryConfig,
        OptimizerConfig,
        Text2CadConfig,
        TrainingConfig,
    )

    # Create sub-configs from the dictionary sections
    config_kwargs = {}

    if "general" in config_dict:
        config_kwargs["general"] = GeneralConfig(**config_dict["general"])
    if "encoder" in config_dict:
        config_kwargs["encoder"] = EncoderConfig(**config_dict["encoder"])
    if "decoder" in config_dict:
        config_kwargs["decoder"] = DecoderConfig(**config_dict["decoder"])
    if "optimizer" in config_dict:
        config_kwargs["optimizer"] = OptimizerConfig(**config_dict["optimizer"])
    if "training" in config_dict:
        config_kwargs["training"] = TrainingConfig(**config_dict["training"])
    if "deepspeed" in config_dict:
        config_kwargs["deepspeed"] = DeepspeedConfig(**config_dict["deepspeed"])
    if "loss" in config_dict:
        config_kwargs["loss"] = LossConfig(**config_dict["loss"])
    if "data" in config_dict:
        config_kwargs["data"] = DataConfig(**config_dict["data"])
    if "inference" in config_dict:
        config_kwargs["inference"] = InferenceConfig(**config_dict["inference"])
    if "logging" in config_dict:
        config_kwargs["logging"] = LoggingConfig(**config_dict["logging"])
    if "brep_encoder" in config_dict:
        config_kwargs["brep_encoder"] = BrepEncoderConfig(**config_dict["brep_encoder"])
    if "text2cad" in config_dict:
        config_kwargs["text2cad"] = Text2CadConfig(**config_dict["text2cad"])
    if "cadcoder" in config_dict:
        config_kwargs["cadcoder"] = CadcoderConfig(**config_dict["cadcoder"])
    if "memory" in config_dict:
        config_kwargs["memory"] = MemoryConfig(**config_dict["memory"])

    c = Config(**config_kwargs)
    trainer = BaseTrainer(c)
    if c.mode == "train":
        trainer.train()
    elif c.mode == "tune":
        import optuna

        study = optuna.create_study(direction="minimize")
        study.optimize(trainer.tune, n_trials=10)
        print("Best trial:", study.best_trial)
    elif c.mode == "evaluate":
        trainer.evaluate()
