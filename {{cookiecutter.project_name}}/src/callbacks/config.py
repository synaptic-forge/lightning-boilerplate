import os
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger

class MLflowSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage: str) -> None:
        super().save_config(trainer, pl_module, stage)

        if isinstance(trainer.logger, MLFlowLogger):
            config_path = os.path.join(trainer.default_root_dir, "config.yaml")

            if os.path.exists(config_path):
                trainer.logger.experiment.log_artifact(
                    trainer.logger.run_id,
                    config_path,
                    artifact_path="configs"
                )

                print(f"✓ Logged config to MLflow: {config_path}")