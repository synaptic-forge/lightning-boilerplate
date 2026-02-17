import logging, warnings, torch, os
from src.models import *
from src.datasets import *
warnings.filterwarnings("ignore")
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf
from datetime import datetime
from dotenv import load_dotenv
from src.utils.calibration import GeneralCalibrationDataReader
from src.callbacks.export import ONNXExportCallback

OmegaConf.register_new_resolver(
    "timestamp",
    lambda fmt="%Y%m%d_%H%M%S": datetime.now().strftime(fmt),
)

class CustomLightningCLI(LightningCLI):
    def before_fit(self):
        config_file = getattr(self.config, "config", None)
        if isinstance(self.trainer.logger, MLFlowLogger):
            if config_file and os.path.exists(config_file):
                experiment = self.trainer.logger.experiment
                experiment.log_artifact(
                    run_id = self.trainer.logger.run_id,
                    local_path=os.path.basename(config_file),
                )
                logging.info(f"✓ Logged config file to Comet: {config_file}")

def cli_main():
    
    if os.path.exists(".env"):
        load_dotenv(".env")
        logging.info("Loaded .env")
    elif os.path.exists(".env.example"):
        load_dotenv(".env.example")
        logging.info("Loaded .env.example")
    else:
        logging.error("No .env or .env.example file found")
        
    project_name = "{{cookiecutter.project_name}}"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} Model Training - %(levelname)s - %(message)s'
    )
    
    cli = CustomLightningCLI(
        run=True,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"}
    )

if __name__ == "__main__":
    cli_main()