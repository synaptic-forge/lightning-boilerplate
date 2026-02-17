import logging, warnings, os
from src.models import *
from src.datasets import *
warnings.filterwarnings("ignore")
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf
from datetime import datetime
from dotenv import load_dotenv

OmegaConf.register_new_resolver(
    "timestamp",
    lambda fmt="%Y%m%d_%H%M%S": datetime.now().strftime(fmt),
)

class CustomLightningCLI(LightningCLI): 
    def before_fit(self):
        if isinstance(self.trainer.logger, MLFlowLogger):
            config_path = str(self.config.fit.config[0].abs_path)
            if os.path.exists(config_path):
                self.trainer.logger.experiment.log_artifact(
                    self.trainer.logger.run_id,
                    config_path,
                    artifact_path="configs"
                )

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