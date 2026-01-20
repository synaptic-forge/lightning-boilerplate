import logging, warnings, torch, os
from src.models import *
from src.datasets import *
warnings.filterwarnings("ignore")
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CometLogger
from omegaconf import OmegaConf
from datetime import datetime
from dotenv import load_dotenv

OmegaConf.register_new_resolver(
    "timestamp",
    lambda fmt="%Y%m%d_%H%M%S": datetime.now().strftime(fmt),
)

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)
        
    def before_fit(self):
        config_file = getattr(self.config, "config", None)
        if isinstance(self.trainer.logger, CometLogger):
            if config_file and os.path.exists(config_file):
                print('LOGGING CONFIG')
                experiment = self.trainer.logger.experiment
                experiment.log_asset(
                    config_file,
                    file_name=os.path.basename(config_file),
                    overwrite=True,
                )
                print(f"✓ Logged config file to Comet: {config_file}")

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