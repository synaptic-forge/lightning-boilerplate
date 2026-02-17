# {{cookiecutter.project_name}}

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=lightning&logoColor=white)
![COmet](https://img.shields.io/badge/Comet-000000?style=for-the-badge&logo=comet&logoColor=white)
![CodeCarbon](https://img.shields.io/badge/CodeCarbon-00C853?style=for-the-badge&logo=leaflet&logoColor=white)


A simple template for building and training deep learning models using Lightning. 
This project provides a flexible and easy-to-use set of tools for rapid model development, training pipelines, and evaluation.

**Corresponding Author:** {{cookiecutter.author_name}}

## 🚀 Getting Started

### 1\. Installation


#### Install pyenv

```bash
curl -fsSL https://pyenv.run | bash
```

Export variables in ```.bashrc```

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
```

#### Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Export variables in ```.bashrc```

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verify installation:
```bash
poetry --version
```

#### Setup Project
Once the prerequisites are installed, clone the repository and set up the environment:

```bash
git clone <repository-url>
cd {{cookiecutter.project_name}}
pyenv install 3.13
pyenv virtualenv 3.13 {{cookiecutter.project_name}}
pyenv activate {{cookiecutter.project_name}}
poetry install
```

### 2\. Configure Tracking

This template uses **CometML** for experiment tracking. To enable it, create a `.env` file from the example and add your credentials:

```bash
cp .env.example .env
nano .env

COMET_API_KEY=my_api_key
COMET_WORKSPACE=my_workspace
```

### 3. Configure DVC

In this project, DVC is enabled to track model versioning.
To access the data, it should also be configured :

```bash
poetry run dvc init
poetry run dvc remote add -d dvc_storage s3://mybucket/dvcstore
poetry run dvc remote modify dvc_storage endpointurl http://localhost:9000
poetry run dvc remote modify --local dvc_storage access_key_id ****
poetry run dvc remote modify --local dvc_storage secret_access_key ****
```


### 4. Run training, evaluation and testing

Running tasks follows the **PyTorch Lightning** workflow. Each task (training, evaluation, or testing) is fully configured via a YAML file and called with the Lightning CLI.

```bash
python main.py fit --config ./config/train.yaml
```

or using DVC :

```bash
dvc repro
```

For more information about how to set the parameters for a task, please refer to the [**Lightning documentation**](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).

### 5. Introducing new models

When creating a new model as LightningModule, pytest will expect to see a ```example_input_array``` class variable that defines the expected input tensor (with batch dimension) for that model's forward pass.

```python
class DefaultNN(L.LightningModule):
    def __init__(self, criterion: nn.Module, in_channels: int = 1, out_channels: int = 10):
        super().__init__()
        self.example_input_array = torch.randn((1, 1, 28, 28))
```

## 📊 Tracking & Logging

### 📈 Experiment Tracking with MLflow

This template is fully integrated with MLflow, the open-source experiment tracking platform developed by Databricks. When you provide valid environment variables in your `.env` file (such as `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`), the framework will automatically:

* Log all hyperparameters from your `.yaml` config.
* Track training, validation, and test metrics (e.g., loss, accuracy, emissions, energy) in real time.

Model saving is handled by the `ModelCheckpoint` callback to ensure compatibility with DVC-based artifact versioning and reproducible pipelines.


### 🌍 Energy Consumption Tracking

This template uses [CodeCarbon](https://github.com/mlco2/codecarbon) to track energy consumption and estimate carbon emissions. This is implemented via a lightning callback  `CodeCarbonCallback` in `./src/callbacks/codecarbon.py`:

> You can customize codecarbon by creating a `.codecarbon.config` file in the project's root directory. See the [CodeCarbon documentation](https://mlco2.github.io/codecarbon/usage.html#configuration) for details.


## Tests

```bash
poetry run pytest
```