# {{cookiecutter.project_name}}

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/Lightning-792EE5?style=for-the-badge&logo=lightning&logoColor=white)
![COmet](https://img.shields.io/badge/Comet-000000?style=for-the-badge&logo=comet&logoColor=white)
![CodeCarbon](https://img.shields.io/badge/CodeCarbon-00C853?style=for-the-badge&logo=leaflet&logoColor=white)


A simple template for building and training deep learning models using Lightning. 
This project provides a flexible and easy-to-use set of tools for rapid model development, training pipelines, and evaluation.

**Corresponding Author:** {{cookiecutter.author_name}}

-----

## ✨ Key Features

  * **Config-Driven:** Define your entire experiment—from data loading to model parameters—in a single `.yaml` file.
  * **Training Pipelines:** Robust training, validation, and testing loops right out of the box.
  * **📈 Experiment Tracking:** Automatic logging of metrics, parameters, and artifacts with [Comet ML](https://www.comet.com).
  * **🌍 Energy Tracking:** Monitor energy consumption and CO2 emissions during training using [CodeCarbon](https://github.com/mlco2/codecarbon).

-----

## 🚀 Getting Started

### 1\. Installation

First, clone the repository and install the dependencies using poetry :

```bash
git clone https://github.com/your-username/{{cookiecutter.project_name}}.git
cd {{cookiecutter.project_name}}

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry shell
```

### 2\. Configure Tracking

This template uses **CometML** for experiment tracking. To enable it, create a `.env` file from the example and add your credentials:

```bash
cp .env.example .env
nano .env

COMET_API_KEY=my_api_key
COMET_WORKSPACE=my_workspace
```

### 3. Run training, evaluation and testing

Running tasks follows the **PyTorch Lightning** workflow. Each task (training, evaluation, or testing) is fully configured via a YAML file and called with the Lightning CLI.

```bash
python main.py fit --config ./config/mnist.yaml
```

For more information about how to set the parameters for a task, please refer to the [**Lightning documentation**](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).

-----

## 📊 Tracking & Logging

### 📈 Experiment Tracking with CometML

This template is fully integrated with [CometML](https://www.comet.com). When you provide valid env variables in your `.env` file, the framework will automatically:

  * Log all hyperparameters from your `.yaml` config.
  * Track training, validation, and test metrics (e.g., loss, accuracy) in real-time.

Note that model saving is done by the ModelCheckpoint Callback to better work with a DVC integration.

### 🌍 Energy Consumption Tracking

This template uses [CodeCarbon](https://github.com/mlco2/codecarbon) to track energy consumption and estimate carbon emissions. This is implemented via a lightning callback  `CodeCarbonCallback` in `./src/callbacks/codecarbon.py`:

```python
self.tracker = EmissionsTracker(
    project_name=f"{self.project_name}",
    save_to_file=save_to_file,
    log_level=log_level,
)
```

**How it works:**

  * **Local CSV:** By default, metrics are saved locally to an `emissions.csv` file within your experiment's log directory. This file is also automatically uploaded to your Comet experiment's **Assets & Artifacts** tab.
  * **Comet Integration:** The tracker also saves summary metrics (like `total_energy_kwh` and `total_co2_emissions`) in the **Others** tab of your Comet experiment.
  * **Task-Specific Tracking:** It automatically tracks energy for two distinct tasks:
      * `training`: Energy consumed during the main trainin/validation.
      * `testing`: Energy consumed during the testing phase.

> You can customize this behavior by creating a `.codecarbon.config` file in the project's root directory. See the [CodeCarbon documentation](https://mlco2.github.io/codecarbon/usage.html#configuration) for details.


## Tests

To run tests : 

```bash
poetry run pytest
```

## TODO

- [ ] Post-Training quantization
- [ ] ONNX Export
- [ ] TensorRT Export
- [ ] Edit Readme with DVC model registration