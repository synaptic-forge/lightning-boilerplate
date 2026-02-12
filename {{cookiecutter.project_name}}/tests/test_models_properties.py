import inspect
import pkgutil
import importlib
import torch
import lightning as L
from src import models

def test_models_have_example_input_array():
    missing_attr = []

    for loader, module_name, is_pkg in pkgutil.iter_modules(models.__path__):
        full_module_name = f"{models.__name__}.{module_name}"
        module = importlib.import_module(full_module_name)

        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, L.LightningModule) and cls.__module__ == module.__name__:
                try:
                    instance = cls(criterion=torch.nn.CrossEntropyLoss())
                except Exception as e:
                    missing_attr.append(f"{cls.__name__} (from {full_module_name}) -- failed to init: {e}")
                    continue

                if not hasattr(instance, "example_input_array") or not isinstance(instance.example_input_array, torch.Tensor):
                    missing_attr.append(f"{cls.__name__} (from {full_module_name}) -- missing example_input_array or wrong type")

    assert not missing_attr, (
        "The following LightningModule classes are missing 'example_input_array':\n"
        + "\n".join(missing_attr)
    )
