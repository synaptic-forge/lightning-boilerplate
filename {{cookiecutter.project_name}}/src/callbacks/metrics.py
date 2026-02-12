import os, json
from pathlib import Path
import lightning as L

class ExportMetricsCallback(L.Callback):
    def __init__(self, output_dir="./logs"):
        self.output_path = Path(os.path.join(output_dir, 'metrics.json'))

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        clean_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if hasattr(v, "item")
        }

        with self.output_path.open("w") as f:
            json.dump(clean_metrics, f, indent=2)
