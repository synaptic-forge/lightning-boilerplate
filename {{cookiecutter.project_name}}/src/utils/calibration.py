from typing import Dict, List, Tuple, Union, Optional, Callable
import numpy as np
import torch
from torch.utils.data import DataLoader
from onnxruntime.quantization import CalibrationDataReader


class GeneralCalibrationDataReader(CalibrationDataReader):
    """
    Generic calibration data reader for any PyTorch DataLoader.
    Automatically handles various data formats and shapes.
    """
    def __init__(
        self,
        data_loader: DataLoader,
        num_samples: int = 100,
        input_names: Optional[List[str]] = None,
        preprocess_fn: Optional[Callable] = None,
    ):
        self.data_loader = data_loader
        self.num_samples = num_samples
        self.preprocess_fn = preprocess_fn
        self.data_iter = None
        self.sample_count = 0
        self._input_names = input_names

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Get next batch of calibration data."""
        if self.data_iter is None:
            self.data_iter = iter(self.data_loader)

        if self.sample_count >= self.num_samples:
            return None

        try:
            batch = next(self.data_iter)
            
            # Use custom preprocessing if provided
            if self.preprocess_fn is not None:
                processed = self.preprocess_fn(batch)
            else:
                # Default preprocessing
                processed = self._default_preprocess(batch)
            
            # Update sample count AFTER processing
            batch_size = self._get_batch_size(batch)
            self.sample_count += batch_size
            
            if self.sample_count > self.num_samples:
                # Trim to exact num_samples if we exceeded it
                for key in processed:
                    data = processed[key]
                    trim_size = self.num_samples - (self.sample_count - batch_size)
                    if trim_size > 0 and trim_size < batch_size:
                        processed[key] = data[:trim_size]
            
            return processed

        except StopIteration:
            return None

    def _default_preprocess(self, batch: Union[torch.Tensor, Tuple, List]) -> Dict[str, np.ndarray]:
        """Default preprocessing for common batch formats."""
        
        # Handle tuple/list format (images, labels) or (images, labels, ...)
        if isinstance(batch, (tuple, list)):
            # Extract only the data tensors, ignore labels
            inputs = batch[:-1] if len(batch) > 1 else batch
            
            # Single input
            if len(inputs) == 1:
                data = inputs[0]
                input_name = self._input_names[0] if self._input_names else "input"
                return {input_name: self._tensor_to_numpy(data)}
            
            # Multiple inputs
            else:
                result = {}
                for i, data in enumerate(inputs):
                    if self._input_names and i < len(self._input_names):
                        input_name = self._input_names[i]
                    else:
                        input_name = f"input_{i}"
                    result[input_name] = self._tensor_to_numpy(data)
                return result
        
        # Handle single tensor
        elif isinstance(batch, torch.Tensor):
            input_name = self._input_names[0] if self._input_names else "input"
            return {input_name: self._tensor_to_numpy(batch)}
        
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to NumPy array with correct dtype."""
        if isinstance(tensor, torch.Tensor):
            array = tensor.cpu().numpy()
        else:
            array = np.array(tensor)
        
        # Ensure float32 for inference
        if np.issubdtype(array.dtype, np.floating):
            return array.astype(np.float32)
        elif np.issubdtype(array.dtype, np.integer):
            return array.astype(np.int32)
        else:
            return array

    @staticmethod
    def _get_batch_size(batch: Union[torch.Tensor, Tuple, List]) -> int:
        """Get the batch size from various batch formats."""
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        elif isinstance(batch, (tuple, list)) and len(batch) > 0:
            first_item = batch[0]
            if isinstance(first_item, torch.Tensor):
                return first_item.shape[0]
            elif isinstance(first_item, (list, tuple)):
                return len(first_item)
        return 1

    def rewind(self) -> None:
        """Reset the data iterator."""
        self.data_iter = None
        self.sample_count = 0