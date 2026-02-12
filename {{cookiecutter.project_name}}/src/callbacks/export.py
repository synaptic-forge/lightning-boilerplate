import os
from pathlib import Path
import lightning as L

class ONNXExportCallback(L.Callback):
    """
    Lightning callback to export trained models to ONNX format.
    """
    def __init__(
        self,
        output_dir: str = "./logs",
        model_name: str = "model",
        verbose: bool = True,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_onnx(self, pl_module: L.LightningDataModule) -> None:
        pl_module.eval()
        fp32_onnx_path = self.output_dir / f"{self.model_name}.onnx"
        pl_module.to_onnx(fp32_onnx_path, export_params = True)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"FP32 ONNX exported: {fp32_onnx_path}")
            self._print_model_size(fp32_onnx_path)
    
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.export_to_onnx(pl_module)

    @staticmethod
    def _print_model_size(model_path: Path) -> None:
        size_mb = os.path.getsize(model_path) / (1024 ** 2)
        print(f"  File size: {size_mb:.2f} MB")


# class ONNXExportCallback(L.Callback):
#     """
#     Lightning callback to export trained models to ONNX format.
#     Exports both FP32 and quantized (INT8) versions.
#     Supports both dynamic and static quantization with calibration data.
#     """
#     def __init__(
#         self,
#         output_dir: str = "./logs",
#         model_name: str = "model",
#         verbose: bool = True,
#         quantize: bool = True,
#         optimize: bool = True,
#         calibration_data_reader: Optional[object] = None,
#         input_name: str = "input",
#     ):
#         super().__init__()
#         self.output_dir = Path(output_dir)
#         self.model_name = model_name
#         self.verbose = verbose
#         self.quantize = quantize
#         self.optimize = optimize
#         self.calibration_data_reader = calibration_data_reader
#         self.input_name = input_name
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     def export_to_onnx(self, pl_module: L.LightningModule) -> None:
#         """Export model to ONNX format."""
#         pl_module.eval()
#         fp32_onnx_path = self.output_dir / f"{self.model_name}.onnx"
#         dummy_input = self._create_dummy_input(pl_module)
        
#         pl_module.to_onnx(
#             fp32_onnx_path,
#             input_sample=dummy_input,
#             export_params=True,
#             dynamic_axes={"input_0": {0: "batch_size"}},
#         )
        
#         if self.verbose:
#             print(f"\n{'='*60}")
#             print(f"FP32 ONNX exported: {fp32_onnx_path}")
#             self._print_model_size(fp32_onnx_path)
#             self._inspect_onnx_inputs(fp32_onnx_path)
        
#         # Optimize ONNX model if requested
#         if self.optimize:
#             optimized_path = self._optimize_onnx_model(fp32_onnx_path)
#             # Use optimized model for quantization if successful
#             model_for_quantization = optimized_path if optimized_path else fp32_onnx_path
#         else:
#             model_for_quantization = fp32_onnx_path
        
#         # Quantize ONNX model if requested
#         if self.quantize:
#             self._quantize_onnx_model(model_for_quantization)

#     def _optimize_onnx_model(self, onnx_path: Path) -> Optional[Path]:
#         """Optimize ONNX model using ONNX Runtime tools."""
#         try:
#             from onnx import load, save
            
#             optimized_path = self.output_dir / f"{self.model_name}_optimized.onnx"
            
#             # Load and save the model (applies basic optimizations)
#             model = load(str(onnx_path))
#             save(model, str(optimized_path))
            
#             if self.verbose:
#                 print(f"Optimized ONNX exported: {optimized_path}")
#                 self._print_model_size(optimized_path)
            
#             return optimized_path
            
#         except Exception as e:
#             if self.verbose:
#                 print(f"⚠️  Optimization skipped: {str(e)}")
#             return None

#     def _quantize_onnx_model(self, onnx_path: Path) -> None:
#         """Quantize ONNX model to INT8 format."""
#         try:
#             quantized_path = self.output_dir / f"{self.model_name}_quantized.onnx"
            
#             if self.calibration_data_reader is not None:
#                 # Static quantization with calibration data
#                 try:
#                     if self.verbose:
#                         print("Attempting static quantization with calibration data...")
#                     quantize_static(
#                         model_input=str(onnx_path),
#                         model_output=str(quantized_path),
#                         calibration_data_reader=self.calibration_data_reader,
#                         quant_format=QuantFormat.QDQ,
#                     )
#                 except Exception as static_error:
#                     print(f"⚠️  Static quantization failed: {str(static_error)}")
#                     print("Falling back to dynamic quantization...")
#                     quantize_dynamic(
#                         model_input=str(onnx_path),
#                         model_output=str(quantized_path),
#                         weight_type=QuantType.QInt8,
#                     )
#             else:
#                 # Dynamic quantization (no calibration needed)
#                 if self.verbose:
#                     print("Using dynamic quantization (no calibration data)...")
#                 quantize_dynamic(
#                     model_input=str(onnx_path),
#                     model_output=str(quantized_path),
#                     weight_type=QuantType.QInt8,
#                 )
            
#             if os.path.exists(quantized_path):
#                 if self.verbose:
#                     print(f"INT8 Quantized ONNX exported: {quantized_path}")
#                     self._print_model_size(quantized_path)
#                     compression_ratio = (1 - os.path.getsize(quantized_path) / os.path.getsize(str(onnx_path))) * 100
#                     print(f"  Compression ratio: {compression_ratio:.2f}%")
#             else:
#                 print(f"❌ Quantized model file not created at {quantized_path}")
#         except Exception as e:
#             import traceback
#             print(f"❌ Quantization failed: {str(e)}")
#             print(f"Traceback: {traceback.format_exc()}")

#     def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
#         """Called at the end of training."""
#         self.export_to_onnx(pl_module)
#         if self.verbose:
#             print(f"{'='*60}\n")

#     @staticmethod
#     def _print_model_size(model_path: Path) -> None:
#         """Print the file size of the model."""
#         size_mb = os.path.getsize(model_path) / (1024 ** 2)
#         print(f"  File size: {size_mb:.2f} MB")

#     @staticmethod
#     def _inspect_onnx_inputs(model_path: Path) -> None:
#         """Inspect and print ONNX model input names."""
#         try:
#             import onnx
#             model = onnx.load(str(model_path))
#             print("\n📋 ONNX Model Inputs:")
#             for input_tensor in model.graph.input:
#                 print(f"  - Name: {input_tensor.name}")
#                 print(f"    Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
#         except Exception as e:
#             print(f"Could not inspect ONNX inputs: {e}")

#     @staticmethod
#     def _create_dummy_input(pl_module: L.LightningModule) -> torch.Tensor:
#         """Create a dummy input tensor for ONNX export."""
#         import torch
        
#         # Try to infer input shape from the model
#         try:
#             # Common shapes for MNIST-like models
#             dummy_input = torch.randn(1, 1, 28, 28)
#         except:
#             # Fallback: try different shapes
#             try:
#                 dummy_input = torch.randn(1, 3, 224, 224)
#             except:
#                 dummy_input = torch.randn(1, 784)
        
#         return dummy_input