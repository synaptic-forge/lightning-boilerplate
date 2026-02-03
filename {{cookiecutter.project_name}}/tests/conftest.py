import pytest
from torch import Tensor
from torch.nn import CrossEntropyLoss
from src.models.default import DefaultNN

# ============================================================================
# Tensor Fixtures
# ============================================================================

@pytest.fixture
def sample_image_batch() -> Tensor:
    """
    Returns a batch of 3D image tensors.
    """
    import torch
    return torch.randn(1, 1, 28, 28)


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def mnist_model():
    """
    Creates a simple CNN
    """
    criterion = CrossEntropyLoss()
    return DefaultNN(criterion)

