
class TestMNISTModel:
    def test_forward_pass(self, mnist_model, sample_image_batch):
        """Test that forward pass returns correct output shape."""
        output = mnist_model(sample_image_batch)
        assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"


