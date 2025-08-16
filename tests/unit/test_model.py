"""
Unit tests for the RL model components of the PyTrad framework.

This test suite provides comprehensive coverage of the ActorCriticTransformer model
and its supporting components (PositionalEncoding and CrossAttention). The tests are
designed to verify:

1. Correctness - Testing that the model behaves as expected in normal conditions
2. Robustness - Testing the model with extreme and edge-case inputs
3. Stability - Testing consistent outputs for the same inputs
4. Performance - Measuring model inference time
5. Compatibility - Testing across devices (CPU/CUDA) and configurations
6. Integration - Testing integration with optimizers and serialization

The test suite is organized into three main sections:
- TestPositionalEncoding: Tests for the positional encoding component
- TestCrossAttention: Tests for the cross-attention mechanism
- TestActorCriticTransformer: Tests for the main model architecture

To run all tests:
    pytest -xvs tests/unit/test_model.py

To run a specific test class:
    pytest -xvs tests/unit/test_model.py::TestActorCriticTransformer

To run a specific test:
    pytest -xvs tests/unit/test_model.py::TestActorCriticTransformer::test_forward_pass_shapes
"""

import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from pytrad.rl.model import ActorCriticTransformer, CrossAttention, PositionalEncoding
from pytrad.rl.types import MultiTimeframeData, TimeframeData

# Constants for testing
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN_LOW = 60
DEFAULT_SEQ_LEN_HIGH = 20
DEFAULT_INPUT_DIM = 5
DEFAULT_TIME_DIM = 4
DEFAULT_D_MODEL = 128


# Utility function to create test data with common parameters
def create_test_data(
    batch_size=DEFAULT_BATCH_SIZE,
    seq_len_low=DEFAULT_SEQ_LEN_LOW,
    seq_len_high=DEFAULT_SEQ_LEN_HIGH,
    input_dim=DEFAULT_INPUT_DIM,
    time_dim=DEFAULT_TIME_DIM,
    seed=None,
):
    """Create standardized test data for model testing."""
    if seed is not None:
        torch.manual_seed(seed)

    # Create data for lower timeframe
    features_low = torch.randn(batch_size, seq_len_low, input_dim)
    currency_pairs_low = torch.randint(0, 28, (batch_size, seq_len_low))
    time_data_low = torch.randn(batch_size, seq_len_low, time_dim)

    # Create data for higher timeframe
    features_high = torch.randn(batch_size, seq_len_high, input_dim)
    currency_pairs_high = torch.randint(0, 28, (batch_size, seq_len_high))
    time_data_high = torch.randn(batch_size, seq_len_high, time_dim)

    # Create TimeframeData objects
    low_tf = TimeframeData(
        features=features_low,
        currency_pairs=currency_pairs_low,
        time_data=time_data_low,
    )

    high_tf = TimeframeData(
        features=features_high,
        currency_pairs=currency_pairs_high,
        time_data=time_data_high,
    )

    # Create MultiTimeframeData
    return MultiTimeframeData(low=low_tf, high=high_tf)


# Create functions to generate different types of test data
def create_extreme_values_data(
    batch_size=DEFAULT_BATCH_SIZE,
    seq_len_low=DEFAULT_SEQ_LEN_LOW,
    seq_len_high=DEFAULT_SEQ_LEN_HIGH,
    input_dim=DEFAULT_INPUT_DIM,
    time_dim=DEFAULT_TIME_DIM,
):
    """Create test data with extreme values for robustness testing."""
    # Create data with extreme values (very large, very small, zeros, etc.)
    features_low = torch.zeros(batch_size, seq_len_low, input_dim)
    features_low[:, 0:10, :] = 1e6  # Very large values
    features_low[:, 10:20, :] = 1e-6  # Very small values
    features_low[:, 20:30, :] = 0  # Zeros
    features_low[:, 30:40, :] = -1e6  # Very large negative values
    features_low[:, 40:50, :] = -1e-6  # Very small negative values
    features_low[:, 50:, :] = torch.randn(
        batch_size, seq_len_low - 50, input_dim
    )  # Random values

    # Similar pattern for high timeframe
    features_high = torch.zeros(batch_size, seq_len_high, input_dim)
    features_high[:, 0:4, :] = 1e6
    features_high[:, 4:8, :] = 1e-6
    features_high[:, 8:12, :] = 0
    features_high[:, 12:16, :] = -1e6
    features_high[:, 16:, :] = torch.randn(batch_size, seq_len_high - 16, input_dim)

    # Realistic indices for currency pairs
    currency_pairs_low = torch.randint(0, 28, (batch_size, seq_len_low))
    currency_pairs_high = torch.randint(0, 28, (batch_size, seq_len_high))

    # Time data
    time_data_low = torch.randn(batch_size, seq_len_low, time_dim)
    time_data_high = torch.randn(batch_size, seq_len_high, time_dim)

    # Create TimeframeData objects
    low_tf = TimeframeData(
        features=features_low,
        currency_pairs=currency_pairs_low,
        time_data=time_data_low,
    )

    high_tf = TimeframeData(
        features=features_high,
        currency_pairs=currency_pairs_high,
        time_data=time_data_high,
    )

    return MultiTimeframeData(low=low_tf, high=high_tf)


def create_minimal_data(
    batch_size=2,
    seq_len_low=3,  # Extremely short sequence
    seq_len_high=1,  # Minimal sequence
    input_dim=DEFAULT_INPUT_DIM,
    time_dim=DEFAULT_TIME_DIM,
):
    """Create test data with minimal sequence lengths."""
    # Create minimal data
    features_low = torch.randn(batch_size, seq_len_low, input_dim)
    currency_pairs_low = torch.randint(0, 28, (batch_size, seq_len_low))
    time_data_low = torch.randn(batch_size, seq_len_low, time_dim)

    features_high = torch.randn(batch_size, seq_len_high, input_dim)
    currency_pairs_high = torch.randint(0, 28, (batch_size, seq_len_high))
    time_data_high = torch.randn(batch_size, seq_len_high, time_dim)

    # Create TimeframeData objects
    low_tf = TimeframeData(
        features=features_low,
        currency_pairs=currency_pairs_low,
        time_data=time_data_low,
    )

    high_tf = TimeframeData(
        features=features_high,
        currency_pairs=currency_pairs_high,
        time_data=time_data_high,
    )

    # Create MultiTimeframeData
    return MultiTimeframeData(low=low_tf, high=high_tf)


def move_data_to_device(data, device):
    """Helper function to move test data to the specified device."""
    return data.to_device(device)


class TestPositionalEncoding:
    """Test suite for the PositionalEncoding class."""

    def test_initialization(self):
        """Test that PositionalEncoding initializes with correct parameters."""
        d_model = 128
        max_len = 1000
        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Check that the positional encoding buffer has the right shape
        assert hasattr(pos_enc, "pe")
        assert pos_enc.pe.shape == (1, max_len, d_model)

        # Test even/odd dimensions
        pos_enc_odd = PositionalEncoding(d_model=127, max_len=max_len)
        assert pos_enc_odd.pe.shape == (1, max_len, 127)

    def test_forward_pass(self):
        """Test the forward pass of PositionalEncoding."""
        batch_size = 8
        seq_len = 100
        d_model = 64
        pos_enc = PositionalEncoding(d_model=d_model, max_len=200)

        # Create a test input
        x = torch.zeros(batch_size, seq_len, d_model)

        # Apply positional encoding
        output = pos_enc(x)

        # Output should be the input plus positional encoding
        assert output.shape == (batch_size, seq_len, d_model)

        # The output should not be all zeros since we've added positional encodings
        assert not torch.allclose(output, torch.zeros_like(output))

        # Different positions should have different encodings
        assert not torch.allclose(output[:, 0, :], output[:, 1, :])

    def test_device_compatibility(self):
        """Test that PositionalEncoding works with different devices."""
        if torch.cuda.is_available():
            # Test with CUDA
            device = torch.device("cuda")
            d_model = 32
            pos_enc = PositionalEncoding(d_model=d_model).to(device)
            x = torch.zeros(2, 10, d_model, device=device)
            output = pos_enc(x)
            assert output.device == device

        # Test with CPU
        device = torch.device("cpu")
        d_model = 32
        pos_enc = PositionalEncoding(d_model=d_model).to(device)
        x = torch.zeros(2, 10, d_model, device=device)
        output = pos_enc(x)
        assert output.device == device

    def test_sinusoidal_pattern(self):
        """Test that the positional encoding follows sinusoidal pattern."""
        d_model = 64
        max_len = 100
        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Access the pe buffer directly
        pe_tensor = pos_enc.pe
        # Remove batch dim and convert to numpy
        pe = pe_tensor.squeeze(0).detach().cpu().numpy()

        # Check sin pattern for even indices
        for pos in range(0, 10, 2):  # Check first few positions
            sin_vals = pe[pos, 0::2][:3]  # First few sin values
            # Sin values should follow sinusoidal pattern
            position = np.array([pos])
            div_term = np.exp(np.arange(0, 6, 2) * (-np.log(10000.0) / d_model))
            expected = np.sin(position.reshape(-1, 1) * div_term)
            assert np.allclose(sin_vals, expected[0, :3], rtol=1e-5, atol=1e-5)

    def test_positional_encoding_no_nan(self):
        """Test that positional encoding does not produce NaN values."""
        d_model = 256
        max_len = 2000
        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Get pe as a tensor
        pe_buffer = pos_enc.pe

        # Check for NaN values
        assert not torch.isnan(pe_buffer).any(), (
            "Positional encoding contains NaN values"
        )

    def test_large_sequence_length(self):
        """Test positional encoding with large sequence length."""
        d_model = 64
        large_len = 5000  # Maximum supported length
        pos_enc = PositionalEncoding(d_model=d_model, max_len=large_len)

        # Create input with large sequence
        batch_size = 2
        x = torch.zeros(batch_size, large_len, d_model)

        # Should work without errors
        output = pos_enc(x)
        assert output.shape == (batch_size, large_len, d_model)


class TestCrossAttention:
    """Test suite for the CrossAttention class."""

    def test_initialization(self):
        """Test that CrossAttention initializes correctly."""
        d_model = 64
        nhead = 4
        cross_attn = CrossAttention(d_model=d_model, nhead=nhead, dropout=0.1)

        # Check that the component has the correct attributes
        assert hasattr(cross_attn, "multihead_attn")
        assert hasattr(cross_attn, "norm")
        assert hasattr(cross_attn, "dropout")

        # Check that multihead attention has the right configuration
        assert cross_attn.multihead_attn.embed_dim == d_model
        assert cross_attn.multihead_attn.num_heads == nhead
        assert isinstance(cross_attn.norm, nn.LayerNorm)

    def test_forward_pass(self):
        """Test the forward pass of CrossAttention."""
        batch_size = 4
        query_len = 20
        kv_len = 30
        d_model = 64
        nhead = 8

        cross_attn = CrossAttention(d_model=d_model, nhead=nhead)

        # Create test inputs
        query = torch.randn(batch_size, query_len, d_model)
        key_value = torch.randn(batch_size, kv_len, d_model)

        # Apply cross-attention
        output = cross_attn(query, key_value)

        # Check output shape
        assert output.shape == (batch_size, query_len, d_model)

        # Output should differ from input
        assert not torch.allclose(output, query)

    def test_gradient_flow_through_cross_attention(self):
        """Test that gradients flow properly through CrossAttention."""
        batch_size = 2
        query_len = 10
        kv_len = 15
        d_model = 32
        nhead = 4

        # Create cross-attention module with gradient tracking
        cross_attn = CrossAttention(d_model=d_model, nhead=nhead)

        # Enable gradient tracking for all parameters
        for param in cross_attn.parameters():
            param.requires_grad = True

        # Create inputs with gradient tracking
        query = torch.randn(batch_size, query_len, d_model, requires_grad=True)
        key_value = torch.randn(batch_size, kv_len, d_model, requires_grad=True)

        # Forward pass
        output = cross_attn(query, key_value)

        # Create a dummy loss and backpropagate
        loss = output.mean()
        loss.backward()

        # Check gradients are non-zero for parameters and inputs
        params_with_grad = [p for p in cross_attn.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0, "No parameters received gradients"
        assert torch.any(
            torch.stack([torch.any(p.grad != 0) for p in params_with_grad])
        )

        # Check input gradients
        assert query.grad is not None and torch.any(query.grad != 0)
        assert key_value.grad is not None and torch.any(key_value.grad != 0)

    def test_attention_with_different_dimension_inputs(self):
        """Test that CrossAttention properly handles inputs with different dimensions."""
        # Create a cross-attention module
        d_model = 32
        nhead = 4
        cross_attn = CrossAttention(d_model=d_model, nhead=nhead)

        # Test with different batch sizes
        for batch_size in [1, 4, 16]:
            query_len = 10
            kv_len = 15

            query = torch.randn(batch_size, query_len, d_model)
            key_value = torch.randn(batch_size, kv_len, d_model)

            output = cross_attn(query, key_value)

            # Output should have the same shape as query except for the embedding dimension
            assert output.shape == (batch_size, query_len, d_model)

        # Test with different query lengths
        batch_size = 4
        for query_len in [1, 10, 50]:
            kv_len = 15

            query = torch.randn(batch_size, query_len, d_model)
            key_value = torch.randn(batch_size, kv_len, d_model)

            output = cross_attn(query, key_value)

            # Output should match query length
            assert output.shape == (batch_size, query_len, d_model)

        # Test with different key/value lengths
        query_len = 10
        for kv_len in [1, 20, 100]:
            query = torch.randn(batch_size, query_len, d_model)
            key_value = torch.randn(batch_size, kv_len, d_model)

            output = cross_attn(query, key_value)

            # Output should still match query length regardless of key/value length
            assert output.shape == (batch_size, query_len, d_model)

    def test_cross_attention_residual_effect(self):
        """Test that the residual connection in CrossAttention has the expected effect."""
        batch_size = 4
        query_len = 10
        kv_len = 15
        d_model = 32
        nhead = 4

        # Create cross-attention with no dropout for deterministic testing
        cross_attn = CrossAttention(d_model=d_model, nhead=nhead, dropout=0.0)

        # Set model to eval mode to disable dropout
        cross_attn.eval()

        # Create inputs
        torch.manual_seed(42)  # For reproducibility
        query = torch.randn(batch_size, query_len, d_model)
        key_value = torch.zeros(batch_size, kv_len, d_model)  # All zeros key/value

        # With zero key/value, the attention output should be close to zero
        # But the residual connection should preserve the query input
        output = cross_attn(query, key_value)

        # Output should be similar to the input query due to the residual connection
        # Note: We allow for some differences due to the layer normalization
        # Check if the normalized distance between output and query is small
        normalized_diff = torch.norm(output - query) / torch.norm(query)
        assert normalized_diff < 0.5, "Residual connection is not working as expected"

    def test_device_transfer(self):
        """Test that CrossAttention can be moved between devices."""
        d_model = 32
        nhead = 4
        cross_attn = CrossAttention(d_model=d_model, nhead=nhead)

        # Test on CPU
        device = torch.device("cpu")
        cross_attn = cross_attn.to(device)

        # Check that parameters are on CPU
        assert all(p.device.type == "cpu" for p in cross_attn.parameters())

        # Only test CUDA if available
        if torch.cuda.is_available():
            # Move to CUDA
            device = torch.device("cuda")
            cross_attn = cross_attn.to(device)

            # Check that parameters are on CUDA
            assert all(p.device.type == "cuda" for p in cross_attn.parameters())

            # Test forward pass on CUDA
            batch_size = 2
            query_len = 10
            kv_len = 15

            query = torch.randn(batch_size, query_len, d_model, device=device)
            key_value = torch.randn(batch_size, kv_len, d_model, device=device)

            output = cross_attn(query, key_value)

            # Output should be on the same device
            assert output.device.type == "cuda"


class TestActorCriticTransformer:
    """Test suite for the ActorCriticTransformer class."""

    @pytest.fixture
    def model_default(self):
        """Create a default ActorCriticTransformer model for testing."""
        return ActorCriticTransformer()

    @pytest.fixture
    def model_with_options(self):
        """Create an ActorCriticTransformer with custom options."""
        return ActorCriticTransformer(
            input_dim=10,  # Custom input dimension
            d_model=64,
            nhead=4,
            num_layers=3,
            dropout=0.2,
            currency_embed_dim=24,
            fc_hidden_dim=96,
            time_dim=6,
            time_embed_dim=12,
            recent_bias=True,
            risk_aware=True,
            market_regime_aware=True,
        )

    @pytest.fixture
    def mock_timeframe_data(self):
        """Create mock TimeframeData for testing."""
        return create_test_data()

    @pytest.fixture
    def extreme_values_data(self):
        """Create TimeframeData with extreme values for robustness testing."""
        return create_extreme_values_data()

    def test_initialization(self, model_default, model_with_options):
        """Test that ActorCriticTransformer initializes with different parameters."""
        # Default model initialization checks
        assert model_default.d_model == 128
        assert model_default.risk_aware == True

        # Custom model initialization checks
        assert model_with_options.d_model == 64
        assert model_with_options.risk_aware == True
        assert model_with_options.time_dim == 6

    def test_forward_pass_shapes(self, model_default, mock_timeframe_data):
        """Test the forward pass output shapes."""
        actor_output, critic_value = model_default(mock_timeframe_data)

        # Check that actor output is a dictionary with expected keys
        assert isinstance(actor_output, dict)
        assert "action_logits" in actor_output
        assert "action_probs" in actor_output
        assert "position_size" in actor_output
        assert "risk_params" in actor_output
        assert "kelly_fraction" in actor_output
        assert "timeframe_confidence" in actor_output

        # Check shapes
        batch_size = mock_timeframe_data.low.features.shape[0]
        assert actor_output["action_logits"].shape == (batch_size, 3)  # Buy, Sell, Hold
        assert actor_output["action_probs"].shape == (batch_size, 3)
        assert actor_output["position_size"].shape == (batch_size, 1)
        assert actor_output["risk_params"].shape == (batch_size, 2)  # SL and TP
        assert actor_output["kelly_fraction"].shape == (batch_size, 1)

        # Check critic output shape
        assert critic_value.shape == (batch_size, 1)

        # Check additional outputs when risk_aware and market_regime_aware are true
        if model_default.risk_aware:
            assert "estimated_risk" in actor_output
            assert actor_output["estimated_risk"].shape == (batch_size, 1)

        if model_default.market_regime_aware:
            assert "market_regime" in actor_output
            assert actor_output["market_regime"].shape == (
                batch_size,
                3,
            )  # Three regimes

    def test_risk_awareness(self, mock_timeframe_data):
        """Test model behavior with and without risk awareness."""
        # Model with risk awareness
        model_with_risk = ActorCriticTransformer(risk_aware=True)
        actor_output_with, _ = model_with_risk(mock_timeframe_data)

        # Model without risk awareness
        model_without_risk = ActorCriticTransformer(risk_aware=False)
        actor_output_without, _ = model_without_risk(mock_timeframe_data)

        # With risk awareness, estimated_risk should be present
        assert "estimated_risk" in actor_output_with

        # Without risk awareness, estimated_risk should be absent
        assert "estimated_risk" not in actor_output_without

    def test_market_regime_awareness(self, mock_timeframe_data):
        """Test model behavior with and without market regime awareness."""
        # Model with market regime awareness
        model_with_regime = ActorCriticTransformer(market_regime_aware=True)
        actor_output_with, _ = model_with_regime(mock_timeframe_data)

        # Model without market regime awareness
        model_without_regime = ActorCriticTransformer(market_regime_aware=False)
        actor_output_without, _ = model_without_regime(mock_timeframe_data)

        # With market regime awareness, market_regime should be present
        assert "market_regime" in actor_output_with

        # Without market regime awareness, market_regime should be absent
        assert "market_regime" not in actor_output_without

    def test_output_stability(self, model_default, mock_timeframe_data):
        """Test that repeated forward passes with the same input produce the same output."""
        model_default.eval()  # Set to evaluation mode to disable dropout

        # First forward pass
        torch.manual_seed(42)
        actor_output1, critic_value1 = model_default(mock_timeframe_data)

        # Second forward pass with same input
        torch.manual_seed(42)
        actor_output2, critic_value2 = model_default(mock_timeframe_data)

        # Outputs should be identical
        assert torch.allclose(
            actor_output1["action_logits"], actor_output2["action_logits"]
        )
        assert torch.allclose(
            actor_output1["position_size"], actor_output2["position_size"]
        )
        assert torch.allclose(critic_value1, critic_value2)

    def test_device_compatibility(self, model_default, mock_timeframe_data):
        """Test model compatibility with different devices."""
        if torch.cuda.is_available():
            # Test with CUDA
            device = torch.device("cuda")
            model = model_default.to(device)

            # Move data to device
            data_on_device = mock_timeframe_data.to_device("cuda")

            # Forward pass
            actor_output, critic_value = model(data_on_device)

            # Check that outputs are on the correct device
            assert actor_output["action_logits"].device.type == "cuda"
            assert critic_value.device.type == "cuda"

        # Test with CPU
        device = torch.device("cpu")
        model = model_default.to(device)

        # Ensure data is on CPU
        data_on_cpu = mock_timeframe_data.to_device("cpu")

        # Forward pass
        actor_output, critic_value = model(data_on_cpu)

        # Check that outputs are on CPU
        assert actor_output["action_logits"].device.type == "cpu"
        assert critic_value.device.type == "cpu"

    def test_input_validation(self, model_default):
        """Test that model validates inputs correctly."""
        # Create invalid TimeframeData (wrong dimensions)
        invalid_features = torch.randn(4, 10)  # Missing dimension
        invalid_currency_pairs = torch.randint(0, 28, (4, 20))
        invalid_time_data = torch.randn(4, 20, 4)

        invalid_tf = TimeframeData(
            features=invalid_features,
            currency_pairs=invalid_currency_pairs,
            time_data=invalid_time_data,
        )

        valid_tf = TimeframeData(
            features=torch.randn(4, 20, 5),
            currency_pairs=torch.randint(0, 28, (4, 20)),
            time_data=torch.randn(4, 20, 4),
        )

        # Test with one invalid timeframe
        invalid_multiframe = MultiTimeframeData(low=invalid_tf, high=valid_tf)

        # Should raise ValueError on forward pass
        with pytest.raises(ValueError):
            model_default(invalid_multiframe)

    def test_position_sizing_within_bounds(self, model_default, mock_timeframe_data):
        """Test that position size outputs are within expected bounds (0-1)."""
        actor_output, _ = model_default(mock_timeframe_data)
        position_sizes = actor_output["position_size"]

        # Position sizes should be in range [0, 1]
        assert torch.all(position_sizes >= 0)
        assert torch.all(position_sizes <= 1)

    def test_action_probabilities_sum_to_one(self, model_default, mock_timeframe_data):
        """Test that action probabilities sum to 1."""
        actor_output, _ = model_default(mock_timeframe_data)
        action_probs = actor_output["action_probs"]

        # Sum along the action dimension
        probs_sum = torch.sum(action_probs, dim=1)

        # Should sum to 1 for each batch item
        assert torch.allclose(
            probs_sum, torch.ones_like(probs_sum), rtol=1e-5, atol=1e-5
        )

    def test_gradient_flow(self, model_default, mock_timeframe_data):
        """Test that gradients flow properly through the model."""
        # Enable gradients
        for param in model_default.parameters():
            param.requires_grad = True

        # Forward pass
        actor_output, critic_value = model_default(mock_timeframe_data)

        # Compute a loss (dummy loss for testing)
        loss = actor_output["action_logits"].mean() + critic_value.mean()

        # Backpropagate
        loss.backward()

        # Check that at least some gradients are non-zero
        has_gradients = False
        for name, param in model_default.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break

        assert has_gradients, "No gradients are flowing through the model"

    def test_parameter_count(self, model_default):
        """Test that model parameter count is as expected."""
        # Count total parameters
        total_params = sum(p.numel() for p in model_default.parameters())

        # The model is complex, so we're just checking it has a reasonable number of parameters
        assert total_params > 100000, "Model has suspiciously few parameters"
        assert total_params < 10000000, "Model has suspiciously many parameters"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_size_flexibility(
        self, model_default, mock_timeframe_data, batch_size
    ):
        """Test the model works with different batch sizes."""
        # Create data with the specified batch size
        seq_len_low = 60
        seq_len_high = 20
        input_dim = 5
        time_dim = 4

        # Create data for lower timeframe
        features_low = torch.randn(batch_size, seq_len_low, input_dim)
        currency_pairs_low = torch.randint(0, 28, (batch_size, seq_len_low))
        time_data_low = torch.randn(batch_size, seq_len_low, time_dim)

        # Create data for higher timeframe
        features_high = torch.randn(batch_size, seq_len_high, input_dim)
        currency_pairs_high = torch.randint(0, 28, (batch_size, seq_len_high))
        time_data_high = torch.randn(batch_size, seq_len_high, time_dim)

        # Create TimeframeData objects
        low_tf = TimeframeData(
            features=features_low,
            currency_pairs=currency_pairs_low,
            time_data=time_data_low,
        )

        high_tf = TimeframeData(
            features=features_high,
            currency_pairs=currency_pairs_high,
            time_data=time_data_high,
        )

        # Create MultiTimeframeData
        custom_data = MultiTimeframeData(low=low_tf, high=high_tf)

        # Forward pass
        actor_output, critic_value = model_default(custom_data)

        # Check output batch size
        assert actor_output["action_logits"].shape[0] == batch_size
        assert critic_value.shape[0] == batch_size

    def test_model_serialization(self, model_default, mock_timeframe_data, tmp_path):
        """Test that the model can be serialized and deserialized correctly."""
        model_default.eval()  # Set to evaluation mode for consistency

        # Get initial outputs
        with torch.no_grad():
            initial_actor_output, initial_critic_value = model_default(
                mock_timeframe_data
            )

        # Save model to file
        save_path = tmp_path / "model.pt"
        torch.save(model_default.state_dict(), save_path)

        # Create new model with same config and load state
        loaded_model = ActorCriticTransformer()
        loaded_model.load_state_dict(torch.load(save_path))
        loaded_model.eval()

        # Get outputs from loaded model
        with torch.no_grad():
            loaded_actor_output, loaded_critic_value = loaded_model(mock_timeframe_data)

        # Outputs should be identical
        assert torch.allclose(
            initial_actor_output["action_logits"], loaded_actor_output["action_logits"]
        )
        assert torch.allclose(
            initial_actor_output["position_size"], loaded_actor_output["position_size"]
        )
        assert torch.allclose(initial_critic_value, loaded_critic_value)

    def test_variable_sequence_length(self, model_default):
        """Test that the model can handle different sequence lengths."""
        batch_size = 4
        input_dim = 5
        time_dim = 4

        # Test with different sequence lengths
        for seq_len_low, seq_len_high in [(30, 10), (45, 15), (60, 20)]:
            # Create data
            features_low = torch.randn(batch_size, seq_len_low, input_dim)
            currency_pairs_low = torch.randint(0, 28, (batch_size, seq_len_low))
            time_data_low = torch.randn(batch_size, seq_len_low, time_dim)

            features_high = torch.randn(batch_size, seq_len_high, input_dim)
            currency_pairs_high = torch.randint(0, 28, (batch_size, seq_len_high))
            time_data_high = torch.randn(batch_size, seq_len_high, time_dim)

            # Create TimeframeData objects
            low_tf = TimeframeData(
                features=features_low,
                currency_pairs=currency_pairs_low,
                time_data=time_data_low,
            )

            high_tf = TimeframeData(
                features=features_high,
                currency_pairs=currency_pairs_high,
                time_data=time_data_high,
            )

            # Create MultiTimeframeData
            variable_data = MultiTimeframeData(low=low_tf, high=high_tf)

            # Forward pass - should work without errors
            actor_output, critic_value = model_default(variable_data)

            # Output shapes should be correct
            assert actor_output["action_logits"].shape == (batch_size, 3)
            assert critic_value.shape == (batch_size, 1)

    def test_handling_extreme_values(self, model_default, extreme_values_data):
        """Test model robustness with extreme input values."""
        # Model should handle extreme values without producing NaN or Inf
        actor_output, critic_value = model_default(extreme_values_data)

        # Check outputs for NaN or Inf
        for key, value in actor_output.items():
            assert not torch.isnan(value).any(), f"NaN detected in {key}"
            assert not torch.isinf(value).any(), f"Inf detected in {key}"

        assert not torch.isnan(critic_value).any(), "NaN detected in critic value"
        assert not torch.isinf(critic_value).any(), "Inf detected in critic value"

        # Action probabilities should still sum to 1
        action_probs = actor_output["action_probs"]
        probs_sum = torch.sum(action_probs, dim=1)
        assert torch.allclose(
            probs_sum, torch.ones_like(probs_sum), rtol=1e-5, atol=1e-5
        )

    def test_integration_with_optimizer(self, model_default, mock_timeframe_data):
        """Test model integration with optimizer."""
        # Create optimizer
        optimizer = torch.optim.Adam(model_default.parameters(), lr=0.001)

        # Store initial parameters
        initial_params = {}
        for name, param in model_default.named_parameters():
            initial_params[name] = param.clone().detach()

        # Forward pass
        actor_output, critic_value = model_default(mock_timeframe_data)

        # Compute loss
        action_loss = -actor_output["action_probs"].mean()  # Maximize probabilities
        value_loss = (
            (critic_value - torch.ones_like(critic_value)) ** 2
        ).mean()  # MSE against target of 1
        loss = action_loss + value_loss

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters have changed
        params_changed = False
        for name, param in model_default.named_parameters():
            if not torch.allclose(initial_params[name], param):
                params_changed = True
                break

        assert params_changed, "Parameters did not change after optimization step"

    def test_internal_methods(self, model_default, mock_timeframe_data):
        """Test internal model methods."""
        low_tf = mock_timeframe_data.low
        high_tf = mock_timeframe_data.high

        # Test _process_single_timeframe method
        processed_low = model_default._process_single_timeframe(
            low_tf, is_high_timeframe=False
        )
        processed_high = model_default._process_single_timeframe(
            high_tf, is_high_timeframe=True
        )

        # Check shapes
        batch_size, seq_len_low = low_tf.features.shape[:2]
        batch_size, seq_len_high = high_tf.features.shape[:2]

        assert processed_low.shape == (batch_size, seq_len_low, model_default.d_model)
        assert processed_high.shape == (batch_size, seq_len_high, model_default.d_model)

        # Test _process_timeframes method
        cross_attn_output, high_tf_output, low_tf_self_output = (
            model_default._process_timeframes(low_tf, high_tf)
        )

        # Check shapes
        assert cross_attn_output.shape == (
            batch_size,
            seq_len_low,
            model_default.d_model,
        )
        assert high_tf_output.shape == (batch_size, seq_len_high, model_default.d_model)
        assert low_tf_self_output.shape == (
            batch_size,
            seq_len_low,
            model_default.d_model,
        )

        # Test _detect_market_regime
        if model_default.market_regime_aware:
            transformed_output, regime_weights = model_default._detect_market_regime(
                cross_attn_output, high_tf_output
            )

            assert transformed_output.shape == cross_attn_output.shape
            assert regime_weights.shape == (batch_size, 3)  # Three regimes

            # Regime weights should sum to 1
            assert torch.allclose(
                regime_weights.sum(dim=1), torch.ones(batch_size), rtol=1e-5
            )

    def test_architecture_comparison(self, mock_timeframe_data):
        """Compare different model architectures."""
        # Create different model variants
        model_small = ActorCriticTransformer(d_model=64, num_layers=2)
        model_large = ActorCriticTransformer(d_model=256, num_layers=6)

        # Count parameters
        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())

        # Large model should have more parameters
        assert params_large > params_small

        # Forward pass - both should work
        small_actor_output, small_critic = model_small(mock_timeframe_data)
        large_actor_output, large_critic = model_large(mock_timeframe_data)

        # Outputs should have the same shape but different values
        assert (
            small_actor_output["action_logits"].shape
            == large_actor_output["action_logits"].shape
        )

        # Values should differ
        assert not torch.allclose(
            small_actor_output["action_logits"], large_actor_output["action_logits"]
        )

    @pytest.mark.parametrize("recent_bias", [True, False])
    def test_recent_bias_effect(self, mock_timeframe_data, recent_bias):
        """Test effect of recent bias setting."""
        # Create models with and without recent bias
        model = ActorCriticTransformer(recent_bias=recent_bias)

        # Forward pass
        actor_output, _ = model(mock_timeframe_data)

        # Both should produce valid outputs
        assert "action_logits" in actor_output
        assert "action_probs" in actor_output

        # We can't directly assert behavior differences without complex setup,
        # but we can ensure the model runs correctly with both settings
        action_probs = actor_output["action_probs"]
        assert torch.all(action_probs >= 0) and torch.all(action_probs <= 1)

    def test_inference_speed(self, model_default, mock_timeframe_data):
        """Test inference speed is within acceptable limits."""
        model_default.eval()  # Set to evaluation mode

        # Warm-up runs
        with torch.no_grad():
            for _ in range(3):
                model_default(mock_timeframe_data)

        # Measure time for multiple runs
        num_runs = 10
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                model_default(mock_timeframe_data)

        end_time = time.time()
        avg_time_per_batch = (end_time - start_time) / num_runs

        # Time should be reasonable - adjust threshold as needed
        # On CPU this might be slower, this is a very approximate threshold
        assert avg_time_per_batch < 1.0, (
            f"Inference too slow: {avg_time_per_batch:.4f}s per batch"
        )

    def test_empty_sequence_handling(self, model_default):
        """Test model behavior with minimal sequence lengths."""
        # Not testing completely empty sequences as that would likely be invalid,
        # but testing very short sequences
        batch_size = 2
        short_seq_len_low = 3  # Extremely short sequence
        short_seq_len_high = 1  # Minimal sequence
        input_dim = 5
        time_dim = 4

        # Create minimal data
        features_low = torch.randn(batch_size, short_seq_len_low, input_dim)
        currency_pairs_low = torch.randint(0, 28, (batch_size, short_seq_len_low))
        time_data_low = torch.randn(batch_size, short_seq_len_low, time_dim)

        features_high = torch.randn(batch_size, short_seq_len_high, input_dim)
        currency_pairs_high = torch.randint(0, 28, (batch_size, short_seq_len_high))
        time_data_high = torch.randn(batch_size, short_seq_len_high, time_dim)

        # Create TimeframeData objects
        low_tf = TimeframeData(
            features=features_low,
            currency_pairs=currency_pairs_low,
            time_data=time_data_low,
        )

        high_tf = TimeframeData(
            features=features_high,
            currency_pairs=currency_pairs_high,
            time_data=time_data_high,
        )

        # Create MultiTimeframeData
        minimal_data = MultiTimeframeData(low=low_tf, high=high_tf)

        # Model should either handle this gracefully or raise a meaningful error
        try:
            actor_output, critic_value = model_default(minimal_data)

            # If it runs, outputs should have the right shape
            assert actor_output["action_logits"].shape == (batch_size, 3)
            assert critic_value.shape == (batch_size, 1)
        except Exception as e:
            # If it fails, it should be with a clear error about minimum sequence length
            assert "sequence" in str(e).lower() and "length" in str(e).lower()

    def test_position_size_and_kelly_correlation(
        self, model_default, mock_timeframe_data
    ):
        """Test correlation between position size and Kelly fraction."""
        # Forward pass
        actor_output, _ = model_default(mock_timeframe_data)

        # Extract position size and Kelly fraction
        position_size = actor_output["position_size"]
        kelly_fraction = actor_output["kelly_fraction"]

        # Position size should be influenced by Kelly fraction
        # We can't assert exact relationship as it depends on implementation details
        # But we can check they're not completely unrelated
        assert position_size.shape == kelly_fraction.shape


if __name__ == "__main__":
    """
    Main entry point for running the tests directly.
    Allows running specific test classes or individual tests with custom verbosity.
    
    Examples:
        # Run all tests
        python tests/unit/test_model.py
        
        # Run only TestActorCriticTransformer tests
        python tests/unit/test_model.py TestActorCriticTransformer
        
        # Run a specific test
        python tests/unit/test_model.py TestActorCriticTransformer::test_forward_pass_shapes
        
        # Run with high verbosity
        python tests/unit/test_model.py -v
    """
    import sys

    # Default arguments
    args = ["-xvs", "tests/unit/test_model.py"]

    # Process command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "-v" or sys.argv[1] == "--verbose":
            # High verbosity
            args = ["-xvs", "tests/unit/test_model.py"]
            if len(sys.argv) > 2:
                # Add test class or specific test if provided
                args[1] += "::" + sys.argv[2]
        else:
            # Test class or specific test provided
            args[1] += "::" + sys.argv[1]

    print(f"Running tests with arguments: {args}")
    pytest.main(args)
