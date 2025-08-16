import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants for initializing embeddings
NUM_ASSETS = 50  # Adjust according to number of different assets
NUM_MARKETS = 10  # Adjust according to number of different markets


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If odd, ensure pe has the correct shape
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added, shape (batch_size, seq_len, d_model)
        """
        # Access the buffer as a tensor properly
        pe_tensor = self.pe
        if isinstance(pe_tensor, torch.Tensor):
            pe_tensor = pe_tensor.to(x.device)
            return x + pe_tensor[:, : x.size(1), :]
        else:
            return x  # Fallback if pe is not a tensor


class CrossAttention(nn.Module):
    """
    Cross-attention module that allows one sequence to attend to another.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        Apply cross-attention where query attends to key_value.

        Args:
            query: Tensor of shape (batch_size, query_len, d_model)
            key_value: Tensor of shape (batch_size, kv_len, d_model)

        Returns:
            Tensor of shape (batch_size, query_len, d_model)
        """
        # Apply multihead attention where query attends to key_value
        attn_output, _ = self.multihead_attn(
            query=query, key=key_value, value=key_value
        )

        # Add & norm (residual connection)
        return self.norm(query + self.dropout(attn_output))


class GlobalContextFusion(nn.Module):
    """
    Fuses global context information (market and asset) with local features (timeframes).
    """

    def __init__(self, d_model, global_dim, nhead=4, dropout=0.1):
        super().__init__()
        # Projection of global context to model space
        self.global_projection = nn.Linear(global_dim, d_model)

        # Cross-attention for local features to attend to global context
        self.cross_attention = CrossAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, local_features, global_context):
        """
        Fuses local features with global context.

        Args:
            local_features: Tensor of shape (batch_size, seq_len, d_model)
            global_context: Tensor of shape (batch_size, global_dim)

        Returns:
            Fused tensor of shape (batch_size, seq_len, d_model)
        """
        # Project global context to model space
        projected_global = self.global_projection(global_context)

        # Expand global context to match sequence dimension of local features
        # (batch_size, 1, d_model) -> (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = local_features.shape
        expanded_global = projected_global.unsqueeze(1).expand(-1, seq_len, -1)

        # Apply cross-attention: local features attend to global context
        attended_features = self.cross_attention(local_features, expanded_global)

        # Concatenate and fuse
        combined = torch.cat([local_features, attended_features], dim=-1)
        fused = self.fusion_layer(combined)

        return fused


class TransformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim=8,  # OHLC returns (4) + indicators
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        asset_embed_dim=32,  # Dimension for asset embeddings
        market_embed_dim=16,  # Dimension for market embeddings
        time_dim=21,
        time_embed_dim=32,
        recent_bias=True,
        risk_aware=True,
        market_regime_aware=True,
        max_timeframes=5,
        output_dim=128,  # Dimension of output state representation
    ):
        """
        Initialize the Transformer Feature Extractor for multi-timeframe data processing.

        Args:
            input_dim: Dimension of each time step's feature vector (OHLC returns + indicators).
            d_model: Embedding dimension for the transformer.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            asset_embed_dim: Dimension for asset embeddings.
            market_embed_dim: Dimension for market embeddings.
            time_dim: Dimension of time data.
            time_embed_dim: Embedding dimension for time data.
            recent_bias: Whether to add additional attention to recent data.
            risk_aware: Whether to add risk-aware components to the model.
            market_regime_aware: Whether to detect and adapt to market regimes.
            max_timeframes: Maximum number of timeframes the model can handle.
            output_dim: Dimension of output state representation.
        """
        super().__init__()

        # Store configuration
        self.market_regime_aware = market_regime_aware
        self.risk_aware = risk_aware
        self.recent_bias = recent_bias
        self.d_model = d_model
        self.time_dim = time_dim
        self.max_timeframes = max_timeframes
        self.input_dim = input_dim
        self.asset_embed_dim = asset_embed_dim
        self.market_embed_dim = market_embed_dim
        self.output_dim = output_dim

        # Create embeddings for assets and markets
        self.asset_embedding = nn.Embedding(NUM_ASSETS, asset_embed_dim)
        self.market_embedding = nn.Embedding(NUM_MARKETS, market_embed_dim)

        # Create dynamic components for each possible timeframe
        # Input normalization for each timeframe
        self.input_norms = nn.ModuleList(
            [nn.LayerNorm(input_dim) for _ in range(max_timeframes)]
        )

        # Input embedding layers for each timeframe
        self.input_fcs = nn.ModuleList(
            [nn.Linear(input_dim, d_model) for _ in range(max_timeframes)]
        )

        # Time embedding (shared across timeframes)
        self.time_embedding = nn.Linear(time_dim, time_embed_dim)

        # Calculate total dimension after concatenation (without market/asset global)
        timeframe_total_dim = d_model + time_embed_dim

        # Projection layers to match transformer input dimension for each timeframe
        self.projections = nn.ModuleList(
            [nn.Linear(timeframe_total_dim, d_model) for _ in range(max_timeframes)]
        )

        # Positional encoding (shared architecture but separate instances)
        self.pos_encoders = nn.ModuleList(
            [PositionalEncoding(d_model) for _ in range(max_timeframes)]
        )

        # Transformer encoders for each timeframe with dynamic depth based on position
        self.transformers = nn.ModuleList()
        for i in range(max_timeframes):
            # Longer timeframes get fewer layers, shorter timeframes get more layers
            tf_layers = max(1, num_layers // (i + 1))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
            )
            self.transformers.append(
                nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
            )

        # Cross-attention mechanism for each pair of adjacent timeframes
        self.cross_attentions = nn.ModuleList(
            [
                CrossAttention(d_model=d_model, nhead=nhead, dropout=dropout)
                for _ in range(max_timeframes - 1)
            ]
        )

        # Fusion layer to combine information from multiple timeframes
        self.fusion_layer = nn.Linear(d_model * 2, d_model)

        # Global context fusion - to integrate asset and market information
        global_context_dim = asset_embed_dim + market_embed_dim
        self.global_context_fusion = GlobalContextFusion(
            d_model=d_model, global_dim=global_context_dim, nhead=4, dropout=dropout
        )

        # Recent bias layer (for giving more importance to recent data)
        if recent_bias:
            # Improved recency bias mechanism with learnable decay
            self.recency_decay = nn.Parameter(torch.ones(1) * 0.1)
            self.recency_bias = nn.Linear(d_model, d_model)

        # Market regime detection
        if market_regime_aware:
            self._init_market_regime_components(d_model)

        # Risk-aware component
        if risk_aware:
            self._init_risk_components(d_model)

        # Attention pooling instead of simple average pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1), nn.Softmax(dim=1)
        )

        # Output projection layer to desired state dimension
        self.output_projection = nn.Linear(d_model, output_dim)

    def _init_market_regime_components(self, d_model):
        """Initialize components for market regime detection and adaptation."""
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),  # Trending up, trending down, ranging
            nn.Softmax(dim=-1),
        )
        # Regime-specific processing
        self.regime_adapters = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )

    def _init_risk_components(self, d_model):
        """Initialize components for risk estimation and adjustment."""
        # Enhanced risk estimation with multiple components
        self.volatility_estimator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        # Drawdown risk estimator
        self.drawdown_estimator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        # Combine risk metrics
        self.risk_combiner = nn.Linear(2, 1)

    def _process_single_timeframe(self, timeframe_data, tf_idx):
        """
        Process a single timeframe's data through the initial layers.

        Args:
            timeframe_data: Dictionary containing all tensors for a single timeframe
                           (ohlc_ret, indicators, time, asset, etc.)
            tf_idx: Index of the timeframe (0 for longest, increasing for shorter timeframes)

        Returns:
            Processed tensor (batch_size, seq_len, d_model)
        """
        # Combine ohlc_ret and indicators as our feature input
        features = torch.cat(
            [timeframe_data["ohlc_ret"], timeframe_data["indicators"]], dim=2
        )

        # Apply input normalization
        x = self.input_norms[tf_idx](features)

        # Process numerical data
        x_embedded = self.input_fcs[tf_idx](x)

        # Process time data
        time_embed = self.time_embedding(timeframe_data["time"])

        # Concatenate features with time embeddings
        combined = torch.cat([x_embedded, time_embed], dim=-1)

        # Project to d_model dimensions
        x = self.projections[tf_idx](combined)

        # Add positional encoding
        x = self.pos_encoders[tf_idx](x)

        return x

    def _get_global_context(self, data_batch):
        """
        Extract and process global context information (asset and market).

        Args:
            data_batch: Dictionary with data by timeframe

        Returns:
            Global context tensor (batch_size, asset_embed_dim + market_embed_dim)
        """
        # Take asset and market values from the first timeframe
        # (they are the same for all timeframes of the same batch element)
        first_tf = next(iter(data_batch.values()))

        # Extract indices
        asset_indices = first_tf["asset"].squeeze()  # (batch_size)
        market_indices = first_tf["market"].squeeze()  # (batch_size)

        # Get embeddings
        asset_embeddings = self.asset_embedding(
            asset_indices
        )  # (batch_size, asset_embed_dim)
        market_embeddings = self.market_embedding(
            market_indices
        )  # (batch_size, market_embed_dim)

        # Concatenate to form global context
        global_context = torch.cat([asset_embeddings, market_embeddings], dim=-1)

        return global_context

    def _validate_inputs(self, data_batch):
        """Validate input dimensions and structure."""
        if not data_batch:
            raise ValueError("Empty data batch provided")

        # Verify there are inputs for at least one timeframe
        if len(data_batch) > self.max_timeframes:
            raise ValueError(
                f"Too many timeframes provided ({len(data_batch)}). "
                f"Maximum supported: {self.max_timeframes}"
            )

        # Verify basic structure
        for tf, data in data_batch.items():
            required_keys = ["ohlc_ret", "indicators", "time", "asset"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key '{key}' for timeframe {tf}")

    def _sort_timeframes(self, data_batch):
        """
        Sort timeframes from longest to shortest.

        Args:
            data_batch: Dictionary mapping timeframe names to data dictionaries

        Returns:
            List of (timeframe_name, data_dict) tuples sorted from longest to shortest
        """

        # Helper function to convert timeframe string to minutes
        def timeframe_to_minutes(tf: str) -> int:
            if tf.startswith("M"):
                return int(tf[1:])
            elif tf.startswith("H"):
                return int(tf[1:]) * 60
            elif tf.startswith("D"):
                return int(tf[1:]) * 60 * 24
            elif tf.startswith("W"):
                return int(tf[1:]) * 60 * 24 * 7
            else:
                return 15  # Default to M15 if unknown format

        # Sort timeframes from longest to shortest
        sorted_items = sorted(
            data_batch.items(),
            key=lambda item: timeframe_to_minutes(item[0]),
            reverse=True,
        )

        return sorted_items

    def _process_timeframes(self, data_batch):
        """
        Process all timeframes and combine them using cross-attention cascade.

        Args:a
            data_batch: Dictionary mapping timeframe names to data dictionaries

        Returns:
            Tuple of (final_output, all_tf_outputs, all_tf_names)
        """
        # Sort timeframes from longest to shortest
        sorted_timeframes = self._sort_timeframes(data_batch)

        # Process each timeframe through its transformer
        tf_outputs = []
        tf_names = []

        for idx, (tf_name, tf_data) in enumerate(sorted_timeframes):
            # Process input data for this timeframe
            tf_input = self._process_single_timeframe(tf_data, idx)

            # Apply transformer to get self-attention output
            tf_output = self.transformers[idx](tf_input)

            tf_outputs.append(tf_output)
            tf_names.append(tf_name)

        # Apply cross-attention cascade from longer to shorter timeframes
        # Each timeframe attends to the output of all longer timeframes
        enhanced_outputs = [tf_outputs[0]]  # Longest timeframe has no cross-attention

        for i in range(1, len(tf_outputs)):
            current_output = tf_outputs[i]

            # Apply cross-attention to each longer timeframe
            for j in range(i):
                longer_tf_output = enhanced_outputs[j]
                current_output = self.cross_attentions[j](
                    query=current_output, key_value=longer_tf_output
                )

            enhanced_outputs.append(current_output)

        # Return the shortest timeframe output as the final output (most enhanced)
        # along with all enhanced outputs for information sharing
        return enhanced_outputs[-1], enhanced_outputs, tf_names

    def _apply_recent_bias(self, transformer_output, seq_len):
        """Apply recency bias to the transformer output."""
        if not self.recent_bias:
            return transformer_output

        # Create exponential decay weights that emphasize recent time steps
        positions = torch.arange(
            seq_len, device=transformer_output.device, dtype=torch.float
        ).flip(0)
        decay_rate = F.softplus(self.recency_decay)  # Ensure positive
        decay_weights = torch.exp(-decay_rate * positions)
        decay_weights = decay_weights / decay_weights.sum()  # Normalize

        # Apply decay weights to the transformer output
        weighted_output = transformer_output * decay_weights.view(1, -1, 1)
        recency_enhanced = self.recency_bias(weighted_output.sum(dim=1)).unsqueeze(1)

        return transformer_output + recency_enhanced

    def _detect_market_regime(self, transformer_output, all_tf_outputs):
        """
        Detect market regime using data from all timeframes.

        Args:
            transformer_output: The final combined output
            all_tf_outputs: List of outputs from all timeframes

        Returns:
            Tuple of (transformed_output, regime_weights)
        """
        if not self.market_regime_aware:
            return transformer_output, None

        # Use multiple timeframes for better context
        # Weight the features with exponential decay (longer timeframes get higher weight)
        regime_features = []
        for i, tf_output in enumerate(all_tf_outputs):
            # Take last few candles from each timeframe
            num_candles = min(5, tf_output.shape[1])
            tf_regime_features = tf_output[:, -num_candles:, :].mean(dim=1)
            # Exponential weight based on timeframe position (longer = higher weight)
            weight = 0.6**i  # 0.6, 0.36, 0.216, ...
            regime_features.append(tf_regime_features * weight)

        # Combine regime features
        combined_regime_features = sum(regime_features) / sum(
            0.6**i for i in range(len(all_tf_outputs))
        )
        regime_weights = self.regime_detector(combined_regime_features)

        # Apply regime-specific processing
        regime_outputs = []
        for i, adapter in enumerate(self.regime_adapters):
            regime_outputs.append(
                adapter(transformer_output)
                * regime_weights[:, i].unsqueeze(1).unsqueeze(2)
            )

        # Combine regime-adapted outputs
        transformed_output = sum(regime_outputs)

        return transformed_output, regime_weights

    def _estimate_risk(self, transformer_output, all_tf_outputs):
        """
        Estimate risk based on volatility and drawdown across all timeframes.

        Args:
            transformer_output: The final combined output
            all_tf_outputs: List of outputs from all timeframes

        Returns:
            Combined risk estimate
        """
        if not self.risk_aware:
            return None

        # Verify inputs
        valid_inputs = all(
            isinstance(output, torch.Tensor) for output in all_tf_outputs
        )
        if not valid_inputs:
            return None

        # Calculate timeframe weights using exponential decay (longer = higher weight)
        weights = [0.6**i for i in range(len(all_tf_outputs))]
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]

        # Estimate volatility from recent data in all timeframes
        volatility_estimates = []
        drawdown_estimates = []

        for i, tf_output in enumerate(all_tf_outputs):
            # Recent data for this timeframe
            num_candles = min(5, tf_output.shape[1])
            tf_recent = tf_output[:, -num_candles:, :]

            # Estimate volatility and drawdown
            tf_volatility = self.volatility_estimator(tf_recent.mean(dim=1))
            tf_drawdown = self.drawdown_estimator(tf_output.mean(dim=1))

            # Apply timeframe weight
            volatility_estimates.append(tf_volatility * normalized_weights[i])
            drawdown_estimates.append(tf_drawdown * normalized_weights[i])

        # Combine estimates across timeframes
        combined_volatility = torch.stack(volatility_estimates).sum(dim=0)
        combined_drawdown = torch.stack(drawdown_estimates).sum(dim=0)

        # Combine risk metrics
        risk_inputs = torch.cat([combined_volatility, combined_drawdown], dim=1)
        return torch.sigmoid(self.risk_combiner(risk_inputs))

    def forward(self, data_batch):
        """
        Forward pass through the model with dynamic multi-timeframe data.

        Args:
            data_batch: Dictionary mapping timeframe names (e.g., 'M15', 'H1')
                       to dictionaries with tensors ("ohlc_ret", "indicators", etc.)
                       as produced by MultiWindowDataloader.__next__()

        Returns:
            State representation tensor (batch_size, output_dim)
        """
        # Validate inputs
        self._validate_inputs(data_batch)

        # Extract global information from asset and market (same for all timeframes)
        global_context = self._get_global_context(data_batch)

        # Process all timeframes from longest to shortest
        transformer_output, all_tf_outputs, tf_names = self._process_timeframes(
            data_batch
        )

        # Get sequence length
        seq_len = transformer_output.shape[1]

        # Apply recent bias
        transformer_output = self._apply_recent_bias(transformer_output, seq_len)

        # Detect market regime and apply regime-specific processing
        transformer_output, _ = self._detect_market_regime(
            transformer_output, all_tf_outputs
        )

        # Integrate global context (asset and market) with transformer outputs
        enhanced_output = self.global_context_fusion(transformer_output, global_context)

        # Apply attention pooling to get a fixed-size state representation
        attn_weights = self.attn_pool(enhanced_output)
        pooled = torch.sum(enhanced_output * attn_weights, dim=1)

        # Project to the desired output dimension
        state_representation = self.output_projection(pooled)

        # Return the state representation
        return state_representation


class TanhNormal:
    """TanhNormal distribution for bounded continuous actions with automatic jacobian calculation."""

    def __init__(self, mean, std):
        self.normal = torch.distributions.Normal(mean, std)

    def rsample(self):
        """Reparameterized sample using the tanh squashing function."""
        x = self.normal.rsample()
        y = torch.tanh(x)
        return y

    def log_prob(self, value):
        """Log probability with jacobian adjustment for the tanh transform."""
        # Inverse of tanh (arctanh)
        eps = 1e-6
        value = torch.clamp(value, -1.0 + eps, 1.0 - eps)
        inv_value = 0.5 * torch.log((1 + value) / (1 - value))

        # Log prob from normal distribution
        normal_log_prob = self.normal.log_prob(inv_value)

        # Log det jacobian of tanh transformation: log(1 - tanh^2) = log(sech^2)
        # = log(1) - log(cosh^2) = -2 * log(cosh) = -2 * softplus(2*x) + 2*x
        log_det_jacobian = 2 * (np.log(2) - inv_value - F.softplus(-2 * inv_value))

        # Return log prob with adjustment
        return normal_log_prob - log_det_jacobian


class SACActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim=3,
        hidden_dim=256,
        n_hidden=2,
        log_std_min=-20,
        log_std_max=2,
        enable_position_sizing=True,
        enable_risk_params=True,
        initial_temperature=1.0,
        min_temperature=0.1,
        temperature_decay=0.9995,
    ):
        """
        SAC Actor Network for trading.

        Args:
            state_dim: Dimension of the state representation
            action_dim: Dimension of the action space (typically 3 for Buy, Sell, Hold)
            hidden_dim: Dimension of hidden layers
            n_hidden: Number of hidden layers
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            enable_position_sizing: Whether to output position sizing
            enable_risk_params: Whether to output risk parameters (stop-loss, take-profit)
            initial_temperature: Initial temperature for Gumbel-Softmax
            min_temperature: Minimum temperature after decay
            temperature_decay: Decay rate for temperature (applied each update)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.enable_position_sizing = enable_position_sizing
        self.enable_risk_params = enable_risk_params

        # Temperature scheduling parameters
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.current_temperature = initial_temperature
        self.update_count = 0

        # Add state normalization for stability
        self.state_norm = nn.LayerNorm(state_dim)

        # Shared feature extraction layers
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.shared_net = nn.Sequential(*layers)

        # Action type (discrete) - we use Gumbel-Softmax for differentiable sampling
        self.action_type_net = nn.Linear(hidden_dim, action_dim)

        # For position size (continuous action between 0 and 1)
        if enable_position_sizing:
            self.pos_size_mean = nn.Linear(hidden_dim, 1)
            self.pos_size_log_std = nn.Linear(hidden_dim, 1)

        # For risk parameters (stop-loss and take-profit levels)
        if enable_risk_params:
            self.risk_params_mean = nn.Linear(hidden_dim, 2)  # SL and TP
            self.risk_params_log_std = nn.Linear(hidden_dim, 2)

        # Initialize weights
        self._init_weights()

    def update_temperature(self):
        """Update the temperature according to the decay schedule."""
        self.update_count += 1
        self.current_temperature = max(
            self.min_temperature,
            self.initial_temperature * (self.temperature_decay**self.update_count),
        )
        return self.current_temperature

    def get_temperature(self):
        """Get the current temperature."""
        return self.current_temperature

    def _init_weights(self):
        """Initialize the weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state, deterministic=False, with_logprob=True, temp=None):
        """
        Forward pass that returns sampled actions and their log probabilities.

        Args:
            state: State tensor (batch_size, state_dim)
            deterministic: Whether to sample deterministically
            with_logprob: Whether to compute log probabilities
            temp: Temperature for Gumbel-Softmax sampling (if None, use current scheduled temp)

        Returns:
            Dictionary containing sampled actions and log probabilities
        """
        # Use provided temperature or current scheduled temperature
        temperature = temp if temp is not None else self.current_temperature

        # Normalize state for better stability
        normalized_state = self.state_norm(state)

        # Extract features
        features = self.shared_net(normalized_state)

        # Action type using Gumbel-Softmax
        action_type_logits = self.action_type_net(features)

        if deterministic:
            # Deterministic: take argmax
            action_type = torch.argmax(action_type_logits, dim=-1, keepdim=True)
            action_type_onehot = F.one_hot(
                action_type.squeeze(-1), self.action_dim
            ).float()
            action_type_logprob = None
        else:
            # Sample using RelaxedOneHotCategorical for differentiability
            action_type_dist = torch.distributions.RelaxedOneHotCategorical(
                temperature=temperature, logits=action_type_logits
            )
            action_type_onehot = action_type_dist.rsample()
            action_type = torch.argmax(action_type_onehot, dim=-1, keepdim=True)

            if with_logprob:
                # Get log_prob and ensure consistent shape
                action_type_logprob = action_type_dist.log_prob(
                    action_type_onehot
                ).unsqueeze(-1)
            else:
                action_type_logprob = None

        # Initialize outputs
        outputs = {
            "action_type": action_type,
            "action_type_onehot": action_type_onehot,
            "action_type_logprob": action_type_logprob,
        }

        # Position size (if enabled)
        if self.enable_position_sizing:
            pos_size_mean = torch.sigmoid(self.pos_size_mean(features))
            pos_size_log_std = self.pos_size_log_std(features)
            pos_size_log_std = torch.clamp(
                pos_size_log_std, self.log_std_min, self.log_std_max
            )
            pos_size_std = torch.exp(pos_size_log_std)

            if deterministic:
                pos_size = pos_size_mean
                pos_size_logprob = None
            else:
                # Use TanhNormal for automatic jacobian calculation
                normal = torch.distributions.Normal(pos_size_mean, pos_size_std)

                # Sample using reparameterization
                x_t = normal.rsample()

                # Apply sigmoid squashing
                pos_size = torch.sigmoid(x_t)

                # Compute log_prob with adjustment for sigmoid transform
                if with_logprob:
                    # Apply log-derivative formula for sigmoid transform
                    log_det_jacobian = F.logsigmoid(x_t) + F.logsigmoid(-x_t)
                    pos_size_logprob = normal.log_prob(x_t) - log_det_jacobian
                    pos_size_logprob = pos_size_logprob.unsqueeze(-1)
                else:
                    pos_size_logprob = None

            outputs["position_size"] = pos_size
            outputs["position_size_logprob"] = pos_size_logprob

        # Risk parameters (if enabled)
        if self.enable_risk_params:
            risk_params_mean = self.risk_params_mean(features)
            risk_params_log_std = self.risk_params_log_std(features)
            risk_params_log_std = torch.clamp(
                risk_params_log_std, self.log_std_min, self.log_std_max
            )
            risk_params_std = torch.exp(risk_params_log_std)

            if deterministic:
                risk_params = F.softplus(risk_params_mean)
                risk_params_logprob = None
            else:
                # Use TanhNormal with a modified output scale
                norm_dist = TanhNormal(risk_params_mean, risk_params_std)

                # Sample and scale to positive values
                tanh_sample = norm_dist.rsample()

                # Convert from [-1,1] to [0,∞) using softplus
                risk_params = 0.5 * F.softplus(2.0 * (tanh_sample + 1.0))

                # Compute log_prob with adjustment for the transform
                if with_logprob:
                    raw_log_probs = norm_dist.log_prob(tanh_sample)
                    # Additional jacobian for the softplus transform
                    softplus_jacobian = torch.log(
                        F.softplus(2.0 * (tanh_sample + 1.0)).clamp(min=1e-6)
                    )
                    risk_params_logprob = raw_log_probs - softplus_jacobian
                    risk_params_logprob = risk_params_logprob.sum(dim=1, keepdim=True)
                else:
                    risk_params_logprob = None

            # Split into stop-loss and take-profit
            stop_loss = risk_params[:, 0:1]
            take_profit = risk_params[:, 1:2]

            outputs["stop_loss"] = stop_loss
            outputs["take_profit"] = take_profit
            outputs["risk_params_logprob"] = risk_params_logprob

        # Compute total log probability by summing all components
        if with_logprob:
            log_probs = []

            if outputs["action_type_logprob"] is not None:
                log_probs.append(outputs["action_type_logprob"])

            if (
                self.enable_position_sizing
                and outputs["position_size_logprob"] is not None
            ):
                log_probs.append(outputs["position_size_logprob"])

            if self.enable_risk_params and outputs["risk_params_logprob"] is not None:
                log_probs.append(outputs["risk_params_logprob"])

            if log_probs:
                outputs["log_prob"] = torch.cat(log_probs, dim=1).sum(
                    dim=1, keepdim=True
                )
            else:
                outputs["log_prob"] = None

        return outputs


class SACCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim=3,
        hidden_dim=256,
        n_hidden=2,
        enable_position_sizing=True,
        enable_risk_params=True,
    ):
        """
        SAC Critic Network for trading.

        Args:
            state_dim: Dimension of the state representation
            action_dim: Dimension of the discrete action space (typically 3 for Buy, Sell, Hold)
            hidden_dim: Dimension of hidden layers
            n_hidden: Number of hidden layers
            enable_position_sizing: Whether to include position sizing in action
            enable_risk_params: Whether to include risk parameters in action
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.enable_position_sizing = enable_position_sizing
        self.enable_risk_params = enable_risk_params

        # Calculate total action dimension
        total_action_dim = action_dim  # One-hot encoded action type
        if enable_position_sizing:
            total_action_dim += 1  # Position size (0-1)
        if enable_risk_params:
            total_action_dim += 2  # Stop-loss and take-profit

        # Input layer combines state and action
        self.input_dim = state_dim + total_action_dim

        # Q1 architecture
        q1_layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
        q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 architecture (identical but with different initialization)
        q2_layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
        q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _process_action(self, action_dict):
        """
        Process action dictionary into a flat tensor for critic input.

        Args:
            action_dict: Dictionary containing action components

        Returns:
            Flattened action tensor
        """
        components = [action_dict["action_type_onehot"]]

        if self.enable_position_sizing and "position_size" in action_dict:
            components.append(action_dict["position_size"])

        if self.enable_risk_params:
            if "stop_loss" in action_dict:
                components.append(action_dict["stop_loss"])
            if "take_profit" in action_dict:
                components.append(action_dict["take_profit"])

        return torch.cat(components, dim=1)

    def forward(self, state, action):
        """
        Forward pass to compute Q-values.

        Args:
            state: State tensor (batch_size, state_dim)
            action: Action dictionary or tensor
                If dictionary: contains action components as from SACActor
                If tensor: already flattened action representation

        Returns:
            Tuple of (Q1, Q2) values
        """
        # Process action if it's a dictionary (from actor)
        if isinstance(action, dict):
            action_tensor = self._process_action(action)
        else:
            action_tensor = action

        # Concatenate state and action
        x = torch.cat([state, action_tensor], dim=1)

        # Compute Q values
        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2

    def Q1(self, state, action):
        """
        Forward pass through only Q1 network (used for actor optimization).

        Args:
            state: State tensor
            action: Action dictionary or tensor

        Returns:
            Q1 value
        """
        # Process action if it's a dictionary
        if isinstance(action, dict):
            action_tensor = self._process_action(action)
        else:
            action_tensor = action

        # Concatenate state and action
        x = torch.cat([state, action_tensor], dim=1)

        # Return only Q1 value
        return self.q1(x)


class SACAgent:
    def __init__(
        self,
        feature_extractor,
        actor,
        critic,
        replay_buffer=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha_tuning=True,
        target_entropy=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        SAC Agent that orchestrates the actor, critics, and learning process.

        Args:
            feature_extractor: TransformerFeatureExtractor instance
            actor: SACActor instance
            critic: SACCritic instance
            replay_buffer: Buffer to store transitions (if None, must be set later)
            lr_actor: Learning rate for actor optimizer
            lr_critic: Learning rate for critic optimizer
            lr_alpha: Learning rate for alpha optimizer
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Initial entropy coefficient (or fixed value if auto_tuning=False)
            auto_alpha_tuning: Whether to automatically adjust alpha
            target_entropy: Target entropy for alpha adjustment (if None, set automatically)
            device: Device to run computations on
        """
        self.device = device

        # Set up networks
        self.feature_extractor = feature_extractor.to(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        # Create target critics (we don't need target actor in SAC)
        self.target_critic = SACCritic(
            state_dim=critic.state_dim,
            action_dim=critic.action_dim,
            hidden_dim=critic.q1[0].in_features,  # Extract hidden_dim from first layer
            enable_position_sizing=critic.enable_position_sizing,
            enable_risk_params=critic.enable_risk_params,
        ).to(device)

        # Initialize target weights to match main networks
        self.update_target_networks(tau=1.0)  # Complete copy with tau=1.0

        # Set up optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Entropy coefficient (alpha)
        self.auto_alpha_tuning = auto_alpha_tuning

        if auto_alpha_tuning:
            # Set target entropy
            if target_entropy is None:
                # If action space is discrete, use -|A|
                # For continuous, typically use -dim(A)
                # For hybrid, we use a combined approach
                discrete_part = -0.98 * actor.action_dim  # Slightly less than -|A|
                continuous_part = 0

                if actor.enable_position_sizing:
                    continuous_part -= 1  # -1 for position size

                if actor.enable_risk_params:
                    continuous_part -= 2  # -2 for SL and TP

                self.target_entropy = discrete_part + continuous_part
            else:
                self.target_entropy = target_entropy

            # Create learnable log_alpha
            self.log_alpha = torch.tensor(
                np.log(alpha), requires_grad=True, device=device
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().detach()
        else:
            # Fixed alpha
            self.alpha = alpha
            self.target_entropy = None

        # Replay buffer
        self.replay_buffer = replay_buffer

        # Training parameters
        self.gamma = gamma
        self.tau = tau

        # Tracking variables for training
        self.train_step = 0

    def update_target_networks(self, tau=None):
        """
        Update target network using polyak averaging.

        Args:
            tau: Polyak averaging coefficient (if None, use self.tau)
        """
        if tau is None:
            tau = self.tau

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def extract_features(self, data_batch):
        """
        Extract features from the market data using the feature extractor.

        Args:
            data_batch: Dictionary of market data by timeframe

        Returns:
            State tensor representing the current market state
        """
        self.feature_extractor.eval()  # Set to eval mode
        with torch.no_grad():
            state = self.feature_extractor(data_batch)
        return state

    def select_action(self, state, deterministic=False):
        """
        Select an action based on current state.

        Args:
            state: Current state tensor
            deterministic: Whether to select actions deterministically

        Returns:
            Dictionary of action components
        """
        self.actor.eval()  # Set to eval mode
        with torch.no_grad():
            state = state.to(self.device)
            actions = self.actor(state, deterministic=deterministic, with_logprob=False)

        # Convert tensors to numpy arrays for environment interaction
        for key, value in actions.items():
            if isinstance(value, torch.Tensor):
                actions[key] = value.cpu().numpy()

        return actions

    def process_action_for_env(self, actions):
        """
        Process action dictionary into format expected by environment.
        This should be customized based on the environment's action space.

        Args:
            actions: Dictionary of action components from select_action

        Returns:
            Action in format expected by environment
        """
        # Default implementation - override as needed
        action_type = actions["action_type"].squeeze()

        # Process position size and risk params if present
        position_size = actions.get("position_size", np.array([[1.0]])).squeeze()
        stop_loss = actions.get("stop_loss", np.array([[0.02]])).squeeze()  # Default 2%
        take_profit = actions.get(
            "take_profit", np.array([[0.06]])
        ).squeeze()  # Default 6%

        # Format for environment (example)
        # Adjust this based on what your environment expects
        env_action = {
            "action_type": action_type,  # 0: Buy, 1: Sell, 2: Hold
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        return env_action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if self.replay_buffer is not None:
            self.replay_buffer.add(state, action, reward, next_state, done)

    def _soft_update_target_networks(self):
        """Soft update target networks using Polyak averaging"""
        self.update_target_networks()

    def update_parameters(self, batch_size, updates_per_step):
        """
        Update model parameters using SAC algorithm.

        Args:
            batch_size: Size of batch to sample from replay buffer
            updates_per_step: Number of gradient steps to take

        Returns:
            Dictionary of training metrics
        """
        if self.replay_buffer is None or self.replay_buffer.size < batch_size:
            return {}

        metrics = {
            "actor_loss": 0,
            "critic_loss": 0,
            "alpha_loss": 0,
            "alpha": self.alpha.item()
            if isinstance(self.alpha, torch.Tensor)
            else self.alpha,
            "q1_value": 0,
            "q2_value": 0,
            "target_q": 0,
            "entropy": 0,
        }

        for _ in range(updates_per_step):
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                batch_size
            )

            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = self._actions_to_tensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # Update critic
            critic_loss, q1_value, q2_value, target_q = self._update_critic(
                states, actions, rewards, next_states, dones
            )
            metrics["critic_loss"] += critic_loss
            metrics["q1_value"] += q1_value
            metrics["q2_value"] += q2_value
            metrics["target_q"] += target_q

            # Update actor
            actor_loss, entropy = self._update_actor(states)
            metrics["actor_loss"] += actor_loss
            metrics["entropy"] += entropy

            # Update alpha if automatic tuning is enabled
            if self.auto_alpha_tuning:
                alpha_loss = self._update_alpha(entropy)
                metrics["alpha_loss"] += alpha_loss
                metrics["alpha"] = (
                    self.alpha.item()
                    if isinstance(self.alpha, torch.Tensor)
                    else self.alpha
                )

            # Soft update target networks
            self._soft_update_target_networks()

            # Update temperature for action sampling
            if hasattr(self.actor, "update_temperature"):
                current_temp = self.actor.update_temperature()
                if "temperature" not in metrics:
                    metrics["temperature"] = 0
                metrics["temperature"] += current_temp

            self.train_step += 1

        # Average metrics over updates
        for key in metrics:
            metrics[key] /= updates_per_step

        return metrics

    def _actions_to_tensor(self, actions):
        """
        Convert action dictionaries to tensors.

        Args:
            actions: List of action dictionaries from replay buffer

        Returns:
            Dictionary of action tensors
        """
        # Initialize output dictionary
        action_tensors = {}

        # Get keys from first action
        first_action = actions[0]

        for key in first_action:
            # Skip log_prob or other non-action components
            if key in [
                "log_prob",
                "action_type_logprob",
                "position_size_logprob",
                "risk_params_logprob",
            ]:
                continue

            # Convert list of values to tensor
            values = [action[key] for action in actions]
            action_tensors[key] = torch.FloatTensor(np.array(values)).to(self.device)

        return action_tensors

    def _update_critic(self, states, actions, rewards, next_states, dones):
        """
        Update critic using TD learning.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of terminal flags

        Returns:
            Tuple of (critic_loss, q1_value, q2_value, target_q)
        """
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions = self.actor(next_states, with_logprob=True)

            # Get log probabilities
            next_log_probs = next_actions["log_prob"]

            # Compute target Q values
            next_q1, next_q2 = self.target_critic(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # Compute entropy-regularized target
            target_q = rewards + (1 - dones) * self.gamma * (
                next_q - self.alpha * next_log_probs
            )

        # Compute current Q values
        current_q1, current_q2 = self.critic(states, actions)

        # Compute critic loss using MSE
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return (
            critic_loss.item(),
            current_q1.mean().item(),
            current_q2.mean().item(),
            target_q.mean().item(),
        )

    def _update_actor(self, states):
        """
        Update actor using policy gradient with entropy regularization.

        Args:
            states: Batch of states

        Returns:
            Tuple of (actor_loss, entropy)
        """
        # Sample actions from current policy
        actions = self.actor(states, with_logprob=True)
        log_probs = actions["log_prob"]

        # Compute Q value for the sampled actions
        q1 = self.critic.Q1(states, actions)

        # Compute actor loss: maximize Q - alpha * log_prob
        actor_loss = (self.alpha * log_probs - q1).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -log_probs.mean().item()

        return actor_loss.item(), entropy

    def _update_alpha(self, entropy):
        """
        Update alpha parameter to achieve target entropy.

        Args:
            entropy: Current policy entropy

        Returns:
            alpha_loss value
        """
        # Compute alpha loss: minimize -log_alpha * (entropy - target_entropy)
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()

        # Optimize alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update alpha value
        self.alpha = self.log_alpha.exp().detach()

        return alpha_loss.item()

    def save(self, path):
        """
        Save model parameters.

        Args:
            path: Path to save model
        """
        torch.save(
            {
                "feature_extractor": self.feature_extractor.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha if self.auto_alpha_tuning else None,
                "train_step": self.train_step,
            },
            path,
        )

    def load(self, path):
        """
        Load model parameters.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])

        if self.auto_alpha_tuning and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp().detach()

        self.train_step = checkpoint["train_step"]
