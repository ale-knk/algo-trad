import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Simplified CurrencyPairIndexer for testing
class CurrencyPairIndexer:
    def __init__(self):
        self.vocab_size = 10  # Using 10 currency pairs for testing


# PositionalEncoding class from the original code
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


# CrossAttention class from the original code
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


# Fixed ActorCriticTransformer class
class ActorCriticTransformer(nn.Module):
    def __init__(
        self,
        input_dim=5,  # OHLCV features
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        currency_embed_dim=16,
        fc_hidden_dim=128,
        time_dim=4,  # Dimension of time data (cyclically encoded)
        time_embed_dim=16,  # Embedding dimension for time
        recent_bias=True,  # Add bias toward recent data
        risk_aware=True,  # Add risk-aware components
        enable_shorting=True,  # Allow short positions
        use_layer_norm=True,  # Use layer normalization for inputs
        market_regime_aware=True,  # Enable market regime detection
        transaction_costs=True,  # Consider transaction costs in decisions
    ):
        """
        Initialize the Actor-Critic Transformer for trading with multi-timeframe awareness.
        """
        super().__init__()

        # Store configuration
        self.enable_shorting = enable_shorting
        self.use_layer_norm = use_layer_norm
        self.market_regime_aware = market_regime_aware
        self.transaction_costs = transaction_costs
        self.risk_aware = risk_aware
        self.recent_bias = recent_bias
        self.d_model = d_model
        self.fc_hidden_dim = fc_hidden_dim

        # Get number of currency pairs using CurrencyPairIndexer
        num_currency_pairs = CurrencyPairIndexer().vocab_size

        # Input normalization for both timeframes
        if use_layer_norm:
            self.input_norm_low = nn.LayerNorm(input_dim)
            self.input_norm_high = nn.LayerNorm(input_dim)

        # Input embedding layers for both timeframes
        self.input_fc_low = nn.Linear(input_dim, d_model)
        self.input_fc_high = nn.Linear(input_dim, d_model)

        # Currency pair embedding (shared across timeframes)
        self.currency_embedding = nn.Embedding(num_currency_pairs, currency_embed_dim)

        # Time embedding for both timeframes
        self.time_embedding = nn.Linear(time_dim, time_embed_dim)

        # Calculate total dimension after concatenation
        total_dim = d_model + currency_embed_dim + time_embed_dim

        # Projection layers to match transformer input dimension for both timeframes
        self.projection_low = nn.Linear(total_dim, d_model)
        self.projection_high = nn.Linear(total_dim, d_model)

        # Positional encoding (shared architecture but separate instances)
        self.pos_encoder_low = PositionalEncoding(d_model)
        self.pos_encoder_high = PositionalEncoding(d_model)

        # Higher timeframe transformer (processes 1h data)
        high_tf_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.high_tf_transformer = nn.TransformerEncoder(
            high_tf_encoder_layer,
            num_layers=num_layers // 2,  # Can be shallower
        )

        # Lower timeframe transformer (processes 15min data)
        low_tf_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.low_tf_transformer = nn.TransformerEncoder(
            low_tf_encoder_layer, num_layers=num_layers
        )

        # Cross-attention mechanism to attend from low TF to high TF
        self.cross_attention = CrossAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )

        # Fusion layer to combine information from both timeframes
        self.fusion_layer = nn.Linear(d_model * 2, d_model)

        # Recent bias layer (for giving more importance to recent data)
        if recent_bias:
            # Improved recency bias mechanism with learnable decay
            self.recency_decay = nn.Parameter(torch.ones(1) * 0.1)
            self.recency_bias = nn.Linear(d_model, d_model)

        # Market regime detection
        if market_regime_aware:
            self._init_market_regime_components(d_model, fc_hidden_dim)

        # Risk-aware component
        if risk_aware:
            self._init_risk_components(d_model, fc_hidden_dim)

        # Transaction cost modeling
        if transaction_costs:
            self._init_transaction_cost_components(d_model, fc_hidden_dim)

        # Attention pooling instead of simple average pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1), nn.Softmax(dim=1)
        )

        # Actor and critic components
        self._init_actor_components(d_model, fc_hidden_dim, dropout)
        self._init_critic_components(d_model, fc_hidden_dim, dropout)

    def _init_market_regime_components(self, d_model, fc_hidden_dim):
        """Initialize components for market regime detection and adaptation."""
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 3),  # Trending up, trending down, ranging
            nn.Softmax(dim=-1),
        )
        # Regime-specific processing
        self.regime_adapters = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )

    def _init_risk_components(self, d_model, fc_hidden_dim):
        """Initialize components for risk estimation and adjustment."""
        # Enhanced risk estimation with multiple components
        self.volatility_estimator = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Drawdown risk estimator
        self.drawdown_estimator = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Combine risk metrics
        self.risk_combiner = nn.Linear(2, 1)

    def _init_transaction_cost_components(self, d_model, fc_hidden_dim):
        """Initialize components for transaction cost modeling."""
        self.cost_estimator = nn.Sequential(
            nn.Linear(d_model + 1, fc_hidden_dim // 2),  # +1 for position size
            nn.ReLU(),
            nn.Linear(fc_hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _init_actor_components(self, d_model, fc_hidden_dim, dropout):
        """Initialize actor network components."""
        # Actor Head
        self.actor_fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Action type: Buy, Sell, Hold
        self.action_head = nn.Linear(fc_hidden_dim, 3)

        # Position sizing - magnitude only
        self.position_size_head = nn.Sequential(
            nn.Linear(fc_hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Risk parameters for dynamic stop-loss and take-profit
        self.risk_params_head = nn.Sequential(
            nn.Linear(fc_hidden_dim, 2),
            nn.Softplus(),
        )

        # Kelly criterion-based position sizing
        self.kelly_estimator = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 2),  # Win rate and payoff ratio
            nn.Sigmoid(),
        )

    def _init_critic_components(self, d_model, fc_hidden_dim, dropout):
        """Initialize critic network components."""
        # Critic Head with dual value estimation
        self.critic_fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_head = nn.Linear(fc_hidden_dim, 1)  # Expected return
        self.risk_head = nn.Linear(fc_hidden_dim, 1)  # Risk-adjusted return

    def _process_single_timeframe(
        self, x, currency_pairs, time_data, is_high_timeframe=False
    ):
        """
        Process a single timeframe's data through the initial layers.

        Args:
            x: Features tensor (batch_size, seq_len, input_dim)
            currency_pairs: Currency pair indices (batch_size, seq_len)
            time_data: Time encodings (batch_size, seq_len, time_dim)
            is_high_timeframe: Whether this is the higher timeframe (1h vs 15min)

        Returns:
            Processed tensor (batch_size, seq_len, d_model)
        """
        # Apply input normalization if enabled - FIX HERE
        if self.use_layer_norm:
            input_norm = (
                self.input_norm_high if is_high_timeframe else self.input_norm_low
            )
            x = input_norm(
                x
            )  # Apply normalization to input x, not to the module itself

        # Process numerical data
        x_embedded = (
            self.input_fc_high(x) if is_high_timeframe else self.input_fc_low(x)
        )

        # Process currency pairs (shared embedding)
        currency_embed = self.currency_embedding(currency_pairs)

        # Process time data
        time_embed = self.time_embedding(time_data)

        # Concatenate all features
        combined = torch.cat([x_embedded, currency_embed, time_embed], dim=-1)

        # Project to d_model dimensions
        projection = self.projection_high if is_high_timeframe else self.projection_low
        x = projection(combined)

        # Add positional encoding
        pos_encoder = (
            self.pos_encoder_high if is_high_timeframe else self.pos_encoder_low
        )
        x = pos_encoder(x)

        return x

    def _validate_inputs(
        self,
        x_low,
        currency_pairs_low,
        time_data_low,
        x_high,
        currency_pairs_high,
        time_data_high,
    ):
        """Validate input dimensions."""
        if x_low.dim() != 3:
            raise ValueError(f"Expected x_low to have 3 dimensions, got {x_low.dim()}")
        if currency_pairs_low.dim() != 2:
            raise ValueError(
                f"Expected currency_pairs_low to have 2 dimensions, got {currency_pairs_low.dim()}"
            )
        if time_data_low.dim() != 3:
            raise ValueError(
                f"Expected time_data_low to have 3 dimensions, got {time_data_low.dim()}"
            )

        # Input validation for higher timeframe
        if x_high.dim() != 3:
            raise ValueError(
                f"Expected x_high to have 3 dimensions, got {x_high.dim()}"
            )
        if currency_pairs_high.dim() != 2:
            raise ValueError(
                f"Expected currency_pairs_high to have 2 dimensions, got {currency_pairs_high.dim()}"
            )
        if time_data_high.dim() != 3:
            raise ValueError(
                f"Expected time_data_high to have 3 dimensions, got {time_data_high.dim()}"
            )

    def _process_timeframes(
        self,
        x_low,
        currency_pairs_low,
        time_data_low,
        x_high,
        currency_pairs_high,
        time_data_high,
    ):
        """Process both timeframes and combine them using cross-attention."""
        # Process higher timeframe data first (1h)
        high_tf_input = self._process_single_timeframe(
            x_high, currency_pairs_high, time_data_high, is_high_timeframe=True
        )

        # Pass through higher timeframe transformer
        high_tf_output = self.high_tf_transformer(high_tf_input)

        # Process lower timeframe data (15min)
        low_tf_input = self._process_single_timeframe(
            x_low, currency_pairs_low, time_data_low, is_high_timeframe=False
        )

        # Process lower timeframe through its transformer
        low_tf_self_output = self.low_tf_transformer(low_tf_input)

        # Apply cross-attention: lower timeframe attends to higher timeframe
        cross_attn_output = self.cross_attention(
            query=low_tf_self_output,
            key_value=high_tf_output,
        )

        return cross_attn_output, high_tf_output, low_tf_self_output

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

    def _detect_market_regime(self, transformer_output, high_tf_output):
        """Detect market regime and apply regime-specific processing."""
        if not self.market_regime_aware:
            return transformer_output, None

        # Detect market regime using both timeframes for better context
        high_tf_regime_features = high_tf_output[:, -3:, :].mean(dim=1)
        low_tf_regime_features = transformer_output[:, -5:, :].mean(dim=1)

        # Combine regime features
        combined_regime_features = (
            high_tf_regime_features * 0.7 + low_tf_regime_features * 0.3
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

    def _estimate_risk(self, transformer_output, high_tf_output):
        """Estimate risk based on volatility and drawdown across timeframes."""
        if not self.risk_aware:
            return None

        if not (
            isinstance(transformer_output, torch.Tensor)
            and isinstance(high_tf_output, torch.Tensor)
        ):
            return None

        # Estimate volatility from recent data in both timeframes
        low_tf_recent = transformer_output[:, -5:, :]
        high_tf_recent = high_tf_output[:, -3:, :]

        # Combine volatility estimates from both timeframes
        low_tf_volatility = self.volatility_estimator(low_tf_recent.mean(dim=1))
        high_tf_volatility = self.volatility_estimator(high_tf_recent.mean(dim=1))
        combined_volatility = 0.4 * low_tf_volatility + 0.6 * high_tf_volatility

        # Estimate drawdown risk (using both timeframes)
        low_tf_drawdown = self.drawdown_estimator(transformer_output.mean(dim=1))
        high_tf_drawdown = self.drawdown_estimator(high_tf_output.mean(dim=1))
        combined_drawdown = 0.4 * low_tf_drawdown + 0.6 * high_tf_drawdown

        # Combine risk metrics
        risk_inputs = torch.cat([combined_volatility, combined_drawdown], dim=1)
        return torch.sigmoid(self.risk_combiner(risk_inputs))

    def _compute_action_logits(self, actor_features):
        """Compute action logits with shorting configuration."""
        if self.enable_shorting:
            # All actions available: Buy (0), Sell (1), Hold (2)
            return self.action_head(actor_features)
        else:
            # Only Buy and Hold available, mask out Sell action
            raw_logits = self.action_head(actor_features)
            sell_mask = torch.zeros_like(raw_logits)
            sell_mask[:, 1] = -1e9  # Large negative value for Sell
            return raw_logits + sell_mask

    def _compute_position_sizing(self, actor_features, pooled, risk_estimate):
        """Compute position size with Kelly criterion and risk adjustments."""
        # Base position size
        position_size = self.position_size_head(actor_features)

        # Kelly criterion adjustment
        kelly_outputs = self.kelly_estimator(pooled)
        win_rate = kelly_outputs[:, 0]
        payoff_ratio = 1.0 + kelly_outputs[:, 1] * 4.0  # Scale to 1-5 range
        kelly_fraction = torch.clamp((win_rate * payoff_ratio - 1) / payoff_ratio, 0, 1)
        position_size = position_size * kelly_fraction.unsqueeze(1)

        # Risk adjustment
        if self.risk_aware and risk_estimate is not None:
            risk_adjustment = 1.0 - risk_estimate
            position_size = position_size * risk_adjustment

        return position_size, kelly_fraction

    def _compute_risk_parameters(self, actor_features, risk_estimate):
        """Compute stop-loss and take-profit parameters."""
        base_risk_params = self.risk_params_head(actor_features)

        if self.risk_aware and risk_estimate is not None:
            # Higher volatility = wider stop loss and take profit
            volatility_scale = 1.0 + risk_estimate * 2.0  # Scale from 1x to 3x
            return base_risk_params * volatility_scale

        return base_risk_params

    def _model_transaction_costs(self, pooled, position_size, current_position):
        """Model transaction costs and adjust position size if needed."""
        if not (self.transaction_costs and current_position is not None):
            return position_size, None

        # Estimate transaction costs
        position_change = torch.abs(position_size - current_position)
        cost_input = torch.cat([pooled, position_change], dim=1)
        transaction_cost = self.cost_estimator(cost_input)

        # Apply transaction cost penalty to position size
        net_position_change = position_size - current_position
        profit_potential = (
            torch.abs(net_position_change) * 0.01
        )  # Simplified placeholder
        should_trade = (profit_potential > transaction_cost).float()
        adjusted_position = current_position + net_position_change * should_trade

        return adjusted_position, transaction_cost

    def _compute_critic_value(self, pooled, risk_estimate):
        """Compute critic value with risk adjustment if enabled."""
        critic_features = self.critic_fc(pooled)
        value_estimate = self.value_head(critic_features)

        if self.risk_aware and risk_estimate is not None:
            risk_adjusted_value = self.risk_head(critic_features)
            return value_estimate - risk_estimate * risk_adjusted_value

        return value_estimate

    def _compute_timeframe_confidence(self, high_tf_output, attn_weights):
        """Compute the confidence in each timeframe based on attention entropy."""
        high_tf_attn = self.attn_pool(high_tf_output)
        low_tf_attn = attn_weights

        # Compute entropy of attention distributions
        high_tf_entropy = (
            -(high_tf_attn * torch.log(high_tf_attn + 1e-10)).sum(dim=1).mean()
        )
        low_tf_entropy = (
            -(low_tf_attn * torch.log(low_tf_attn + 1e-10)).sum(dim=1).mean()
        )

        # Calculate timeframe confidence
        return torch.sigmoid(low_tf_entropy - high_tf_entropy)

    def forward(
        self,
        x_low,
        currency_pairs_low,
        time_data_low,  # 15min data
        x_high,
        currency_pairs_high,
        time_data_high,  # 1h data
        current_position=None,
    ):
        """
        Forward pass through the model with multi-timeframe data.

        Args:
            x_low: Tensor (batch_size, seq_len_low, input_dim) - Lower timeframe data (15min)
            currency_pairs_low: Tensor (batch_size, seq_len_low) - Currency pair indices for lower timeframe
            time_data_low: Tensor (batch_size, seq_len_low, time_dim) - Time encoding for lower timeframe
            x_high: Tensor (batch_size, seq_len_high, input_dim) - Higher timeframe data (1h)
            currency_pairs_high: Tensor (batch_size, seq_len_high) - Currency pair indices for higher timeframe
            time_data_high: Tensor (batch_size, seq_len_high, time_dim) - Time encoding for higher timeframe
            current_position: Optional tensor (batch_size, 1) - Current position size

        Returns:
            Tuple of (actor_outputs, critic_value)
        """
        # Validate inputs
        self._validate_inputs(
            x_low,
            currency_pairs_low,
            time_data_low,
            x_high,
            currency_pairs_high,
            time_data_high,
        )

        # Get sequence lengths
        seq_len_low = x_low.shape[1]

        # Process both timeframes and combine them
        transformer_output, high_tf_output, _ = self._process_timeframes(
            x_low,
            currency_pairs_low,
            time_data_low,
            x_high,
            currency_pairs_high,
            time_data_high,
        )

        # Apply recent bias
        transformer_output = self._apply_recent_bias(transformer_output, seq_len_low)

        # Detect market regime and apply regime-specific processing
        transformer_output, regime_weights = self._detect_market_regime(
            transformer_output, high_tf_output
        )

        # Apply attention pooling
        attn_weights = self.attn_pool(transformer_output)
        pooled = torch.sum(transformer_output * attn_weights, dim=1)

        # Estimate risk
        risk_estimate = self._estimate_risk(transformer_output, high_tf_output)

        # Actor computations
        actor_features = self.actor_fc(pooled)
        action_logits = self._compute_action_logits(actor_features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Position sizing with Kelly criterion and risk adjustment
        position_size, kelly_fraction = self._compute_position_sizing(
            actor_features, pooled, risk_estimate
        )

        # Risk parameters (stop-loss and take-profit)
        risk_params = self._compute_risk_parameters(actor_features, risk_estimate)

        # Transaction cost modeling
        position_size, transaction_cost = self._model_transaction_costs(
            pooled, position_size, current_position
        )

        # Critic computation
        critic_value = self._compute_critic_value(pooled, risk_estimate)

        # Compute timeframe confidence
        timeframe_confidence = self._compute_timeframe_confidence(
            high_tf_output, attn_weights
        )

        # Prepare actor outputs
        actor_out = {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "position_size": position_size,
            "risk_params": risk_params,
            "kelly_fraction": kelly_fraction.unsqueeze(1),
            "timeframe_confidence": timeframe_confidence,
        }

        # Add additional outputs if available
        if self.risk_aware and risk_estimate is not None:
            actor_out["estimated_risk"] = risk_estimate

        if self.market_regime_aware and regime_weights is not None:
            actor_out["market_regime"] = regime_weights

        if self.transaction_costs and transaction_cost is not None:
            actor_out["transaction_cost"] = transaction_cost

        return actor_out, critic_value


# Test the model with dummy data
if __name__ == "__main__":
    # Set up dimensions
    batch_size = 2
    seq_len_low = 20  # Sequence length for 15min data
    seq_len_high = 5  # Sequence length for 1h data
    input_dim = 5  # OHLCV features
    time_dim = 4  # Time encoding dimensions

    # Create dummy input tensors
    # Lower timeframe data (15min)
    x_low = torch.randn(batch_size, seq_len_low, input_dim)
    currency_pairs_low = torch.randint(
        0, 10, (batch_size, seq_len_low)
    )  # Assuming 10 currency pairs
    time_data_low = torch.randn(batch_size, seq_len_low, time_dim)

    # Higher timeframe data (1h)
    x_high = torch.randn(batch_size, seq_len_high, input_dim)
    currency_pairs_high = torch.randint(0, 10, (batch_size, seq_len_high))
    time_data_high = torch.randn(batch_size, seq_len_high, time_dim)

    # Optional current position
    current_position = torch.rand(batch_size, 1)  # Random positions between 0 and 1

    # Initialize the model
    model = ActorCriticTransformer()

    # Set model to evaluation mode
    model.eval()

    # Run forward pass
    with torch.no_grad():
        actor_outputs, critic_value = model(
            x_low,
            currency_pairs_low,
            time_data_low,
            x_high,
            currency_pairs_high,
            time_data_high,
            current_position,
        )

    # Print the outputs to verify
    print("Actor outputs:")
    for key, value in actor_outputs.items():
        print(f"  {key}: {value.shape}")

    print("\nCritic value shape:", critic_value.shape)
    print("Critic value:", critic_value)
