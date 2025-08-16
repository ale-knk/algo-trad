import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Comentar la importación problemática ya que crearemos nuestra propia función para esto
# from pytrad.currency_encoder import CurrencyPairIndexer

# Constantes para inicializar embeddings
NUM_ASSETS = 50  # Ajustar según el número de assets diferentes
NUM_MARKETS = 10  # Ajustar según el número de mercados diferentes


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
    Fusiona información de contexto global (market y asset) con los features locales (timeframes).
    """

    def __init__(self, d_model, global_dim, nhead=4, dropout=0.1):
        super().__init__()
        # Proyección del contexto global al espacio del modelo
        self.global_projection = nn.Linear(global_dim, d_model)

        # Atención cruzada para que los features locales atiendan al contexto global
        self.cross_attention = CrossAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )

        # Capa de fusión final
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, local_features, global_context):
        """
        Fusiona features locales con el contexto global.

        Args:
            local_features: Tensor de forma (batch_size, seq_len, d_model)
            global_context: Tensor de forma (batch_size, global_dim)

        Returns:
            Tensor fusionado de forma (batch_size, seq_len, d_model)
        """
        # Proyectar contexto global al espacio del modelo
        projected_global = self.global_projection(global_context)

        # Expandir contexto global para coincidir con la dimensión de secuencia de los features locales
        # (batch_size, 1, d_model) -> (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = local_features.shape
        expanded_global = projected_global.unsqueeze(1).expand(-1, seq_len, -1)

        # Aplicar atención cruzada: features locales atienden al contexto global
        attended_features = self.cross_attention(local_features, expanded_global)

        # Concatenar y fusionar
        combined = torch.cat([local_features, attended_features], dim=-1)
        fused = self.fusion_layer(combined)

        return fused


class ActorCriticTransformer(nn.Module):
    def __init__(
        self,
        input_dim=8,  # OHLC returns (4) + indicadores
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        asset_embed_dim=32,  # Dimensión para embeddings de assets
        market_embed_dim=16,  # Dimensión para embeddings de markets
        fc_hidden_dim=128,
        time_dim=21,
        time_embed_dim=32,
        recent_bias=True,
        risk_aware=True,
        market_regime_aware=True,
        max_timeframes=5,
    ):
        """
        Initialize the Actor-Critic Transformer for trading with dynamic multi-timeframe awareness.

        Args:
            input_dim: Dimension of each time step's feature vector (OHLC returns + indicators).
            d_model: Embedding dimension for the transformer.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            asset_embed_dim: Dimension for asset embeddings.
            market_embed_dim: Dimension for market embeddings.
            fc_hidden_dim: Hidden dimension for actor/critic fully connected layers.
            time_dim: Dimension of time data.
            time_embed_dim: Embedding dimension for time data.
            recent_bias: Whether to add additional attention to recent data.
            risk_aware: Whether to add risk-aware components to the model.
            market_regime_aware: Whether to detect and adapt to market regimes.
            max_timeframes: Maximum number of timeframes the model can handle.
        """
        super().__init__()

        # Store configuration
        self.market_regime_aware = market_regime_aware
        self.risk_aware = risk_aware
        self.recent_bias = recent_bias
        self.d_model = d_model
        self.fc_hidden_dim = fc_hidden_dim
        self.time_dim = time_dim
        self.max_timeframes = max_timeframes
        self.input_dim = input_dim
        self.asset_embed_dim = asset_embed_dim
        self.market_embed_dim = market_embed_dim

        # Create embeddings for assets y markets
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

        # Calculate total dimension after concatenation (sin incluir market/asset global)
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

        # Global context fusion - para integrar información de asset y market
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
            self._init_market_regime_components(d_model, fc_hidden_dim)

        # Risk-aware component
        if risk_aware:
            self._init_risk_components(d_model, fc_hidden_dim)

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
        # Note: Using ohlc_ret instead of ohlc as requested
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
        Extrae y procesa la información de contexto global (asset y market).

        Args:
            data_batch: Diccionario con datos por timeframe

        Returns:
            Tensor de contexto global (batch_size, asset_embed_dim + market_embed_dim)
        """
        # Tomar los valores de asset y market del primer timeframe
        # (son los mismos para todos los timeframes de un mismo elemento del batch)
        first_tf = next(iter(data_batch.values()))

        # Extraer indices
        asset_indices = first_tf["asset"].squeeze()  # (batch_size)
        market_indices = first_tf["market"].squeeze()  # (batch_size)

        # Obtener embeddings
        asset_embeddings = self.asset_embedding(
            asset_indices
        )  # (batch_size, asset_embed_dim)
        market_embeddings = self.market_embedding(
            market_indices
        )  # (batch_size, market_embed_dim)

        # Concatenar para formar el contexto global
        global_context = torch.cat([asset_embeddings, market_embeddings], dim=-1)

        return global_context

    def _validate_inputs(self, data_batch):
        """Validate input dimensions and structure."""
        if not data_batch:
            raise ValueError("Empty data batch provided")

        # Verificar que hay entradas para al menos un timeframe
        if len(data_batch) > self.max_timeframes:
            raise ValueError(
                f"Too many timeframes provided ({len(data_batch)}). "
                f"Maximum supported: {self.max_timeframes}"
            )

        # Verificar estructura básica
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

        Args:
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

        # Combine estimates across timeframes - Fix con torch.stack
        combined_volatility = torch.stack(volatility_estimates).sum(dim=0)
        combined_drawdown = torch.stack(drawdown_estimates).sum(dim=0)

        # Combine risk metrics
        risk_inputs = torch.cat([combined_volatility, combined_drawdown], dim=1)
        return torch.sigmoid(self.risk_combiner(risk_inputs))

    def _compute_action_logits(self, actor_features):
        """Compute action logits with shorting configuration."""
        # All actions available: Buy (0), Sell (1), Hold (2)
        return self.action_head(actor_features)

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

        # Separar stop_loss y take_profit
        stop_loss = base_risk_params[:, 0:1]  # Primera dimensión
        take_profit = base_risk_params[:, 1:2]  # Segunda dimensión

        if self.risk_aware and risk_estimate is not None:
            # Higher volatility = wider stop loss and take profit
            volatility_scale = 1.0 + risk_estimate * 2.0  # Scale from 1x to 3x
            stop_loss = stop_loss * volatility_scale
            take_profit = take_profit * volatility_scale

        return stop_loss, take_profit

    def _compute_critic_value(self, pooled, risk_estimate):
        """Compute critic value with risk adjustment if enabled."""
        critic_features = self.critic_fc(pooled)
        value_estimate = self.value_head(critic_features)

        if self.risk_aware and risk_estimate is not None:
            risk_adjusted_value = self.risk_head(critic_features)
            return value_estimate - risk_estimate * risk_adjusted_value

        return value_estimate

    def forward(self, data_batch):
        """
        Forward pass through the model with dynamic multi-timeframe data.

        Args:
            data_batch: Dictionary mapping timeframe names (e.g., 'M15', 'H1')
                       to dictionaries with tensors ("ohlc_ret", "indicators", etc.)
                       as produced by MultiWindowDataloader.__next__()

        Returns:
            Tuple of (actor_outputs, critic_value)
        """
        # Validate inputs
        self._validate_inputs(data_batch)

        # Extraer información global de asset y market (igual para todos los timeframes)
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
        transformer_output, regime_weights = self._detect_market_regime(
            transformer_output, all_tf_outputs
        )

        # Integrar contexto global (asset y market) a las salidas de los transformers
        enhanced_output = self.global_context_fusion(transformer_output, global_context)

        # Apply attention pooling
        attn_weights = self.attn_pool(enhanced_output)
        pooled = torch.sum(enhanced_output * attn_weights, dim=1)

        # Estimate risk
        risk_estimate = self._estimate_risk(enhanced_output, all_tf_outputs)

        # Actor computations
        actor_features = self.actor_fc(pooled)
        action_logits = self._compute_action_logits(actor_features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Position sizing with Kelly criterion and risk adjustment
        position_size, kelly_fraction = self._compute_position_sizing(
            actor_features, pooled, risk_estimate
        )

        # Risk parameters (stop-loss and take-profit)
        stop_loss, take_profit = self._compute_risk_parameters(
            actor_features, risk_estimate
        )

        # Critic computation
        critic_value = self._compute_critic_value(pooled, risk_estimate)

        # Compute timeframe confidences for each timeframe
        timeframe_confidences = {}
        for i, tf_name in enumerate(tf_names):
            tf_output = all_tf_outputs[i]
            tf_attn = self.attn_pool(tf_output)
            tf_entropy = -(tf_attn * torch.log(tf_attn + 1e-10)).sum(dim=1).mean()
            # Higher entropy = lower confidence
            tf_confidence = torch.sigmoid(
                -tf_entropy * 3
            )  # Scale for better visualization
            timeframe_confidences[tf_name] = tf_confidence

        # Prepare actor outputs
        actor_out = {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "position_size": position_size,
            "stop_loss": stop_loss,  # Nueva clave para stop loss
            "take_profit": take_profit,  # Nueva clave para take profit
            "kelly_fraction": kelly_fraction.unsqueeze(1),
            "timeframe_confidences": timeframe_confidences,
        }

        # Add additional outputs if available
        if self.risk_aware and risk_estimate is not None:
            actor_out["estimated_risk"] = risk_estimate

        if self.market_regime_aware and regime_weights is not None:
            actor_out["market_regime"] = regime_weights

        return actor_out, critic_value
