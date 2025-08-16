# Reinforcement Learning Trading System

This module provides a framework for training and evaluating reinforcement learning (RL) agents for automated trading. The RL agents can learn when to make trading decisions and what levels to set for take profit, stop loss, and trading volume.

## Overview

The system consists of the following main components:

1. **Environment**: A trading environment compatible with the OpenAI Gymnasium interface that simulates market conditions.
2. **Agents**: RL agents that learn trading policies through interaction with the environment.
3. **Trainers**: Utilities for training and evaluating RL agents.

## Components

### Environment

The `TradingEnvironment` class simulates a trading environment where agents can:

-   Observe market data and account information
-   Take trading actions (buy/sell, position sizing, stop-loss/take-profit levels)
-   Receive rewards based on trading performance

### Agents

Two main types of agents are provided:

1. **DQN Agent**: A Deep Q-Network implementation for discrete action spaces.
2. **Actor-Critic Agent**: An implementation that handles continuous action spaces, allowing for more fine-grained control over trading parameters.

Both agents use experience replay to learn efficiently from past experiences.

### Trainers

Training and evaluation utilities:

1. **RLTrainer**: Manages the training process, including logging, saving models, and tracking performance metrics.
2. **RLEvaluator**: Evaluates trained agents on test data to assess performance.

## Installation Requirements

To use this RL trading system, you'll need to install the following dependencies:

```bash
pip install tensorflow numpy pandas matplotlib gymnasium
```

## Usage Examples

The `examples` directory contains sample scripts demonstrating how to use the system:

1. **DQN Trading Agent**: A basic example using DQN for trading.
2. **Actor-Critic Trading Agent**: A more advanced example using Actor-Critic for continuous action spaces.

### Basic Example

```python
from pytrad.candle import CandleCollection
from pytrad.rl.environment import TradingEnvironment
from pytrad.rl.agent import DQNAgent
from pytrad.rl.trainer import RLTrainer

# Load market data
candle_collection = load_data()  # Your data loading function

# Create environment
env = TradingEnvironment(
    candle_collection=candle_collection,
    initial_balance=10000.0,
    window_size=20
)

# Initialize agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    learning_rate=0.001,
    gamma=0.99
)

# Train agent
trainer = RLTrainer(environment=env, agent=agent, episodes=100)
trainer.train()
```

## Customization

The RL trading system is designed to be modular and extensible:

-   **Custom Environments**: You can extend the `TradingEnvironment` class to incorporate additional market features or custom reward functions.
-   **Custom Agents**: You can implement new agent architectures by following the interface of existing agents.
-   **Custom Reward Functions**: Modify the environment's reward calculation to implement different trading objectives.

## Performance Considerations

-   **Data Preprocessing**: Normalize price data before feeding it to the environment for better learning performance.
-   **Hyperparameter Tuning**: The performance of RL agents is sensitive to hyperparameters; experiment to find optimal settings.
-   **Exploration vs. Exploitation**: Balance exploration and exploitation during training to discover optimal trading strategies.

## Future Extensions

Possible extensions to this system include:

1. Support for additional RL algorithms (PPO, SAC, etc.)
2. Multi-asset portfolio management
3. Risk-adjusted reward functions
4. Integration with real-time market data sources

## License

This project is licensed under the MIT License - see the LICENSE file for details.
