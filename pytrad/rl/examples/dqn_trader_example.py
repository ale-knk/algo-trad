#!/usr/bin/env python3
"""
Example of training a DQN agent for trading.
"""

import os
from datetime import datetime, timedelta

import numpy as np

from pytrad.candle import Candle, CandleCollection
from pytrad.rl.model import DQNAgent
from pytrad.rl.environment import TradingEnvironment
from pytrad.rl.trainer import RLEvaluator, RLTrainer


def load_sample_data():
    """
    Load sample data for training.
    In a real application, this would load from your database or files.
    """
    # Create sample candle data
    candles = []
    base_price = 100.0

    # Generate 1000 candles with some price movement
    for i in range(1000):
        time = datetime(2023, 1, 1) + timedelta(hours=i)

        # Generate some price movement
        random_walk = np.random.normal(0, 1)
        price_change = random_walk * 0.5

        # Make sure prices follow a realistic pattern
        open_price = base_price
        close_price = base_price + price_change
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.2))
        volume = abs(np.random.normal(1000, 300))

        # Create and add the candle
        candle = Candle(
            time=time,
            open_price=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        candles.append(candle)

        # Update base price for next candle
        base_price = close_price

    return CandleCollection(candles)


def main():
    """Main function to run the example."""
    print("Loading data...")
    candle_collection = load_sample_data()

    print("Setting up the environment...")
    # Create the RL environment
    env = TradingEnvironment(
        candle_collection=candle_collection,
        initial_balance=10000.0,
        window_size=20,
        commission=0.001,
        max_position_size=0.2,  # Max 20% of balance per trade
    )

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("Creating the agent...")
    # Create the DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
        batch_size=64,
        memory_size=10000,
    )

    print("Setting up the trainer...")
    # Create directories for outputs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create the trainer
    trainer = RLTrainer(
        environment=env,
        agent=agent,
        episodes=50,  # For example purposes, use a small number
        model_dir="models",
        log_dir="logs",
    )

    print("Starting training...")
    # Train the agent
    trainer.train(save_best=True, eval_interval=5)

    print("Training complete!")

    print("Evaluating the trained agent...")
    # Evaluate the trained agent
    evaluator = RLEvaluator(
        environment=env,
        agent=agent,
        # No need to load a model since we're using the already trained agent
    )

    # Run evaluation
    results = evaluator.evaluate(episodes=5, render=True)

    print("\nEvaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Average Final Balance: {results['avg_balance']:.2f}")
    print(f"Average Number of Trades: {results['avg_trades']:.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
