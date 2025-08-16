#!/usr/bin/env python3
"""
Example of training an Actor-Critic agent for trading.
This demonstrates a more advanced approach for continuous action spaces.
"""

import os
from datetime import datetime, timedelta

import numpy as np

from pytrad.candle import Candle, CandleCollection
from pytrad.rl.model import ActorCriticAgent
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

    # Generate 1000 candles with some price movement and trends
    for i in range(1000):
        time = datetime(2023, 1, 1) + timedelta(hours=i)

        # Add some trending behavior (cycles)
        trend = 5 * np.sin(i / 100)

        # Generate some price movement
        random_walk = np.random.normal(0, 1)
        price_change = trend + random_walk * 0.5

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
        window_size=30,  # Larger window for more context
        commission=0.001,
        max_position_size=0.25,  # Max 25% of balance per trade
    )

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action bounds: {action_low} to {action_high}")

    print("Creating the Actor-Critic agent...")
    # Create the Actor-Critic agent
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(action_low, action_high),
        actor_lr=0.0001,
        critic_lr=0.001,
        gamma=0.99,
        tau=0.005,
        memory_size=100000,
        batch_size=64,
    )

    print("Setting up the trainer...")
    # Create directories for outputs
    os.makedirs("models/actor_critic", exist_ok=True)
    os.makedirs("logs/actor_critic", exist_ok=True)

    # Create the trainer
    trainer = RLTrainer(
        environment=env,
        agent=agent,
        episodes=100,  # More episodes for better learning
        model_dir="models/actor_critic",
        log_dir="logs/actor_critic",
    )

    print("Starting training...")
    # Train the agent
    trainer.train(save_best=True, eval_interval=10)

    print("Training complete!")

    print("Evaluating the trained agent...")
    # Evaluate the trained agent
    evaluator = RLEvaluator(
        environment=env,
        agent=agent,
        # Using the trained agent directly
    )

    # Run evaluation on multiple episodes
    results = evaluator.evaluate(episodes=10, render=True)

    print("\nEvaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Average Final Balance: {results['avg_balance']:.2f}")
    print(f"Average Number of Trades: {results['avg_trades']:.2f}")

    print("\nIndividual Episode Results:")
    for i, (reward, balance, trades) in enumerate(
        zip(results["all_rewards"], results["all_balances"], results["all_trades"])
    ):
        print(
            f"Episode {i + 1}: Reward={reward:.2f}, Balance={balance:.2f}, Trades={trades}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
