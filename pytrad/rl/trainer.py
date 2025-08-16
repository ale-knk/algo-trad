import os
import time
from datetime import datetime
from typing import Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytrad.rl.environment import TradingEnvironment


class RLTrainer:
    """
    Trainer class to manage the training process of RL agents.
    """

    def __init__(
        self,
        environment: TradingEnvironment,
        agent: Any,
        episodes: int = 100,
        model_dir: str = "models",
        log_dir: str = "logs",
    ):
        """
        Initialize the trainer.

        Args:
            environment: Training environment
            agent: RL agent (DQN or Actor-Critic)
            episodes: Number of episodes to train
            model_dir: Directory to save models
            log_dir: Directory to save logs
        """
        self.env = environment
        self.agent = agent
        self.episodes = episodes
        self.model_dir = model_dir
        self.log_dir = log_dir

        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Training metrics
        self.episode_rewards = []
        self.episode_balances = []
        self.episode_trade_counts = []
        self.best_reward = -np.inf

    def train(
        self, save_best: bool = True, eval_interval: int = 5, render: bool = False
    ):
        """
        Train the agent.

        Args:
            save_best: Whether to save the best model
            eval_interval: Evaluate model every n episodes
            render: Whether to render the environment during training
        """
        start_time = time.time()

        for episode in range(1, self.episodes + 1):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            # Episode loop
            while not done:
                # Select action
                action = self.agent.act(state)

                # Take action in environment
                next_state, reward, done, truncated, info = self.env.step(action)

                # Store experience in agent's memory
                self.agent.remember(state, action, reward, next_state, done)

                self.agent.learn()

                state = next_state
                episode_reward += reward

                if render:
                    self.env.render()

            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_balances.append(info["balance"])
            self.episode_trade_counts.append(info["trades_executed"])

            # Update best model
            if save_best and episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model("best_model")

            # Evaluation
            if episode % eval_interval == 0:
                self._log_progress(episode, time.time() - start_time)

                # Save checkpoint
                self.save_model(f"checkpoint_{episode}")

        # Save final model
        self.save_model("final_model")

        # Save training history
        self._save_training_history()

        # Plot training results
        self._plot_training_results()

    def _log_progress(self, episode: int, elapsed_time: float):
        """Log training progress."""
        avg_reward = np.mean(self.episode_rewards[-5:])
        avg_balance = np.mean(self.episode_balances[-5:])
        avg_trades = np.mean(self.episode_trade_counts[-5:])

        print(f"Episode: {episode}/{self.episodes}")
        print(f"Elapsed Time: {elapsed_time:.2f}s")
        print(f"Average Reward (last 5): {avg_reward:.2f}")
        print(f"Average Balance (last 5): {avg_balance:.2f}")
        print(f"Average Trades (last 5): {avg_trades:.2f}")
        print("-" * 50)

    def save_model(self, name: str):
        """Save agent model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        actor_filepath = f"{self.model_dir}/actor_{name}_{timestamp}.pt"
        critic_filepath = f"{self.model_dir}/critic_{name}_{timestamp}.pt"
        self.agent.save(actor_filepath, critic_filepath)

    def _save_training_history(self):
        """Save training metrics to CSV."""
        history = pd.DataFrame(
            {
                "episode": range(1, len(self.episode_rewards) + 1),
                "reward": self.episode_rewards,
                "balance": self.episode_balances,
                "trades": self.episode_trade_counts,
            }
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history.to_csv(f"{self.log_dir}/training_history_{timestamp}.csv", index=False)

    def _plot_training_results(self):
        """Plot training metrics."""
        plt.figure(figsize=(15, 10))

        # Plot rewards
        plt.subplot(3, 1, 1)
        plt.plot(self.episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Plot balances
        plt.subplot(3, 1, 2)
        plt.plot(self.episode_balances)
        plt.title("Final Account Balance")
        plt.xlabel("Episode")
        plt.ylabel("Balance")

        # Plot trade counts
        plt.subplot(3, 1, 3)
        plt.plot(self.episode_trade_counts)
        plt.title("Trades per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Number of Trades")

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.log_dir}/training_results_{timestamp}.png")
        plt.close()


class RLEvaluator:
    """
    Class for evaluating trained RL agents.
    """

    def __init__(
        self,
        environment: TradingEnvironment,
        agent: Any,  # DQNAgent or ActorCriticAgent
        model_path: Union[Tuple[str, str], None] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            environment: Evaluation environment
            agent: RL agent
            model_path: Path to saved model (if loading pre-trained model)
                        For DQNAgent: string path to model file
                        For ActorCriticAgent: tuple of (actor_path, critic_path)
        """
        self.env = environment
        self.agent = agent

        actor_path, critic_path = model_path
        self.agent.load(actor_path, critic_path)

    def evaluate(self, episodes: int = 1, render: bool = True):
        """
        Evaluate the agent.

        Args:
            episodes: Number of episodes to evaluate
            render: Whether to render the environment

        Returns:
            Dictionary of evaluation metrics
        """
        all_rewards = []
        all_balances = []
        all_trade_counts = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.agent.act(state, add_noise=False)

                # Take action
                next_state, reward, done, truncated, info = self.env.step(action)
                state = next_state
                episode_reward += reward

                if render:
                    self.env.render()

            # Collect metrics
            all_rewards.append(episode_reward)
            all_balances.append(info["balance"])
            all_trade_counts.append(info["trades_executed"])

            print(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Balance = {info['balance']:.2f}, Trades = {info['trades_executed']}"
            )

        # Compile results
        results = {
            "avg_reward": np.mean(all_rewards),
            "avg_balance": np.mean(all_balances),
            "avg_trades": np.mean(all_trade_counts),
            "all_rewards": all_rewards,
            "all_balances": all_balances,
            "all_trades": all_trade_counts,
        }

        return results
