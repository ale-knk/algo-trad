#!/usr/bin/env python
"""
Ejemplo de uso del sistema de Reinforcement Learning para trading.
Este script muestra cómo configurar y entrenar un agente ActorCriticAgent
en un entorno de trading con datos históricos.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from pytrad.rl.agent import ActorCriticAgent
from pytrad.dataset import CandleDataset
from pytrad.rl.environment import TradingEnvironment


def setup_dataset(timeframe: str = "H1") -> CandleDataset:
    """
    Configura el dataset con datos históricos y cálculo de indicadores.

    Args:
        timeframe: Timeframe de los candles ('M15', 'H1', 'D1', etc.)

    Returns:
        Dataset listo para el entrenamiento
    """
    # Configurar período de tiempo
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2020, 12, 31)

    # Usar lista vacía de indicadores para evitar errores de importación
    indicators = []

    # Configurar pares de divisas
    currency_pairs = ["EURUSD", "GBPUSD"]

    print(
        f"Cargando datos históricos para {currency_pairs} con timeframe {timeframe}..."
    )

    # Crear dataset
    dataset = CandleDataset(
        currency_pairs=currency_pairs,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        window_size=30,
        stride=1,
        indicators=indicators,
        normalize=True,
    )

    print(f"Dataset cargado con {len(dataset)} ventanas de datos")

    # Guardar estadísticas de normalización para uso futuro
    stats_dir = os.path.join("data", "normalization_stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, f"forex_{timeframe}_stats.json")
    dataset.save_normalization_stats(stats_file)

    return dataset


def setup_environment(dataset: CandleDataset) -> TradingEnvironment:
    """
    Configura el entorno de trading.

    Args:
        dataset: Dataset con datos históricos

    Returns:
        Entorno de trading configurado
    """
    env = TradingEnvironment(
        dataset=dataset,
        window_size=30,
        initial_balance=10000.0,
        transaction_cost=0.0001,  # 0.01% (1 pip aprox)
        reward_scaling=100.0,  # Escalar recompensas para facilitar aprendizaje
        risk_free_rate=0.02,  # 2% anual
        max_position_size=0.2,  # Máximo 20% del balance por operación
        use_dynamic_stop_loss=True,
        use_risk_based_position_sizing=True,
    )

    return env


def setup_agent(env: TradingEnvironment) -> ActorCriticAgent:
    """
    Configura el agente ActorCriticAgent.

    Args:
        env: Entorno de trading

    Returns:
        Agente configurado
    """
    # Obtener dimensiones del estado y acción
    state = env.reset()[0]
    state_dim = state.shape[1]  # [sequence_length, features]

    # Dimensión de acciones (tipo, tamaño, stop-loss, take-profit)
    action_dim = 4

    # Límites de las acciones
    action_low = np.array([0, 0, 0.01, 0.02])
    action_high = np.array([2, 1.0, 0.1, 0.2])

    # Configurar dispositivo (GPU si está disponible)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Crear agente
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bounds=(action_low, action_high),
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        memory_size=100000,
        batch_size=64,
        device=device,
        use_prioritized_replay=True,
    )

    return agent


def plot_training_history(
    history: Dict[str, List[float]], save_path: Optional[str] = None
):
    """
    Visualiza los resultados del entrenamiento.

    Args:
        history: Diccionario con métricas de entrenamiento
        save_path: Ruta para guardar la gráfica (opcional)
    """
    # Crear figura con 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Gráfico de recompensas
    axs[0, 0].plot(history["episode_rewards"])
    axs[0, 0].set_title("Recompensas por Episodio")
    axs[0, 0].set_xlabel("Episodio")
    axs[0, 0].set_ylabel("Recompensa")
    axs[0, 0].grid(True)

    # Gráfico de win rate
    axs[0, 1].plot(history["win_rates"])
    axs[0, 1].set_title("Win Rate por Episodio")
    axs[0, 1].set_xlabel("Episodio")
    axs[0, 1].set_ylabel("Win Rate")
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].grid(True)

    # Gráfico de Sharpe ratio
    axs[1, 0].plot(history["sharpe_ratios"])
    axs[1, 0].set_title("Sharpe Ratio por Episodio")
    axs[1, 0].set_xlabel("Episodio")
    axs[1, 0].set_ylabel("Sharpe Ratio")
    axs[1, 0].grid(True)

    # Gráfico de pérdidas
    if "actor_losses" in history and "critic_losses" in history:
        ax2 = axs[1, 1].twinx()
        (p1,) = axs[1, 1].plot(history["actor_losses"], "b-", label="Actor Loss")
        (p2,) = ax2.plot(history["critic_losses"], "r-", label="Critic Loss")
        axs[1, 1].set_title("Pérdidas del Modelo")
        axs[1, 1].set_xlabel("Episodio")
        axs[1, 1].set_ylabel("Actor Loss", color="b")
        ax2.set_ylabel("Critic Loss", color="r")
        axs[1, 1].grid(True)
        axs[1, 1].legend(handles=[p1, p2], loc="upper right")

    plt.tight_layout()

    # Guardar figura si se especifica ruta
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfica guardada en {save_path}")

    plt.show()


def main():
    """Función principal para configurar y ejecutar el entrenamiento."""
    # Configurar dataset
    dataset = setup_dataset(timeframe="H1")

    # Configurar entorno
    env = setup_environment(dataset)

    # Configurar agente
    agent = setup_agent(env)

    # Crear directorio para modelos
    models_dir = os.path.join("models", "rl")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "rl_trader")

    # Entrenar agente
    print("Iniciando entrenamiento...")
    history = agent.train(
        env=env,
        num_episodes=100,
        max_steps_per_episode=1000,
        warmup_steps=5000,
        update_interval=1,
        eval_interval=5,
        save_path=model_path,
        verbose=True,
    )

    # Visualizar resultados
    plot_training_history(
        history, save_path=os.path.join("results", "rl_training_history.png")
    )

    # Evaluar agente entrenado
    print("\nEvaluando agente entrenado...")
    eval_reward, eval_sharpe, eval_drawdown = agent._evaluate(env, num_episodes=10)
    print(f"Recompensa media: {eval_reward:.2f}")
    print(f"Sharpe Ratio: {eval_sharpe:.2f}")
    print(f"Drawdown máximo: {eval_drawdown:.2%}")

    # Guardar modelo final
    agent.save(f"{model_path}_final.pt")
    print(f"Modelo guardado en {model_path}_final.pt")


if __name__ == "__main__":
    main()
