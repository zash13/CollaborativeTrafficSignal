#!/usr/bin/env python3
import os
import sys
import numpy as np
from DQN.DQN_Agent import AgentFactory, AgentType
from sumo_env import SumoEnv
from config import Config

CHECKPOINT_BASE = "checkpoints/v1/model.keras"


def run_prediction(env, num_episodes=1, max_steps_per_episode=Config.MAX_STEPS):
    action_size = len(env.phases[env.tls_ids[0]])
    obs_dim = env.obs_dim

    agent = AgentFactory.create_agent(
        AgentType.DUELING_DQN,
        action_size=action_size,
        state_size=obs_dim,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=0.0,
        batch_size=32,
        buffer_size=50000,
        max_episodes=num_episodes,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        epsilon_policy=None,
        reward_policy=None,
        fc1_units=128,
        fc2_units=128,
        update_factor=0.005,
        update_target_network_method=None,
        target_update_frequency=50,
    )

    required_files = [
        CHECKPOINT_BASE + "_online.keras",
        CHECKPOINT_BASE + "_target.keras",
        CHECKPOINT_BASE + "_meta.json",
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"[error] missing checkpoint file: {f}")
            sys.exit(1)

    # Load from checkpoint
    agent.load(CHECKPOINT_BASE)
    print(f"[INFO] Loaded model from {CHECKPOINT_BASE}")

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done and step < max_steps_per_episode:
            action = agent.select_action(obs.reshape(1, -1))
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            total_reward += reward
            step += 1

        print(
            f"[PREDICT] Episode {episode + 1}/{num_episodes}, "
            f"Total Reward: {total_reward}, Vehicles: {info.get('active_vehicles', 0)}"
        )

    env.close()


if __name__ == "__main__":
    if not os.path.exists(Config.NET_FILE):
        print(f"[ERROR] Net file not found: {Config.NET_FILE}")
        sys.exit(1)

    env = SumoEnv(Config)
    run_prediction(env, num_episodes=1)
