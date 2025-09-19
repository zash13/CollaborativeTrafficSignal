#!/usr/bin/env python3
from DQN.DQN_Agent import (
    AgentFactory,
    AgentType,
    EpsilonPolicyType,
    EpsilonPolicy,
    UpdateTargetNetworkType,
    RewardPolicyType,
)
from sumo_env import SumoEnv
import os
import sys


EPOCHES = 100


class Config:
    SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")  # Non-GUI for training
    NET_FILE = os.path.join(os.getcwd(), "1way.net.xml")
    ROUTE_FILE = os.path.join(os.getcwd(), "1way.rou.xml")
    SUMOCFG_FILE = os.path.join(os.getcwd(), "1way.sumocfg")  # Updated to 1way
    SIM_STEP = 1.0
    MAX_STEPS = 100
    MAX_VEHICLES = 10
    SPAWN_MIN_INTERVAL = 0.2
    SPAWN_MAX_INTERVAL = 2.0
    DEFAULT_SPEED = 13.89  # m/s (≈50km/h)
    MIN_GREEN_TIME = 5.0
    OBS_DIM = 6  # Updated to match sumo_env.py


def train_dqn(env, num_episodes=EPOCHES, max_steps_per_episode=Config.MAX_STEPS):
    action_size = len(env.phases)
    obs_dim = Config.OBS_DIM
    epsilon_min = 0.1
    epsilon_decay = 0.995  # Faster decay
    ep_policy = EpsilonPolicy(
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        policy=EpsilonPolicyType.DECAY,
    )

    agent = AgentFactory.create_agent(
        AgentType.DQN,
        action_size=action_size,
        state_size=obs_dim,
        learning_rate=0.0001,  # Lowered
        gamma=0.99,
        epsilon=1.0,
        batch_size=32,
        buffer_size=10000,  # Increased
        max_episodes=num_episodes,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        epsilon_policy=ep_policy,
        reward_policy=RewardPolicyType.NONE,
        fc1_units=64,
        fc2_units=64,
        update_target_network_method=UpdateTargetNetworkType.SOFT,
        update_factor=0.005,
        target_update_frequency=10,
    )

    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        step = 0
        done = False
        prev_score = None  # track vehicles+waiting from previous step

        while not done and step < max_steps_per_episode:
            action = agent.select_action(obs.reshape(1, -1))
            next_obs, raw_reward, done, info = env.step(action)

            # Extract components (your env puts them in obs[0:2])
            vehicles = next_obs[0]
            waiting = next_obs[1]
            score = vehicles + waiting

            # Use delta: positive if situation improved vs last step
            if prev_score is None:
                reward = -score  # first step, fall back to absolute
            else:
                reward = prev_score - score
            prev_score = score

            agent.store_experience(obs, next_obs, reward, action, done, huristic=None)
            loss = agent.train(episode)

            obs = next_obs
            total_reward += reward
            step += 1

        rewards.append(total_reward)
        print(
            f"Episode {episode + 1}/{num_episodes}, "
            f"Total Reward: {total_reward:.2f}, "
            f"Epsilon: {agent.get_epsilon():.3f}, "
            f"Loss: {loss}, "
            f"Vehicles: {info.get('active_vehicles', 0)}"
        )

    env.close()

    # Plot rewards
    import matplotlib.pyplot as plt

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (delta-based)")
    plt.title("DQN Training on 1-Way SUMO Intersection")
    plt.show()

    agent.save("checkpoints/dqn_final")
    return rewards


if __name__ == "__main__":
    if not os.path.exists(Config.NET_FILE):
        print(f"[error] net file not found: {Config.NET_FILE}")
        sys.exit(1)

    # Example: Train DQN
    env = SumoEnv()
    rewards = train_dqn(env, num_episodes=EPOCHES)
    print("[INFO] Training complete.")
