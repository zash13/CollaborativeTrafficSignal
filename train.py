#!/usr/bin/env python3
from DQN.DQN_Agent import (
    AgentFactory,
    AgentType,
    EpsilonPolicyType,
    EpsilonPolicy,
    UpdateTargetNetworkType,
    RewardPolicyType,
)
import os
import sys
from sumo_env import SumoEnv  # Import after Config definition
from config import Config

EPOCHES = 100


def train_dqn(env, num_episodes=EPOCHES, max_steps_per_episode=Config.MAX_STEPS):
    action_size = len(env.phases[env.tls_ids[0]])
    obs_dim = env.obs_dim
    epsilon_min = 0.1
    epsilon_decay = 0.995
    ep_policy = EpsilonPolicy(
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        policy=EpsilonPolicyType.DECAY,
    )

    agent = AgentFactory.create_agent(
        AgentType.DQN,
        action_size=action_size,
        state_size=obs_dim,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=32,
        buffer_size=50000,
        max_episodes=num_episodes,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        epsilon_policy=ep_policy,
        reward_policy=RewardPolicyType.NONE,
        fc1_units=128,
        fc2_units=128,
        update_factor=0.005,
        update_target_network_method=UpdateTargetNetworkType.SOFT,
        target_update_frequency=50,
    )

    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done and step < max_steps_per_episode:
            action = agent.select_action(obs.reshape(1, -1))
            next_obs, reward, done, info = env.step(action)
            agent.store_experience(obs, next_obs, reward, action, done, huristic=None)
            loss = agent.train(episode)
            obs = next_obs
            total_reward += reward
            step += 1

        rewards.append(total_reward)
        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, "
            f"Epsilon: {agent.get_epsilon():.3f}, Loss: {loss}, Vehicles: {info.get('active_vehicles', 0)}"
        )

    env.close()

    import matplotlib.pyplot as plt

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training on 4-Way SUMO Intersection")
    plt.show()
    agent.save("checkpoints/dqn_final")
    return rewards


if __name__ == "__main__":
    if not os.path.exists(Config.NET_FILE):
        print(f"[ERROR] Net file not found: {Config.NET_FILE}")
        sys.exit(1)
    env = SumoEnv(Config)
    rewards = train_dqn(env, num_episodes=EPOCHES)
    print("[INFO] Training complete.")
