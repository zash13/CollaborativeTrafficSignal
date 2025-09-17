import gymnasium as gym
import numpy as np
import time
from DQN.DQN_Agent import (
    AgentFactory,
    AgentType,
    EpsilonPolicyType,
    EpsilonPolicy,
    UpdateTargetNetworkType,
    RewardPolicyType,
)
import sumo_rl

import matplotlib.pyplot as plt


def main():
    # Correct environment ID and adjusted paths
    net_file = "sumo-rl/nets/2way-3tlC/2way-3tlC.net.xml"  # Adjusted path
    route_file = "sumo-rl/nets/2way-3tlC/2way-3tlC.rou.xml"  # Adjusted path
    env = gym.make(
        "sumo-rl-v0",  # Correct environment ID
        net_file=net_file,
        route_file=route_file,
        out_csv_name="sumo-rl/outputs/multi.csv",  # Adjusted path
        use_gui=False,
        num_seconds=3600,
        delta_time=5,  # Decision interval (seconds)
        yellow_time=4,  # Yellow phase duration (seconds)
        single_agent=True,  # Single agent for one traffic light
        # Remove num_agents for single-agent mode
    )

    obs_dim = env.observation_space.shape[0]  # e.g., 17 for 2way-3tlC
    action_size = env.action_space.n  # e.g., 4 phases

    epsilon_min = 0.1
    epsilon_decay = 0.995
    ep_policy = EpsilonPolicy(
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        policy=EpsilonPolicyType.DECAY,
    )

    num_episodes = 1000
    max_steps = 720  # e.g., 3600s / 5s delta_time
    render = False  # SUMO GUI via use_gui above

    agent = AgentFactory.create_agent(
        AgentType.DQN,
        action_size=action_size,
        state_size=obs_dim,  # Vector dim (no discrete n)
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=32,
        buffer_size=2000,
        max_episodes=num_episodes,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        epsilon_policy=ep_policy,
        reward_policy=RewardPolicyType.ERM,
        progress_bonus=0.05,
        exploration_bonus=0.1,
    )

    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()  # obs is np.array shape (obs_dim,)
        total_reward = 0
        step = 0
        done = False

        while not done and step < max_steps:
            # Select action (pass reshaped vector; agent must handle it)
            action = agent.select_action(obs.reshape(1, -1))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store raw vectors (agent must handle)
            agent.store_experience(obs, next_obs, reward, action, done, huristic=None)
            loss = agent.train(episode)

            obs = next_obs
            total_reward += reward
            step += 1

            if render:  # Won't work with LIBSUMO_AS_TRACI=1
                time.sleep(0.05)

        rewards.append(total_reward)
        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, "
            f"Epsilon: {agent.get_epsilon():.3f}, Loss: {loss}"
        )

    env.close()

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress on Four-Way SUMO Intersection")
    plt.show()


if __name__ == "__main__":
    main()
