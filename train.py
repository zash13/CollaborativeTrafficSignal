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


EPOCHES = 100


class Config:
    SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")
    NET_FILE = os.path.join(os.getcwd(), "my_4way.net.xml")
    ROUTE_FILE = os.path.join(os.getcwd(), "my_4way.rou.xml")
    SUMOCFG_FILE = os.path.join(os.getcwd(), "my_4way.sumocfg")
    SIM_STEP = 1.0
    MAX_STEPS = 600
    MAX_VEHICLES = 50
    SPAWN_MIN_INTERVAL = 0.2
    SPAWN_MAX_INTERVAL = 2.0
    DEFAULT_SPEED = 13.89  # m/s (≈50km/h)
    MIN_GREEN_TIME = 5.0
    OBS_DIM = 20


def train_dqn(env, num_episodes=EPOCHES, max_steps_per_episode=Config.MAX_STEPS):
    action_size = len(env.phases)
    obs_dim = Config.OBS_DIM
    epsilon_min = 0.1
    epsilon_decay = 0.99
    ep_policy = EpsilonPolicy(
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        policy=EpsilonPolicyType.DECAY,
    )

    agent = AgentFactory.create_agent(
        AgentType.DQN,
        action_size=action_size,
        state_size=obs_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=32,
        buffer_size=2000,
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

        while not done and step < max_steps_per_episode:
            # DQN action on vector obs
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
    # Plot rewards (adapt from your plt code)
    import matplotlib.pyplot as plt

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training on 4-Way SUMO Intersection")
    plt.show()
    return rewards


if __name__ == "__main__":
    if not os.path.exists(Config.NET_FILE):
        print(f"[ERROR] Net file not found: {Config.NET_FILE}")
        sys.exit(1)

    # Example: Train DQN
    env = SumoEnv()
    rewards = train_dqn(env, num_episodes=100)  # Or run basic loop if no DQN
    print("[INFO] Training complete.")
