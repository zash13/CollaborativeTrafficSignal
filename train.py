import numpy as np
from DQN.DQN_Agent import (
    AgentFactory,
    AgentType,
    EpsilonPolicyType,
    EpsilonPolicy,
    UpdateTargetNetworkType,
    RewardPolicyType,
)
from sumo.traditional_traffic.sumo_env import SumoEnvironment
import time


def main():
    # Environment configuration
    sumo_config = "traditional_traffic.sumo.cfg"
    gui = True  # Set to False for faster training
    max_steps = 360  # Adjusted for 10-step phase changes (3600 / 10)
    episodes = 50
    render = False

    # Initialize environment
    env = SumoEnvironment(sumo_config=sumo_config, gui=gui)

    # Agent configuration
    state_size = env.get_observation_shape()[0]  # 36
    action_size = env.get_action_count()  # 8 traffic light phases
    epsilon_min = 0.1
    epsilon_decay = 0.95
    ep_policy = EpsilonPolicy(
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        policy=EpsilonPolicyType.DECAY,
    )
    agent = AgentFactory.create_agent(
        AgentType.DOUBLE_DQN,
        action_size=action_size,
        state_size=state_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=50,
        buffer_size=2000,
        max_episodes=episodes,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        epsilon_policy=ep_policy,
        reward_policy=RewardPolicyType.NONE,
        progress_bonus=0.05,
        exploration_bonus=0.1,
        update_target_network_method=UpdateTargetNetworkType.SOFT,
        update_factor=0.8,
        target_update_frequency=5,
        reward_range=(-10, 20),
        use_normalization=False,
    )

    success_count = 0
    for episode in range(episodes):
        state = env.initialize_simulation()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.select_action(np.array(state))
            next_state, reward, terminated, truncated, info = env.take_action(action)
            done = terminated or truncated
            heuristic = None
            agent.store_experience(state, next_state, reward, action, done, heuristic)
            loss = agent.train(episode)
            state = next_state
            total_reward += reward
            steps += 1
            if render:
                time.sleep(0.1)

        if done:
            success_count += 1
        print(
            f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, "
            f"Steps: {steps}, Epsilon: {agent.get_epsilon():.3f}, "
            f"Loss: {loss}, Success Rate: {success_count / (episode + 1):.2%}, "
            f"{'Simulation Ended' if done else 'Max Steps Reached'}, "
            f"Avg Waiting Time: {info.get('total_waiting_time', 0) / max(1, traci.vehicle.getIDCount()):.2f}s, "
            f"Throughput: {info.get('throughput', 0)}"
        )

    env.close()


if __name__ == "__main__":
    main()
