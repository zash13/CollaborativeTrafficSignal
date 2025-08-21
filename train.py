# Force CPU usage before any TensorFlow imports
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs except errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

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
from DQN.logHandler import Logging, Verbosity
import time
import csv
import traci


def main():
    sumo_config = "traditional_traffic.sumo.cfg"
    gui = False
    max_steps = 720  # 3600s / 5s
    episodes = 100
    render = False

    # Initialize logging for reward status
    logger = Logging(
        "reward_log.txt",
        verbosity=Verbosity.DEBUG,
        user_name="TrafficDQN",
    )

    env = SumoEnvironment(sumo_config=sumo_config, gui=gui)

    state_size = env.get_observation_shape()[0]  # 44
    action_size = env.get_action_count()  # 8
    epsilon_min = 0.1
    epsilon_decay = 0.99
    ep_policy = EpsilonPolicy(
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        policy=EpsilonPolicyType.DECAY,
    )
    agent = AgentFactory.create_agent(
        AgentType.DOUBLE_DQN,
        action_size=action_size,
        state_size=state_size,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
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
        target_update_frequency=10,
        reward_range=(-100, 100),
        use_normalization=False,
    )

    with open("training_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Episode",
                "Total Reward",
                "Steps",
                "Epsilon",
                "Loss",
                "Success Rate",
                "Avg Waiting Time",
                "Throughput",
            ]
        )

    success_count = 0
    for episode in range(episodes):
        state = env.initialize_simulation()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            state_batched = np.array(state)[np.newaxis, :]  # Shape: (1, 44)
            action = agent.select_action(state_batched)
            next_state, reward, terminated, truncated, info = env.take_action(action)
            done = terminated or truncated
            heuristic = None
            agent.store_experience(state, next_state, reward, action, done, heuristic)
            loss = agent.train(episode)
            state = next_state
            total_reward += reward
            steps += 1
            if render and gui:
                time.sleep(0.01)

            # Log reward components per step (debug level)
            logger.debug(
                "Step {}: Reward = {:.2f}, Waiting Time = {:.2f}, Queue Length = {}, Stopped Vehicles = {}, Throughput = {}, Phase Penalty = {:.2f}",
                steps,
                reward,
                info.get("total_waiting_time", 0),
                info.get("total_queue_length", 0),
                info.get("stopped_vehicles", 0),
                info.get("throughput", 0),
                info.get("phase_change_penalty", 0),
                show_in_console=False,
            )

        if done:
            success_count += 1
        avg_waiting_time = info.get("total_waiting_time", 0) / max(
            1, traci.vehicle.getIDCount()
        )
        print(
            f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, "
            f"Steps: {steps}, Epsilon: {agent.get_epsilon():.3f}, "
            f"Loss: {loss:.4f}, Success Rate: {success_count / (episode + 1):.2%}, "
            f"{'Simulation Ended' if done else 'Max Steps Reached'}, "
            f"Avg Waiting Time: {avg_waiting_time:.2f}s, "
            f"Throughput: {info.get('throughput', 0)}"
        )
        with open("training_metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode + 1,
                    total_reward,
                    steps,
                    agent.get_epsilon(),
                    loss,
                    success_count / (episode + 1),
                    avg_waiting_time,
                    info.get("throughput", 0),
                ]
            )

        # Log episode summary (info level)
        logger.info(
            "Episode {}: Total Reward = {:.2f}, Avg Waiting Time = {:.2f}s, Throughput = {}",
            episode + 1,
            total_reward,
            avg_waiting_time,
            info.get("throughput", 0),
            show_in_console=True,
        )

    env.close()


if __name__ == "__main__":
    main()
