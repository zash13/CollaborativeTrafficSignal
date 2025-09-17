# Collaborative Traffic Signal Control with DQN

this project implements a deep q-network (DQN)-based agent for adaptive traffic signal control in a sumo (simulation of urban mobility) environment. the goal is to reduce congestion, minimize waiting times, and improve throughput in a simulated 4-way traffic intersection.

---

## Requirements

1. **Set UP python**
   - Install required packages:
     `pip install -r requirements.txt`
2. **Set Up SUMO**
   SUMO is required for running simulations.
   You can either:

   - Use your own SUMO installation
   - Use the provided submodule
     Choose one of the following options:

   1. Use the included SUMO submodule
      Clone with submodules
      `git clone --recurse-submodules https://github.com/zash13/CollaborativeTrafficSignal.git `
      Build SUMO from source:

      ```
      cd V3/sumo
      cmake .
      make -j$(nproc)
      export SUMO_HOME=$(pwd)/V3/sumo
      ```

   2. **Using Your Own SUMO Installation**

   - Ensure SUMO is installed (e.g., via `sudo apt-get install sumo` on Ubuntu or `yay -S sumo sumo-doc` on arch or download from the [SUMO website](https://sumo.dlr.de/docs/Downloads.php)).

   - Set the `SUMO_HOME` environment variable to your SUMO installation directory:
     `export SUMO_HOME=/usr/share/sumo`

## run the project

> `python train.py`

## Project Structure

- V3/DQN/
  Submodule: DQN agent implementation
- V3/sumo/
  Submodule: SUMO (optional if you use system SUMO)
- sumo_env.py
  SUMO environment wrapper (Gym-like API: initialize_simulation, take_action, close).
- train.py
  use dqn agent on simulation

---

1. **visualization of simulation**

   - To enable GUI visualization, edit `train.py` and set `USE_GUI = True` before running. For headless (faster) training, keep `USE_GUI = False` (default).
   - The script trains for 100 episodes by default. You can adjust `num_episodes` or `max_steps_per_episode` in `train.py` as needed.
     - Keep in mind that changing the number of **episodes** may also require adjusting **epsilon_decay** to maintain proper exploration behavior.

2. **Monitor Training**

- Verbosity is set to 1, printing progress every 10 steps (e.g., reward, epsilon, vehicle count).
- A plot of total rewards per episode is displayed at the end.

3. **Output**

- Training progress and final reward plot are saved to the console and a matplotlib window, respectively.

## About the Project

This project simulates a 4-way intersection using SUMO and trains a DQN agent to control traffic lights. The environment (`sumo_env.py`) provides a reinforcement learning interface with:

- **State**: A 20-dimensional vector (vehicle counts on incoming edges, padded with zeros).
- **Action**: Discrete phase selection for the traffic light.
- **Reward**: Negative sum of vehicle counts (to minimize congestion).

The DQN agent learns to optimize traffic flow over time, with random vehicle spawning (0.2-2s intervals, max 200 vehicles) and a hardcoded speed of 13.89 m/s (≈50 km/h).

## Agent Configuration Variables

The DQN agent is created with the following parameters in `train.py`:

- **AgentType**: Type of DQN agent (options from `AgentType` enum):
- `DQN` (default)
- `DOUBLE_DQN`
- `DUELING_DQN`
- **action_size**: Number of discrete actions (set to the number of traffic light phases).
- **state_size**: Dimension of the observation vector (set to `Config.OBS_DIM`, default 20).
- **learning_rate**: Learning rate for the neural network (default 0.001).
- **gamma**: Discount factor for future rewards (default 0.99).
- **epsilon**: Initial exploration rate (default 1.0, decays to `epsilon_min`).
- **batch_size**: Size of the experience replay batch (default 32).
- **buffer_size**: Maximum size of the replay buffer (default 2000).
- **max_episodes**: Total number of training episodes (default 100).
- **epsilon_min**: Minimum exploration rate (default 0.1).
- **epsilon_decay**: Decay rate for epsilon (default 0.995).
- **epsilon_policy**: Exploration policy type (default `EpsilonPolicyType.DECAY`).
- **reward_policy**: Reward handling policy (options from `RewardPolicyType` enum):
- `NONE` (0): No special reward processing.
- `ERM` (1): Expected Reward Maximization (default).
- **progress_bonus**: Bonus for training progress (default 0.05).
- **exploration_bonus**: Bonus for exploration (default 0.1).

Adjust these parameters in `train.py` to tune the agent's performance based on your needs.

## Submodule References

- This is an excellent resource for learning how to use SUMO in a short time. Check it out if you're interested in creating your own traffic network:

  - [SUMO Tutorials](https://sumo.dlr.de/docs/Tutorials/index.html)

- To learn more about the DQN agent submodule, explore the following repositories:

  - [DQN Agent](https://github.com/zash13/DQN_Agent)
  - [Maze Solver with Deep Q-Network](https://github.com/zash13/DQN_Maze)

- The DQN agent still needs improvement, especially in the **reward policies** and **epsilon decay strategy**.  
  I'm happy to collaborate so feel free to contribute to these repositories!
