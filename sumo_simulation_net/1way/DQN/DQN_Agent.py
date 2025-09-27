from os import stat
from types import new_class
from matplotlib.pyplot import cla
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.layers import Dense, Input
from collections import deque
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional


class EpsilonPolicyType(Enum):
    NONE = 0
    DECAY = 1
    SOFTLINEAR = 2


class RewardPolicyType(Enum):
    NONE = 0
    ERM = 1


class UpdateTargetNetworkType(Enum):
    HARD = 0
    SOFT = 1


class ModelType(Enum):
    STANDARD = "standard"
    DUELING = "dueling"


class AgentType(Enum):
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"


INFINITY = float("inf")


class ExperienceBuffer:
    def __init__(
        self,
        rewarder: "RewardHelper",
        buffer_size=2000,
        prefer_lower_heuristic=True,
        reward_range=(0, 100),
        use_normalization=False,
    ) -> None:
        self.memory_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.reward_helper = rewarder
        self.prefer_lower_heuristic = prefer_lower_heuristic
        self.reward_range = reward_range
        self.use_normalization = use_normalization

    def store_experience(
        self, current_state, next_state, imm_reward, action, done, heuristic=0
    ):
        imm_reward = self.reward_helper.findReward(
            current_state, imm_reward, heuristic, self.prefer_lower_heuristic
        )
        if self.use_normalization:
            imm_reward = self.reward_helper.normalize_reward(
                self.reward_range, imm_reward
            )
        self.memory_buffer.append(
            {
                "current_state": current_state,
                "action": action,
                "reward": imm_reward,
                "next_state": next_state,
                "heuristic": heuristic,
                "done": done,
            }
        )

    def sample_batch(self, count):
        if len(self.memory_buffer) < count:
            return None
        batch = random.sample(self.memory_buffer, count)
        states = np.vstack([item["current_state"] for item in batch])
        next_states = np.vstack([item["next_state"] for item in batch])
        rewards = np.array([item["reward"] for item in batch])
        actions = np.array([item["action"] for item in batch])
        dones = np.array([item["done"] for item in batch])
        heuristics = np.array([item["heuristic"] for item in batch])
        return states, next_states, rewards, actions, dones, heuristics


class AbstractQNetwork(ABC):
    @abstractmethod
    @abstractmethod
    def predict(self, states: np.ndarray, verbose: int = 0) -> np.ndarray:
        """predict Q-values for given states."""
        pass

    @abstractmethod
    def fit(
        self,
        states: np.ndarray,
        q_targets: np.ndarray,
        epochs: int = 1,
        verbose: int = 0,
    ) -> float:
        """train the network on states and target Q-values, returning the loss."""
        pass

    @abstractmethod
    def set_weights(self, weights: list) -> bool:
        """set the network's weights."""
        pass

    @abstractmethod
    def get_weights(self) -> list:
        """get the network's weights."""
        pass


class QNetworkFactory:
    @staticmethod
    def create_model(
        model_type: ModelType,
        state_size: int,
        action_size: int,
        fc1_units: int,
        fc2_units: int,
        learning_rate: float,
    ) -> AbstractQNetwork:
        if model_type == ModelType.STANDARD:
            return QNetwork(
                state_size, action_size, fc1_units, fc2_units, learning_rate
            )
        elif model_type == ModelType.DUELING:
            return DuelingQNetwork(
                state_size, action_size, fc1_units, fc2_units, learning_rate
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class QNetwork(AbstractQNetwork):
    def __init__(
        self,
        state_size,
        action_size,
        fc1_units: int,
        fc2_units: int,
        learning_rate=0.001,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self._model = self._initiate_model()
        self._model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))

    def _initiate_model(self):
        return keras.Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(units=self.fc1_units, activation="relu"),
                Dense(units=self.fc2_units, activation="relu"),
                Dense(units=self.action_size, activation="linear"),
            ]
        )

    def predict(self, states: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self._model.predict(states, verbose=verbose)

    def fit(
        self,
        states: np.ndarray,
        q_targets: np.ndarray,
        epochs: int = 1,
        verbose: int = 0,
    ) -> float:
        history = self._model.fit(states, q_targets, epochs=epochs, verbose=verbose)
        return history.history["loss"][0]

    def set_weights(self, weights: list) -> bool:
        self._model.set_weights(weights)
        return True

    def get_weights(self) -> list:
        return self._model.get_weights()

    def save(self, path: str) -> None:
        """
        save the model to the given path.
        example: network.save("checkpoints/dqn_model")
        """
        self._model.save(path)

    @classmethod
    def load(cls, path: str) -> "QNetwork":
        """
        load a qnetwork from the given path.
        example: net = qnetwork.load("checkpoints/dqn_model")
        """
        model = keras.models.load_model(path)
        state_size = model.input_shape[-1]
        action_size = model.output_shape[-1]
        fc1_units = model.layers[1].units
        fc2_units = model.layers[2].units

        net = cls(state_size, action_size, fc1_units, fc2_units)
        net._model = model
        return net


# Dueling Q-Network:
# The main difference compared to a standard Q-network is that the network
# splits into two streams near the end: one for the state-value (V) and one for the advantage (A).
# These are then combined using the formula:
# Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
class DuelingQNetwork(AbstractQNetwork):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        fc1_units: int,
        fc2_units: int,
        learning_rate: float = 0.001,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self._model = self._initiate_model()
        self._model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))

    def _initiate_model(self):
        inputs = keras.Input(shape=(self.state_size,))
        x = keras.layers.Dense(self.fc1_units, activation="relu")(inputs)
        x = keras.layers.Dense(self.fc2_units, activation="relu")(x)
        value = keras.layers.Dense(1, activation=None)(x)
        advantages = keras.layers.Dense(self.action_size, activation=None)(x)
        # combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        mean_advantages = keras.layers.Lambda(
            lambda a: a - keras.backend.mean(a, axis=1, keepdims=True),
            output_shape=(int(self.action_size),),
        )(advantages)
        q_values = keras.layers.Add()([value, mean_advantages])
        return keras.Model(inputs=inputs, outputs=q_values)

    def predict(self, states: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self._model.predict(states, verbose=verbose)

    def fit(
        self,
        states: np.ndarray,
        q_targets: np.ndarray,
        epochs: int = 1,
        verbose: int = 0,
    ) -> float:
        history = self._model.fit(states, q_targets, epochs=epochs, verbose=verbose)
        return history.history["loss"][0]

    def set_weights(self, weights: list) -> bool:
        self._model.set_weights(weights)
        return True

    def get_weights(self) -> list:
        return self._model.get_weights()


# we learnd that agent should balance between exploration and exploitation
# without this section , it seems that agent just try to explor new path
#
class EpsilonPolicy:
    def __init__(
        self,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        policy: EpsilonPolicyType = EpsilonPolicyType.DECAY,
        update_per_episod: bool = True,
    ):
        self.policy = policy
        self.visited_states = {}
        self.epsilon: float = INFINITY
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_per_episod = update_per_episod
        self.previous_episode = 0

    def adjust_epsilon(
        self,
        epsilon,
        episode_count,
        max_episodes=100,
    ):
        self.epsilon = epsilon
        if self.update_per_episod:
            if self.previous_episode == episode_count:
                return self.epsilon
            else:
                self.previous_episode = episode_count
        if self.policy == EpsilonPolicyType.DECAY:
            return self.linear_decay(episode_count, max_episodes)
        elif self.policy == EpsilonPolicyType.SOFTLINEAR:
            return self.soft_linear_decay(episode_count, max_episodes)
        else:
            return self.epsilon

    def linear_decay(self, episode_count, max_episodes):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon

    def soft_linear_decay(self, episode_count, max_episodes=200):
        if self.epsilon > self.epsilon_min:
            target_epsilon = max(
                self.epsilon_min,
                1.0 - (1.0 - self.epsilon_min) * (episode_count / max_episodes),
            )
            self.epsilon = self.epsilon * self.epsilon_decay + target_epsilon * (
                1 - self.epsilon_decay
            )
        return self.epsilon


class RewardHelper:
    def __init__(
        self,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
        policy: RewardPolicyType = RewardPolicyType.ERM,
    ):
        self.progress_bonus = progress_bonus
        self.exploration_bonus = exploration_bonus
        self.policy = policy
        self.old_huristic = INFINITY
        self.visited_states = {}

    def findReward(self, state, reward, heuristic=None, lowerHuristicBetter=True):
        if self.policy == RewardPolicyType.ERM:
            return self._ERM(state, reward, heuristic, lowerHuristicBetter)
        elif self.policy == RewardPolicyType.NONE:
            return self._none(reward)
        else:
            return reward

    def _none(self, reward):
        return reward

    def _ERM(self, state, reward, heuristic=None, lowerHuristicBetter=True):
        state_key = tuple(state.flatten()) if state is not None else None
        is_new_state = state_key is not None and state_key not in self.visited_states
        if is_new_state:
            if is_new_state and state_key is not None:
                self.visited_states[state_key] = (
                    heuristic if heuristic is not None else INFINITY
                )

        progress = 0.0
        if heuristic is not None and state_key is not None:
            old_heuristic = self.visited_states[state_key]
            if (
                old_heuristic != INFINITY
                and (heuristic < old_heuristic and lowerHuristicBetter)
                or (heuristic > old_heuristic and not lowerHuristicBetter)
            ):
                progress = self.progress_bonus
                self.visited_states[state_key] = heuristic

        new_reward = reward
        if is_new_state:
            new_reward += self.exploration_bonus
        if progress > 0:
            new_reward += self.progress_bonus
        return new_reward

    # be carefull with nomalization
    # and reward_range , it may keep you loss close to zero , while your model is learning nothing
    # in short , cast Qt = r + gamma*Q  , with smaller r , you will have smaller change , and since Q already predicted by model it self,
    # its like model fit on it own material , so , you will see small loss , with not progress
    def normalize_reward(self, reward_range, reward):
        min_r, max_r = reward_range
        if max_r == min_r:
            return 0.0
        norm = (reward - min_r) / (max_r - min_r)
        return norm * 2 - 1


class IAgent(ABC):
    @abstractmethod
    def train(self, episode):
        pass

    @abstractmethod
    def select_action(self, current_state):
        pass

    @abstractmethod
    def store_experience(self, state, next_state, reward, action, done, huristic):
        pass

    @abstractmethod
    def get_epsilon(self):
        pass


class AgentFactory:
    @staticmethod
    def create_agent(
        agent_type: AgentType,
        action_size: int,
        state_size: int,
        learning_rate: float = 0.001,
        buffer_size: int = 2000,
        batch_size: int = 32,
        gamma: float = 0.99,
        max_episodes: int = 200,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        epsilon_policy: Optional[EpsilonPolicy] = None,
        reward_policy: RewardPolicyType = RewardPolicyType.NONE,
        prefer_lower_heuristic: bool = True,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
        update_target_network_method: UpdateTargetNetworkType = UpdateTargetNetworkType.HARD,
        update_factor: float = 0.005,
        target_update_frequency: int = 10,
        reward_range: tuple = (0, 0),
        use_normalization: bool = False,
        fc1_units: int = 64,
        fc2_units: int = 64,
    ) -> IAgent:
        """
        create an agent instance based on the specified agent type.

        Args:
            agent_type: Type of agent to create (DQN, DOUBLE_DQN, DUELING_DQN)

        Returns:
            Interface IAgent.

        """
        # Common dependencies
        epsilon_policy = epsilon_policy or EpsilonPolicy(
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            policy=EpsilonPolicyType.DECAY,
        )
        reward_helper = RewardHelper(
            progress_bonus=progress_bonus,
            exploration_bonus=exploration_bonus,
            policy=reward_policy,
        )
        buffer_helper = ExperienceBuffer(
            reward_helper,
            buffer_size=buffer_size,
            prefer_lower_heuristic=prefer_lower_heuristic,
            reward_range=reward_range,
            use_normalization=use_normalization,
        )

        if agent_type == AgentType.DQN:
            return DQNAgent(
                action_size=action_size,
                state_size=state_size,
                learning_rate=learning_rate,
                buffer_helper=buffer_helper,
                batch_size=batch_size,
                gamma=gamma,
                max_episodes=max_episodes,
                epsilon=epsilon,
                epsilon_policy=epsilon_policy,
                fc1_units=fc1_units,
                fc2_units=fc2_units,
            )
        elif agent_type == AgentType.DOUBLE_DQN:
            return DoubleDQNAgent(
                action_size=action_size,
                state_size=state_size,
                learning_rate=learning_rate,
                buffer_helper=buffer_helper,
                batch_size=batch_size,
                gamma=gamma,
                max_episodes=max_episodes,
                epsilon=epsilon,
                epsilon_policy=epsilon_policy,
                update_target_network_method=update_target_network_method,
                update_factor=update_factor,
                target_update_frequency=target_update_frequency,
                fc1_units=fc1_units,
                fc2_units=fc2_units,
            )
        elif agent_type == AgentType.DUELING_DQN:
            return DuelingDQNAgent(
                action_size=action_size,
                state_size=state_size,
                learning_rate=learning_rate,
                buffer_helper=buffer_helper,
                batch_size=batch_size,
                gamma=gamma,
                max_episodes=max_episodes,
                epsilon=epsilon,
                epsilon_policy=epsilon_policy,
                update_target_network_method=update_target_network_method,
                update_factor=update_factor,
                target_update_frequency=target_update_frequency,
                fc1_units=fc1_units,
                fc2_units=fc2_units,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


class DQNAgent(IAgent):
    """single network Deep Q-learning model"""

    def __init__(
        self,
        action_size: int,
        state_size: int,
        learning_rate: float,
        buffer_helper: ExperienceBuffer,
        batch_size: int,
        gamma: float,
        max_episodes: int,
        epsilon: float,
        epsilon_policy: EpsilonPolicy,
        fc1_units: int,
        fc2_units: int,
    ) -> None:
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.episode_count = 0
        self.max_episodes = max_episodes
        self.model: "AbstractQNetwork"
        self.epsilon_policy = epsilon_policy
        self.buffer_helper = buffer_helper
        self._define_model(state_size, action_size, fc1_units, fc2_units, learning_rate)

    def _define_model(
        self, state_size, action_size, fc1_units, fc2_units, learning_rate
    ) -> None:
        self.model = QNetworkFactory.create_model(
            ModelType.STANDARD,
            state_size,
            action_size,
            fc1_units,
            fc2_units,
            learning_rate,
        )

    def train(self, episode):
        """use to train the model !! yep , so strange"""
        data = self.buffer_helper.sample_batch(self.batch_size)
        if data is None:
            return None
        states, next_states, rewards, actions, dones, heuristics = data
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
            else:
                q_targets[i, actions[i]] = rewards[i]
            # q_targets[i, actions[i]] = np.clip(q_targets[i, actions[i]], -10, 10)

        self._update_exploration_rate(episode_count=episode)
        loss = self.model.fit(states, q_targets, epochs=1, verbose=0)
        return loss

    def _update_exploration_rate(self, episode_count):
        self.epsilon = self.epsilon_policy.adjust_epsilon(
            self.epsilon, episode_count=episode_count, max_episodes=self.max_episodes
        )

    def select_action(self, current_state):
        """self an action base on epsilon value , higher epsilon -> more random actions"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_value = self.model.predict(current_state, verbose=0)[0]
            return np.argmax(q_value)

    def store_experience(self, state, next_state, reward, action, done, huristic):
        self.buffer_helper.store_experience(
            state, next_state, reward, action, done, huristic
        )

    def get_epsilon(self):
        return self.epsilon


class DoubleDQNAgent(DQNAgent):
    def __init__(
        self,
        action_size: int,
        state_size: int,
        learning_rate: float,
        buffer_helper: ExperienceBuffer,
        batch_size: int,
        gamma: float,
        max_episodes: int,
        epsilon: float,
        epsilon_policy: EpsilonPolicy,
        update_target_network_method: UpdateTargetNetworkType,
        update_factor: float,
        target_update_frequency: int,
        fc1_units: int,
        fc2_units: int,
    ) -> None:
        super().__init__(
            action_size=action_size,
            state_size=state_size,
            learning_rate=learning_rate,
            buffer_helper=buffer_helper,
            batch_size=batch_size,
            gamma=gamma,
            max_episodes=max_episodes,
            epsilon=epsilon,
            epsilon_policy=epsilon_policy,
            fc1_units=fc1_units,
            fc2_units=fc2_units,
        )
        self.update_target_network_method = update_target_network_method
        self.online_model: "AbstractQNetwork"
        self.target_model: "AbstractQNetwork"
        self.previous_episode = 0
        self.update_factor = update_factor
        self.target_update_frequency = target_update_frequency
        self._define_model(state_size, action_size, fc1_units, fc2_units, learning_rate)

    def _define_model(
        self, state_size, action_size, fc1_units, fc2_units, learning_rate
    ):
        self.online_model = QNetworkFactory.create_model(
            ModelType.STANDARD,
            state_size,
            action_size,
            fc1_units,
            fc2_units,
            learning_rate,
        )
        self.target_model = QNetworkFactory.create_model(
            ModelType.STANDARD,
            state_size,
            action_size,
            fc1_units,
            fc2_units,
            learning_rate,
        )
        self.target_model.set_weights(self.online_model.get_weights())

    def train(self, episode):
        """no commnet"""
        data = self.buffer_helper.sample_batch(self.batch_size)
        if data is None:
            return None
        states, next_states, rewards, actions, dones, heuristics = data

        # Here's what I understand is happening:
        # We have two models: one acts based on Temporal Difference (TD) learning, and the other (called the target network) behaves more like a Monte Carlo method.
        # Basically, one model is updated frequently (every epoch), while the other is updated less often (at a fixed interval).
        # The frequently updated model is used for action selection, while the target network is used to evaluate Q-values.
        # The TD update equation: Q(s, a) = r + γ * (r + max_a' Q(s', a') - Q(s, a)) becomes:
        # Q_target(s, a) = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
        q_current = self.online_model.predict(states, verbose=0)
        q_next_online = self.online_model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                best_action = np.argmax(q_next_online[i])
                q_targets[i, actions[i]] = (
                    rewards[i] + self.gamma * q_next_target[i, best_action]
                )
            else:
                q_targets[i, actions[i]] = rewards[i]
            # q_targets[i, actions[i]] = np.clip(q_targets[i, actions[i]], -10, 10)

        # and here is where target network update base on frequency
        if episode % self.target_update_frequency == 0:
            self._update_target_network()
        self._update_exploration_rate(episode_count=episode)
        loss = self.online_model.fit(states, q_targets, epochs=1, verbose=0)
        return loss

    def _update_target_network(self):
        if self.update_target_network_method == UpdateTargetNetworkType.HARD:
            self.target_model.set_weights(self.online_model.get_weights())
        elif self.update_target_network_method == UpdateTargetNetworkType.SOFT:
            online_weights = self.online_model.get_weights()
            target_weights = self.target_model.get_weights()
            updated_weights = []
            for online_w, target_w in zip(online_weights, target_weights):
                updated_weights.append(
                    self.update_factor * online_w + (1 - self.update_factor) * target_w
                )
            self.target_model.set_weights(updated_weights)

    def select_action(self, current_state):
        """self an action base on epsilon value , higher epsilon -> more random actions"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_value = self.online_model.predict(current_state, verbose=0)[0]
            return np.argmax(q_value)


class DuelingDQNAgent(DoubleDQNAgent):
    def __init__(
        self,
        action_size: int,
        state_size: int,
        learning_rate: float,
        buffer_helper: ExperienceBuffer,
        batch_size: int,
        gamma: float,
        max_episodes: int,
        epsilon: float,
        epsilon_policy: EpsilonPolicy,
        update_target_network_method: UpdateTargetNetworkType,
        update_factor: float,
        target_update_frequency: int,
        fc1_units: int,
        fc2_units: int,
    ) -> None:
        super().__init__(
            action_size=action_size,
            state_size=state_size,
            learning_rate=learning_rate,
            buffer_helper=buffer_helper,
            batch_size=batch_size,
            gamma=gamma,
            max_episodes=max_episodes,
            epsilon=epsilon,
            epsilon_policy=epsilon_policy,
            update_target_network_method=update_target_network_method,
            update_factor=update_factor,
            target_update_frequency=target_update_frequency,
            fc1_units=fc1_units,
            fc2_units=fc2_units,
        )

    def _define_model(
        self, state_size, action_size, fc1_units, fc2_units, learning_rate
    ):
        self.online_model = QNetworkFactory.create_model(
            ModelType.DUELING,
            state_size,
            action_size,
            fc1_units,
            fc2_units,
            learning_rate,
        )
        self.target_model = QNetworkFactory.create_model(
            ModelType.DUELING,
            state_size,
            action_size,
            fc1_units,
            fc2_units,
            learning_rate,
        )
        self.target_model.set_weights(self.online_model.get_weights())
