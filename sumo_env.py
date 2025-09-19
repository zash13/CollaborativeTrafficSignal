"""
SUMO environment compatible with train.py (DQN).
- Provides reset() returning a numpy observation vector.
- step(action) -> (next_obs: np.ndarray, reward: float, done: bool, info: dict)
- Uses sumolib for topology and traci for runtime.
- Includes detailed debug logging for per-lane vehicle counts and waiting times.
"""

import os
import random
import sys
import time
import numpy as np

try:
    import traci
except Exception as e:
    raise RuntimeError(
        "traci import failed. Ensure SUMO is installed and SUMO/tools is on PYTHONPATH."
    ) from e

try:
    import sumolib
except Exception as e:
    raise RuntimeError(
        "sumolib import failed. Install with `pip install sumolib` or ensure SUMO tools are available."
    ) from e

from config import Config

SHOW_LOG = False


class VehicleSpawner:
    """Handles spawning of vehicles along valid routes."""

    def __init__(self, config):
        self.net = sumolib.net.readNet(config.NET_FILE)
        self.last_spawn_time = -9999.0
        self.active_vehicles = 0
        self.valid_routes = self._discover_valid_routes()

    def _discover_valid_routes(self):
        valid_routes = []
        tls_ids = traci.trafficlight.getIDList()  # Removed [:24] slicing
        for tls_id in tls_ids:
            node = self.net.getNode(tls_id)
            if node is None:
                continue
            incoming_edges = [e.getID() for e in node.getIncoming()]
            outgoing_edges = [e.getID() for e in node.getOutgoing()]
            for conn in node.getConnections():
                src = conn.getFrom().getID()
                dst = conn.getTo().getID()
                if src in incoming_edges and dst in outgoing_edges:
                    valid_routes.append((src, dst))
        print(f"[INFO] Discovered {len(valid_routes)} valid routes.")
        return valid_routes

    def maybe_spawn_vehicle(self, sim_time):
        """Spawn a vehicle if allowed by interval and max vehicles."""
        if (sim_time - self.last_spawn_time) < random.uniform(
            Config.SPAWN_MIN_INTERVAL, Config.SPAWN_MAX_INTERVAL
        ):
            return False
        if self.active_vehicles >= Config.MAX_VEHICLES:
            return False

        if not self.valid_routes:
            print("[WARN] No valid routes available.")
            return False

        src, dst = random.choice(self.valid_routes)
        route_id = f"r_{int(sim_time * 1000)}_{random.randint(0, 9999)}"
        veh_id = f"v_{int(sim_time * 1000)}_{random.randint(0, 9999)}"
        try:
            traci.route.add(route_id, [src, dst])
            traci.vehicle.add(veh_id, routeID=route_id, typeID="DEFAULT_VEHTYPE")
            for lane in range(traci.edge.getLaneNumber(src)):
                traci.lane.setMaxSpeed(f"{src}_{lane}", Config.DEFAULT_SPEED)
            self.last_spawn_time = sim_time
            self.active_vehicles += 1
            if SHOW_LOG:
                print(f"[DEBUG] Spawned vehicle {veh_id} on route {src} -> {dst}")
            return True
        except traci.exceptions.TraCIException as e:
            print(f"[WARN] Failed to spawn vehicle: {e}")
            return False

    def on_vehicle_removed(self):
        """Decrement active vehicle count when a vehicle is removed."""
        self.active_vehicles = max(0, self.active_vehicles - 1)


class SumoEnv:
    """SUMO environment for DQN training with per-lane debugging."""

    def __init__(self, config):
        self.config = config
        sumo_cmd = [
            self.config.SUMO_BINARY,
            "-c",
            self.config.SUMOCFG_FILE,
            "--no-warnings",
            "--step-length",
            str(self.config.SIM_STEP),
        ]
        try:
            traci.start(sumo_cmd)
        except traci.exceptions.TraCIException as e:
            raise RuntimeError(f"Failed to start SUMO: {e}")
        self.tls_ids = traci.trafficlight.getIDList()  # Removed [:24] slicing
        self.phases = {
            tls_id: [
                p.state
                for p in traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[
                    0
                ].phases
            ]
            for tls_id in self.tls_ids
        }
        self.lane_map = self._build_lane_map()
        self.obs_dim = (
            sum(len(lanes) for lanes in self.lane_map.values()) * 3
        )  # 3 features per lane
        print(
            f"[INFO] Loaded {len(self.tls_ids)} traffic lights, {self.obs_dim} observation dimensions based on {sum(len(lanes) for lanes in self.lane_map.values())} unique lanes."
        )
        self.spawner = VehicleSpawner(self.config)
        self.sim_time = 0.0
        self.step_count = 0
        self.closed = False
        self.last_total_waiting = 0.0  # Initialize for reward calculation

    def _build_lane_map(self):
        """Build a map of traffic lights to their controlled lanes."""
        lane_map = {}
        for tls_id in self.tls_ids:
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            lanes = set()
            for link in controlled_links:
                if link and link[0]:  # Ensure link and incoming lane exist
                    lanes.add(link[0][0])  # Add incoming lane ID
            if lanes:
                lane_map[tls_id] = list(lanes)
        if SHOW_LOG:
            print(f"[DEBUG] Lane map: {lane_map}")
        return lane_map

    def reset(self):
        """Reset the environment and return the initial observation."""
        if self.closed:
            raise RuntimeError("Environment is closed; cannot reset.")
        try:
            traci.close()
        except Exception:
            pass
        sumo_cmd = [
            self.config.SUMO_BINARY,
            "-c",
            self.config.SUMOCFG_FILE,
            "--no-warnings",
            "--step-length",
            str(self.config.SIM_STEP),
        ]
        try:
            traci.start(sumo_cmd)
        except traci.exceptions.TraCIException as e:
            raise RuntimeError(f"Failed to start SUMO on reset: {e}")
        self.tls_ids = traci.trafficlight.getIDList()  # Removed [:24] slicing
        self.phases = {
            tls_id: [
                p.state
                for p in traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[
                    0
                ].phases
            ]
            for tls_id in self.tls_ids
        }
        self.lane_map = self._build_lane_map()
        self.obs_dim = sum(len(lanes) for lanes in self.lane_map.values()) * 3
        self.spawner = VehicleSpawner(self.config)
        self.sim_time = 0.0
        self.step_count = 0
        self.last_total_waiting = 0.0  # Reset waiting for new episode
        return self._build_obs()

    def _build_obs(self):
        """Build observation vector with per-lane data and debug output."""
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        try:
            idx = 0
            for tls_id, lanes in self.lane_map.items():
                current_phase = traci.trafficlight.getRedYellowGreenState(tls_id)
                if SHOW_LOG:
                    print(f"[DEBUG] TLS {tls_id} Phase State: {current_phase}")
                for lane in lanes:
                    vehicles = traci.lane.getLastStepVehicleNumber(lane)
                    waiting = traci.lane.getWaitingTime(lane)
                    phase_idx = traci.trafficlight.getPhase(tls_id)
                    normalized_vehicles = min(
                        vehicles / self.config.MAX_VEHICLES_PER_LANE, 1.0
                    )
                    normalized_waiting = min(
                        waiting / self.config.MAX_WAIT_EXPECTED, 1.0
                    )
                    normalized_phase = phase_idx / len(self.phases.get(tls_id, [1]))
                    vec[idx] = normalized_vehicles
                    vec[idx + 1] = normalized_waiting
                    vec[idx + 2] = normalized_phase
                    if SHOW_LOG:
                        print(
                            f"[DEBUG] TLS {tls_id}, Lane {lane}: "
                            f"Vehicles={vehicles} (norm={normalized_vehicles:.3f}), "
                            f"Waiting={waiting:.1f}s (norm={normalized_waiting:.3f}), "
                            f"Phase={phase_idx} (norm={normalized_phase:.3f})"
                        )
                    idx += 3
        except Exception as e:
            print(f"[WARN] Observation error: {e}")
        return vec

    def step(self, action=None):
        if self.closed:
            raise RuntimeError("Environment is closed; cannot step.")

        if action is not None:
            for tls_id in self.tls_ids:
                if 0 <= action < len(self.phases.get(tls_id, [])):
                    try:
                        traci.trafficlight.setRedYellowGreenState(
                            tls_id, self.phases[tls_id][action]
                        )
                    except traci.exceptions.TraCIException as e:
                        print(f"[WARN] Failed to set TLS state for {tls_id}: {e}")

        self.spawner.maybe_spawn_vehicle(self.sim_time)
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError as e:
            print(f"[ERROR] SUMO connection closed: {e}")
            self.closed = True
            return (
                np.zeros(self.obs_dim, dtype=np.float32),
                0.0,
                True,
                {"error": str(e)},
            )

        self.sim_time += self.config.SIM_STEP
        self.step_count += 1

        # Remove finished vehicles
        for vid in list(traci.vehicle.getIDList()):
            if traci.vehicle.getRoadID(vid) == "":
                self.spawner.on_vehicle_removed()

        obs = self._build_obs()
        done = self.step_count >= self.config.MAX_STEPS
        current_waiting = sum(
            traci.lane.getWaitingTime(lane)
            for lanes in self.lane_map.values()
            for lane in lanes
        )

        # Reward is improvement in waiting
        reward = self.last_total_waiting - current_waiting

        # Alternative: reward vehicles that left the network
        reward += traci.simulation.getArrivedNumber()

        self.last_total_waiting = current_waiting
        clipped_reward = np.clip(reward, -10.0, 10.0)
        if SHOW_LOG:
            print(
                f"[DEBUG] Reward components per lane: {[-float(obs[i]) for i in range(0, len(obs), 3)] + [-float(obs[i + 1]) for i in range(1, len(obs), 3)]}, "
                f"Waiting Diff={reward:.3f}, Clipped={clipped_reward:.3f}"
            )
        info = {"time": self.sim_time, "active_vehicles": self.spawner.active_vehicles}
        return obs, clipped_reward, done, info

    def close(self):
        if self.closed:
            return
        try:
            traci.close()
        except Exception as e:
            print(f"[WARN] closing traci failed: {e}")
        self.closed = True


# Quick test main
if __name__ == "__main__":
    if not os.path.exists(Config.NET_FILE):
        print(f"[ERROR] Net file not found: {Config.NET_FILE}")
        sys.exit(1)
    try:
        env = SumoEnv(Config)
        obs = env.reset()
        print("Initial obs vector:", obs)
        for i in range(10):
            action = random.randint(
                0, max(0, len(env.phases.get(env.tls_ids[0], [])) - 1)
            )
            obs, reward, done, info = env.step(action)
            print(
                f"step {i} action={action} reward={reward:.3f} active={info.get('active_vehicles')}"
            )
            if done:
                break
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
