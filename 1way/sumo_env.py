#!/usr/bin/env python3
"""
SUMO environment compatible with train.py (DQN) for 1-way network.
- Provides reset() returning a numpy observation vector.
- step(action) -> (next_obs: np.ndarray, reward: float, done: bool, info: dict)
- Uses sumolib for topology and traci for runtime.
- Configured for GUI visualization and runs for at least 1 minute with random actions.
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


class Config:
    # SUMO binary set to GUI for visualization
    SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo-gui")
    NET_FILE = os.path.join(os.getcwd(), "1way.net.xml")
    ROUTE_FILE = os.path.join(os.getcwd(), "1way.rou.xml")
    SUMOCFG_FILE = os.path.join(os.getcwd(), "1way.sumocfg")
    SIM_STEP = 1.0
    MAX_STEPS = 3600
    MAX_VEHICLES = 200
    SPAWN_MIN_INTERVAL = 0.2
    SPAWN_MAX_INTERVAL = 2.0
    DEFAULT_SPEED = 13.89  # m/s (≈50 km/h)
    # OBSERVATION vector dimension: vehicle count + waiting + phase one-hot (3 phases)
    OBS_DIM = 6


def ensure_minimal_sumocfg():
    """Create minimal route and sumocfg files if missing."""
    if not os.path.exists(Config.ROUTE_FILE):
        with open(Config.ROUTE_FILE, "w") as f:
            f.write('<?xml version="1.0"?>\n<routes>\n</routes>\n')
        print(f"[INFO] Created empty route file: {Config.ROUTE_FILE}")

    if not os.path.exists(Config.SUMOCFG_FILE):
        cfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{Config.NET_FILE}"/>
        <route-files value="{Config.ROUTE_FILE}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{Config.MAX_STEPS}"/>
        <step-length value="{Config.SIM_STEP}"/>
    </time>
</configuration>
"""
        with open(Config.SUMOCFG_FILE, "w") as f:
            f.write(cfg)
        print(f"[INFO] Created minimal sumocfg: {Config.SUMOCFG_FILE}")


class VehicleSpawner:
    def __init__(self, net_path, center_tls_id=None):
        self.net = sumolib.net.readNet(net_path)
        self.center_tls_id = center_tls_id
        self.tls_node = None
        self.last_spawn_time = -9999.0
        self.active_vehicles = 0
        self.incoming_edges = []  # Will be ["E0"]
        self.outgoing_edges = []  # Will be ["E1"]
        self.valid_routes = []

    def _discover_center_junction(self):
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("No traffic lights found in running SUMO instance.")
        selected_tls = (
            self.center_tls_id
            if (self.center_tls_id and self.center_tls_id in tls_ids)
            else tls_ids[0]
        )
        try:
            node = self.net.getNode(selected_tls)
        except KeyError:
            raise RuntimeError(f"TLS id '{selected_tls}' not found in net file.")
        self.tls_node = node
        self.incoming_edges = [e.getID() for e in node.getIncoming()]
        self.outgoing_edges = [e.getID() for e in node.getOutgoing()]
        if not self.incoming_edges:
            raise RuntimeError(f"No incoming edges found for TLS '{selected_tls}'.")
        # Build valid routes for 1-way: E0 to E1
        self.valid_routes = []
        for conn in node.getConnections():
            src = conn.getFrom().getID()
            dst = conn.getTo().getID()
            if src in self.incoming_edges and dst in self.outgoing_edges:
                self.valid_routes.append((src, dst))
        print(f"[INFO] Using TLS '{selected_tls}' mapped to node '{node.getID()}'.")
        print(f"[INFO] Incoming edges: {self.incoming_edges}")
        print(f"[INFO] Outgoing edges: {self.outgoing_edges}")
        print(f"[INFO] Valid routes: {self.valid_routes}")

    def maybe_spawn_vehicle(self, sim_time):
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
            traci.vehicle.add(
                veh_id, routeID=route_id, typeID="DEFAULT_VEHTYPE"
            )  # Fixed API
            for lane in range(traci.edge.getLaneNumber(src)):
                traci.lane.setMaxSpeed(f"{src}_{lane}", Config.DEFAULT_SPEED)
            self.last_spawn_time = sim_time
            self.active_vehicles += 1
            return True
        except traci.exceptions.TraCIException as e:
            print(f"[WARN] Failed to spawn vehicle: {e}")
            return False

    def on_vehicle_removed(self):
        self.active_vehicles = max(0, self.active_vehicles - 1)


class SumoEnv:
    def __init__(self):
        ensure_minimal_sumocfg()
        sumo_cmd = [
            Config.SUMO_BINARY,
            "-c",
            Config.SUMOCFG_FILE,
            "--no-warnings",
            "--step-length",
            str(Config.SIM_STEP),
            "--start",  # Auto-start the GUI
        ]
        traci.start(sumo_cmd)
        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("No traffic lights after SUMO start.")
        self.tls_id = tls_ids[0]
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        self.phases = [p.state for p in logic.phases]
        print(f"[INFO] TLS '{self.tls_id}' phases loaded: {len(self.phases)} phases.")
        self.spawner = VehicleSpawner(Config.NET_FILE, center_tls_id=self.tls_id)
        try:
            self.spawner._discover_center_junction()
        except Exception as e:
            raise RuntimeError(f"Failed to discover junction: {e}")
        self.sim_time = 0.0
        self.step_count = 0
        self.closed = False
        self.obs_dim = Config.OBS_DIM

    def reset(self):
        if self.closed:
            raise RuntimeError("Environment is closed; cannot reset.")
        try:
            traci.close()
        except Exception:
            pass
        sumo_cmd = [
            Config.SUMO_BINARY,
            "-c",
            Config.SUMOCFG_FILE,
            "--no-warnings",
            "--step-length",
            str(Config.SIM_STEP),
            "--start",
        ]
        traci.start(sumo_cmd)
        tls_ids = traci.trafficlight.getIDList()
        self.tls_id = tls_ids[0]
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        self.phases = [p.state for p in logic.phases]
        self.spawner = VehicleSpawner(Config.NET_FILE, center_tls_id=self.tls_id)
        try:
            self.spawner._discover_center_junction()
        except Exception as e:
            raise RuntimeError(f"Failed to discover junction: {e}")
        self.sim_time = 0.0
        self.step_count = 0
        return self._build_obs()

    def _build_obs(self):
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        try:
            src = (
                self.spawner.incoming_edges[0] if self.spawner.incoming_edges else "E0"
            )
            vehicles = traci.edge.getLastStepVehicleNumber(src)
            waiting = traci.edge.getWaitingTime(src) / 60.0  # Normalized to minutes
            phase_idx = traci.trafficlight.getPhase(self.tls_id)
            phase_onehot = np.eye(len(self.phases))[phase_idx]
            vec[0] = vehicles
            vec[1] = waiting
            vec[2 : 2 + len(phase_onehot)] = phase_onehot
        except Exception as e:
            print(f"[WARN] Observation error: {e}")
        return vec

    def step(self, action=None):
        if self.closed:
            raise RuntimeError("Environment is closed; cannot step.")

        if action is not None:
            if not (0 <= action < len(self.phases)):
                raise ValueError("Action out of range.")
            try:
                traci.trafficlight.setRedYellowGreenState(
                    self.tls_id, self.phases[action]
                )
            except traci.exceptions.TraCIException as e:
                print(f"[WARN] Failed to set TLS state: {e}")

        try:
            spawned = self.spawner.maybe_spawn_vehicle(self.sim_time)
            traci.simulationStep()
            if spawned:
                print(f"[INFO] Spawned vehicle at time {self.sim_time:.1f}s")
        except traci.exceptions.FatalTraCIError as e:
            print(f"[ERROR] SUMO connection closed during simulationStep: {e}")
            self.closed = True
            return (
                np.zeros(self.obs_dim, dtype=np.float32),
                0.0,
                True,
                {"error": str(e)},
            )
        except traci.exceptions.TraCIException as e:
            print(f"[WARN] TraCI exception during step: {e}")

        self.sim_time += Config.SIM_STEP
        self.step_count += 1

        try:
            for vid in list(traci.vehicle.getIDList()):
                if traci.vehicle.getRoadID(vid) == "":
                    self.spawner.on_vehicle_removed()
        except Exception as e:
            print(f"[WARN] Vehicle removal error: {e}")

        obs = self._build_obs()
        done = self.step_count >= Config.MAX_STEPS
        vehicles = obs[0]
        waiting = obs[1]
        reward = -(vehicles + waiting)
        print(f"v :{vehicles} w:{waiting} r: {reward}")
        info = {"time": self.sim_time, "active_vehicles": self.spawner.active_vehicles}
        return obs, reward, done, info

    def close(self):
        if self.closed:
            return
        try:
            traci.close()
        except Exception as e:
            print(f"[WARN] closing traci failed: {e}")
        self.closed = True


# Manual run with random actions for visualization, running for at least 1 minute
if __name__ == "__main__":
    if not os.path.exists(Config.NET_FILE):
        print(f"[ERROR] Net file not found: {Config.NET_FILE}")
        sys.exit(1)

    env = SumoEnv()
    obs = env.reset()
    print("Initial obs vector:", obs)
    try:
        target_steps = 60  # 1 minute with SIM_STEP=1.0
        for i in range(target_steps):
            action = random.randint(0, len(env.phases) - 1)  # Random phase selection
            obs, reward, done, info = env.step(action)
            print(
                f"Step {i} | Action={action} | Reward={reward:.2f} | Vehicles={info['active_vehicles']} | Time={info['time']:.1f}s"
            )
            if done:
                print("Simulation done due to max steps.")
                break
            time.sleep(1.0)  # 1-second delay for 1-minute total runtime
        print(f"Simulation ran for {target_steps} steps (~{target_steps}s).")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nSimulation error: {e}")
    finally:
        env.close()
