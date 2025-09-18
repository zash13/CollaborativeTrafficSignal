#!/usr/bin/env python3
"""
SUMO environment compatible with train.py (DQN).
- Provides reset() returning a numpy observation vector.
- step(action) -> (next_obs: np.ndarray, reward: float, done: bool, info: dict)
- Uses sumolib for topology and traci for runtime.
"""
# This code was completely written using GPT

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
    # SUMO binary (use env SUMO_BINARY to override)
    SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")
    NET_FILE = os.path.join(os.getcwd(), "my_4way.net.xml")
    ROUTE_FILE = os.path.join(os.getcwd(), "my_4way.rou.xml")
    SUMOCFG_FILE = os.path.join(os.getcwd(), "my_4way.sumocfg")
    SIM_STEP = 1.0
    MAX_STEPS = 3600
    MAX_VEHICLES = 200
    SPAWN_MIN_INTERVAL = 0.2
    SPAWN_MAX_INTERVAL = 2.0
    DEFAULT_SPEED = 13.89  # m/s (≈50 km/h)
    # OBSERVATION vector dimension expected by your agent (train.py uses 20)
    OBS_DIM = 20


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
        self.incoming_edges = []
        self.outgoing_edges = []
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
        # Build valid routes: edge->edge connections through the junction
        self.valid_routes = []
        for conn in node.getConnections():
            try:
                src = conn.getFrom().getID()
                dst = conn.getTo().getID()
            except Exception:
                continue
            if src in self.incoming_edges and dst in self.outgoing_edges:
                self.valid_routes.append((src, dst))

    def maybe_spawn_vehicle(self, sim_time):
        # Ensure discovered; attempt discovery lazily if needed
        if not self.tls_node:
            try:
                self._discover_center_junction()
            except Exception:
                return False

        if (sim_time - self.last_spawn_time) < random.uniform(
            Config.SPAWN_MIN_INTERVAL, Config.SPAWN_MAX_INTERVAL
        ):
            return False
        if self.active_vehicles >= Config.MAX_VEHICLES:
            return False
        if not self.valid_routes:
            # nothing valid to spawn
            return False

        src, dst = random.choice(self.valid_routes)
        route_id = f"r_{int(sim_time * 1000)}_{random.randint(0, 9999)}"
        veh_id = f"v_{int(sim_time * 1000)}_{random.randint(0, 9999)}"
        try:
            traci.route.add(route_id, [src, dst])
            traci.vehicle.add(veh_id, routeID=route_id, typeID="DEFAULT_VEHTYPE")
        except traci.exceptions.TraCIException as e:
            print(f"[WARN] Failed to add vehicle/route: {e}")
            return False

        # set lane speed best-effort
        try:
            lane_count = traci.edge.getLaneNumber(src)
            for li in range(lane_count):
                traci.lane.setMaxSpeed(f"{src}_{li}", Config.DEFAULT_SPEED)
        except Exception:
            pass

        self.last_spawn_time = sim_time
        self.active_vehicles += 1
        return True

    def on_vehicle_removed(self):
        self.active_vehicles = max(0, self.active_vehicles - 1)


class SumoEnv:
    def __init__(self, sumo_binary=None, net_file=None, sumocfg=None, obs_dim=None):
        self.sumocfg = sumocfg or Config.SUMOCFG_FILE
        self.net_file = net_file or Config.NET_FILE
        self.sumo_binary = sumo_binary or Config.SUMO_BINARY
        self.closed = False
        self.obs_dim = obs_dim or Config.OBS_DIM

        ensure_minimal_sumocfg()
        self._start_sumo_and_init()

    def _start_sumo_and_init(self):
        cmd = [
            self.sumo_binary,
            "-c",
            self.sumocfg,
            "--step-length",
            str(Config.SIM_STEP),
            "--start",
            "--no-warnings",
        ]
        print("[INFO] Starting SUMO with command:", " ".join(cmd))
        try:
            traci.start(cmd)
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO: {e}") from e

        tls_ids = traci.trafficlight.getIDList()
        if not tls_ids:
            raise RuntimeError("No traffic lights after SUMO start.")
        self.tls_id = tls_ids[0]
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        self.phases = [p.state for p in logic.phases]
        print(f"[INFO] TLS '{self.tls_id}' phases loaded: {len(self.phases)} phases.")

        self.spawner = VehicleSpawner(self.net_file, center_tls_id=self.tls_id)
        # discover mapping now (so valid_routes available)
        try:
            self.spawner._discover_center_junction()
        except Exception as e:
            # warn but continue
            print(f"[WARN] Spawner discovery warning: {e}")

        self.sim_time = 0.0
        self.step_count = 0

    def reset(self):
        """Reset the SUMO simulation and return initial observation vector (numpy)."""
        # close existing SUMO instance if open
        try:
            traci.close()
        except Exception:
            pass

        # restart SUMO cleanly
        self.closed = False
        self._start_sumo_and_init()

        # return initial observation
        obs = self._build_obs()
        return obs

    def _build_obs(self):
        """
        Build an OBS_DIM vector:
          - first K = len(incoming_edges) entries: vehicle counts on incoming edges
          - next entries: zeros (padding) up to OBS_DIM
        """
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        try:
            incoming = self.spawner.incoming_edges
            for i, e in enumerate(incoming):
                if i >= self.obs_dim:
                    break
                vec[i] = float(traci.edge.getLastStepVehicleNumber(e))
        except Exception:
            # if traci fails, keep zeros
            pass
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
            # try to spawn and step
            self.spawner.maybe_spawn_vehicle(self.sim_time)
            traci.simulationStep()
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

        # update removed vehicles
        try:
            for vid in list(traci.vehicle.getIDList()):
                if traci.vehicle.getRoadID(vid) == "":
                    self.spawner.on_vehicle_removed()
        except Exception:
            pass

        obs = self._build_obs()
        done = self.step_count >= Config.MAX_STEPS
        reward = -float(np.sum(obs))  # simple negative total vehicles on incoming edges
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


# Quick manual run for sanity (kept minimal)
if __name__ == "__main__":
    if not os.path.exists(Config.NET_FILE):
        print(f"[ERROR] Net file not found: {Config.NET_FILE}")
        sys.exit(1)

    env = SumoEnv()
    obs = env.reset()
    print("Initial obs vector:", obs)
    for i in range(10):
        action = random.randint(0, max(0, len(env.phases) - 1))
        obs, reward, done, info = env.step(action)
        print(
            f"step {i} action={action} reward={reward} active={info.get('active_vehicles')}"
        )
        if done:
            break
    env.close()
