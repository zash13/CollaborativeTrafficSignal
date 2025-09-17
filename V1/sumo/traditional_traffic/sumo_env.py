import numpy as np
import traci
import os
import sys
from plexe import Plexe, ACC
from sumo.traditional_traffic.utils import communicate

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

VEHICLE_LENGTH = 4
DISTANCE = 4
LANE_NUM = 12
PLATOON_SIZE = 2
SPEED = 27.78
V2I_RANGE = 200
PLATOON_LENGTH = VEHICLE_LENGTH * PLATOON_SIZE + DISTANCE * (PLATOON_SIZE - 1)
ADD_PLATOON_PRO = 0.7
ADD_PLATOON_STEP = 60
MAX_ACCEL = 2.6
STOP_LINE = 15.0
SIM_STEP_SIZE = 5.0
VEHICLE_ID_COUNTER = 0
MIN_GREEN_STEPS = 3


def add_single_platoon(plexe, topology, step, lane):
    global VEHICLE_ID_COUNTER
    import random

    for i in range(PLATOON_SIZE):
        vid = f"v.{VEHICLE_ID_COUNTER}.{lane}.{i}"
        VEHICLE_ID_COUNTER += 1
        routeID = f"route_{lane}"
        try:
            traci.vehicle.add(
                vid,
                routeID,
                departPos=str(100 - i * (VEHICLE_LENGTH + DISTANCE)),
                departSpeed=str(20),
                departLane=str(lane % 3),
                typeID="vtypeauto",
            )
        except traci.exceptions.TraCIException as e:
            print(f"Warning: Could not add vehicle {vid}: {e}")
            continue
        plexe.set_path_cacc_parameters(vid, DISTANCE, 2, 1, 0.5)
        plexe.set_cc_desired_speed(vid, SPEED)
        plexe.set_acc_headway_time(vid, 1.0)
        plexe.use_controller_acceleration(vid, False)
        plexe.set_fixed_lane(vid, lane % 3, False)
        traci.vehicle.setSpeedMode(vid, 31)
        plexe.set_active_controller(vid, ACC)
        traci.vehicle.setColor(vid, (255, 255, 255, 255))
        topology[vid] = {}


def add_platoons(plexe, topology, step):
    import random

    for lane in range(LANE_NUM):
        if random.random() < ADD_PLATOON_PRO:
            add_single_platoon(plexe, topology, step, lane)


class SumoEnvironment:
    """
    Updated SumoEnvironment:
      - sim_step_size: small physics timestep (e.g. 0.1 - 0.5s)
      - control_interval: how often the agent acts (seconds), e.g. 5.0
      - agent acts every control_steps = control_interval / sim_step_size
      - rewards aggregated over the control interval
    """

    def __init__(
        self,
        sumo_config="traditional_traffic.sumo.cfg",
        gui=False,
        sim_step_size=0.2,
        control_interval=5.0,
        add_platoon_interval=60.0,
    ):
        base_path = os.path.dirname(__file__)
        self.sumo_config = os.path.join(base_path, sumo_config)
        self.gui = gui
        self.junction_id = "junction"
        self.phases = [
            "GGrGrrGGrGrr",
            "gyrgrrgyrgrr",
            "grGgrrgrGgrr",
            "grygrrgrygrr",
            "GrrGGrGrrGGr",
            "grrgyrgrrgyr",
            "grrgrGgrrgrG",
            "grrgrygrrgry",
        ]
        self.lane_ids = [
            "end1_junction_0",
            "end1_junction_1",
            "end1_junction_2",
            "end2_junction_0",
            "end2_junction_1",
            "end2_junction_2",
            "end3_junction_0",
            "end3_junction_1",
            "end3_junction_2",
            "end4_junction_0",
            "end4_junction_1",
            "end4_junction_2",
        ]

        # timing controls
        self.sim_step_size = float(sim_step_size)  # e.g. 0.1 or 0.2
        self.control_interval = float(control_interval)  # e.g. 5.0 seconds per decision
        self.control_steps = max(
            1, int(round(self.control_interval / self.sim_step_size))
        )
        self.add_platoon_interval_sec = float(add_platoon_interval)
        self.add_platoon_interval_steps = max(
            1, int(round(self.add_platoon_interval_sec / self.sim_step_size))
        )

        self.max_steps = 720  # agent decision steps per episode (unchanged)
        self.agent_step_count = 0
        self.sim_step_count = 0  # raw SUMO steps
        self.plexe = None
        self.topology = {}
        self.departed_vehicles = 0
        self.prev_vehicle_count = 0
        self.observation_shape = (12 * 3 + 8,)
        self.action_count = len(self.phases)
        self.last_action = 0
        self.prev_phase = 0
        self.green_steps_remaining = 0

        # runtime accumulators for episode reporting
        self.cum_waiting_time = 0.0
        self.cum_queue_length = 0.0
        self.cum_throughput = 0
        self.cum_stopped = 0

    def initialize_simulation(self):
        global VEHICLE_ID_COUNTER
        VEHICLE_ID_COUNTER = 0
        if traci.isLoaded():
            traci.close()
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "--duration-log.statistics",
            "--tripinfo-output",
            "output_file.xml",
            "-c",
            self.sumo_config,
            "--step-length",
            str(self.sim_step_size),
            "--no-step-log",
            "--no-warnings",
        ]
        try:
            traci.start(sumo_cmd)
            print("Available traffic light IDs:", traci.trafficlight.getIDList())
        except traci.exceptions.TraCIException as e:
            print(f"Failed to start SUMO: {e}")
            sys.exit(1)

        self.plexe = Plexe()
        traci.addStepListener(self.plexe)
        self.agent_step_count = 0
        self.sim_step_count = 0
        self.topology = {}
        self.departed_vehicles = 0
        self.prev_vehicle_count = 0
        self.green_steps_remaining = 0
        self.cum_waiting_time = 0.0
        self.cum_queue_length = 0.0
        self.cum_throughput = 0
        self.cum_stopped = 0

        try:
            traci.trafficlight.setRedYellowGreenState(self.junction_id, self.phases[0])
        except traci.exceptions.TraCIException as e:
            print(f"Error setting traffic light state: {e}")
            traci.close()
            sys.exit(1)

        return self.get_observation()

    def take_action(self, action):
        """Agent calls this once per decision interval.
        Internally we step SUMO `control_steps` times and aggregate rewards/metrics.
        Returns: observation, reward, terminated, truncated, info
        """
        if not 0 <= action < self.action_count:
            raise ValueError(
                f"Action {action} is invalid. Must be in [0, {self.action_count - 1}]."
            )

        # Enforce minimum green logic as you had before (per agent decision)
        phase_change = self.last_action != action
        if phase_change and self.green_steps_remaining > 0:
            action = self.last_action
        else:
            self.green_steps_remaining = (
                MIN_GREEN_STEPS if action in [0, 2, 4, 6] else 0
            )
        self.last_action = action

        # apply phase immediately
        try:
            traci.trafficlight.setRedYellowGreenState(
                self.junction_id, self.phases[action]
            )
            self.prev_phase = action
        except traci.exceptions.TraCIException as e:
            print(f"Error setting traffic light phase: {e}")
            traci.close()
            sys.exit(1)

        # run many small SUMO steps and aggregate metrics
        agg_reward = 0.0
        agg_throughput = 0
        agg_waiting = 0.0
        agg_queue = 0.0
        agg_stopped = 0

        for inner in range(self.control_steps):
            # single SUMO physics step
            traci.simulationStep()
            self.sim_step_count += 1

            # spawn platoons occasionally (based on sim-step intervals now)
            if self.sim_step_count % self.add_platoon_interval_steps == 0:
                add_platoons(self.plexe, self.topology, self.sim_step_count)

            # communicate occasionally (preserve your original timing)
            if self.sim_step_count % 10 == 1:
                try:
                    communicate(self.plexe, self.topology)
                except Exception:
                    pass

            # instantaneous metrics this sim-step
            vehicle_ids = traci.vehicle.getIDList()
            # sum waiting times across active vehicles (instant)
            waiting_time_sum = 0.0
            for vid in vehicle_ids:
                try:
                    waiting_time_sum += traci.vehicle.getWaitingTime(vid)
                except Exception:
                    pass
            queue_len = sum(
                traci.lane.getLastStepVehicleNumber(l) for l in self.lane_ids
            )
            arrived = (
                traci.simulation.getArrivedIDList()
            )  # list of vehicles that arrived this sim-step
            arrived_count = len(arrived)
            stopped_veh = sum(
                1
                for lane in self.lane_ids
                for v in traci.lane.getLastStepVehicleIDs(lane)
                if traci.vehicle.getSpeed(v) < 0.1
            )

            # compute per-sim-step reward (weights chosen to produce moderate magnitude)
            # NOTE: tune these weights to your objectives
            step_reward = (
                -0.01 * waiting_time_sum
                - 0.1 * queue_len
                - 0.05 * stopped_veh
                + 1.0 * arrived_count
            )
            # accumulate
            agg_reward += step_reward
            agg_throughput += arrived_count
            agg_waiting += waiting_time_sum
            agg_queue += queue_len
            agg_stopped += stopped_veh

            # decrease enforced green remaining after each sim-step when appropriate
            self.green_steps_remaining = max(0, self.green_steps_remaining - 1)

        # after the control interval, increment agent step
        self.agent_step_count += 1

        # update cumulative episode stats (for final printing)
        self.cum_waiting_time += agg_waiting
        self.cum_queue_length += agg_queue
        self.cum_throughput += agg_throughput
        self.cum_stopped += agg_stopped

        # build observation at the end of interval
        observation = self.get_observation()

        # termination / truncation
        terminated = self.agent_step_count >= self.max_steps
        truncated = False

        # info dictionary: aggregated numbers for this decision interval
        info = {
            "total_waiting_time": float(agg_waiting),
            "total_queue_length": int(agg_queue),
            "throughput": int(agg_throughput),
            "stopped_vehicles": int(agg_stopped),
            "phase_change_penalty": -0.5 if phase_change else 0.0,
        }

        return observation, float(agg_reward), terminated, truncated, info

    def get_observation(self):
        # same as before: queue_lengths, avg_speeds, waiting_times + phase one-hot
        queue_lengths = []
        avg_speeds = []
        waiting_times = []
        for lane in self.lane_ids:
            q = traci.lane.getLastStepVehicleNumber(lane)
            queue_lengths.append(q)
            speeds = [
                traci.vehicle.getSpeed(v)
                for v in traci.lane.getLastStepVehicleIDs(lane)
            ]
            avg_speed = np.mean(speeds) if speeds else 0.0
            avg_speeds.append(avg_speed)
            waiting_time = sum(
                traci.vehicle.getWaitingTime(v)
                for v in traci.lane.getLastStepVehicleIDs(lane)
            )
            waiting_times.append(waiting_time / max(1, q))
        # normalize same as your code but keep meaningful ranges
        queue_lengths = [min(q, 100) / 100.0 for q in queue_lengths]
        avg_speeds = [min(s, SPEED) / SPEED for s in avg_speeds]
        waiting_times = [min(w, 100) / 100.0 for w in waiting_times]
        current_phase = traci.trafficlight.getPhase(self.junction_id)
        phase_one_hot = [
            1 if i == current_phase else 0 for i in range(len(self.phases))
        ]
        observation = np.array(
            queue_lengths + avg_speeds + waiting_times + phase_one_hot, dtype=np.float32
        )

        if np.any(np.isnan(observation)):
            observation = np.nan_to_num(observation, nan=0.0)

        return observation

    def get_reward(self):
        # Kept for compatibility but your agent will get reward from take_action aggregated over control interval.
        # This function is left simple (not used directly in the new flow).
        return 0.0

    def get_observation_shape(self):
        return self.observation_shape

    def get_action_count(self):
        return self.action_count

    def close(self):
        if traci.isLoaded():
            traci.close()
