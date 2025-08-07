import numpy as np
import traci
import os
import sys
from plexe import Plexe, ACC, CACC
from sumo.traditional_traffic.utils import communicate

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

VEHICLE_LENGTH = 4
DISTANCE = 4
LANE_NUM = 12
PLATOON_SIZE = 1
SPEED = 27.78  # 100 km/h
V2I_RANGE = 200
PLATOON_LENGTH = VEHICLE_LENGTH * PLATOON_SIZE + DISTANCE * (PLATOON_SIZE - 1)
ADD_PLATOON_PRO = 0.3
ADD_PLATOON_STEP = 120  # Adjusted for 5s steps (600s / 5 = 120 steps)
MAX_ACCEL = 2.6
STOP_LINE = 15.0
SIM_STEP_SIZE = 5.0  # Each simulation step advances 5 seconds


def add_single_platoon(plexe, topology, step, lane):
    import random

    for i in range(PLATOON_SIZE):
        vid = "v.%d.%d.%d" % (step // ADD_PLATOON_STEP, lane, i)
        routeID = "route_%d" % lane
        traci.vehicle.add(
            vid,
            routeID,
            departPos=str(100 - i * (VEHICLE_LENGTH + DISTANCE)),
            departSpeed=str(20),  # 72 km/h
            departLane=str(lane % 3),
            typeID="vtypeauto",
        )
        plexe.set_path_cacc_parameters(vid, DISTANCE, 2, 1, 0.5)
        plexe.set_cc_desired_speed(vid, SPEED)
        plexe.set_acc_headway_time(vid, 1.0)
        plexe.use_controller_acceleration(vid, False)
        plexe.set_fixed_lane(vid, lane % 3, False)
        traci.vehicle.setSpeedMode(vid, 31)
        if i == 0:
            plexe.set_active_controller(vid, ACC)
            traci.vehicle.setColor(vid, (255, 255, 255, 255))
            topology[vid] = {}
        else:
            plexe.set_active_controller(vid, CACC)
            traci.vehicle.setColor(vid, (200, 200, 0, 255))
            topology[vid] = {
                "front": "v.%d.%d.%d" % (step // ADD_PLATOON_STEP, lane, i - 1),
                "leader": "v.%d.%d.0" % (step // ADD_PLATOON_STEP, lane),
            }


def add_platoons(plexe, topology, step):
    import random

    for lane in range(LANE_NUM):
        if random.random() < ADD_PLATOON_PRO:
            add_single_platoon(plexe, topology, step, lane)


class SumoEnvironment:
    def __init__(self, sumo_config="traditional_traffic.sumo.cfg", gui=False):
        base_path = os.path.dirname(__file__)
        self.sumo_config = os.path.join(base_path, sumo_config)
        self.gui = gui
        self.junction_id = "junction"
        self.phases = [
            "GGrGrrGGrGrr",  # Phase 0: North-South straight
            "gyrgrrgyrgrr",  # Phase 1: North-South yellow
            "grGgrrgrGgrr",  # Phase 2: North-South left
            "grygrrgrygrr",  # Phase 3: North-South left yellow
            "GrrGGrGrrGGr",  # Phase 4: East-West straight
            "grrgyrgrrgyr",  # Phase 5: East-West yellow
            "grrgrGgrrgrG",  # Phase 6: East-West left
            "grrgrygrrgry",  # Phase 7: East-West left yellow
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
        self.max_steps = 720  # 3600s / 5s per step
        self.step_count = 0
        self.plexe = None
        self.topology = {}
        self.departed_vehicles = 0
        self.prev_vehicle_count = 0
        self.observation_shape = (
            12 * 3 + 8,
        )  # Queue lengths, speeds, waiting times, phase one-hot
        self.action_count = len(self.phases)  # 8 phases
        self.last_action = 0
        self.prev_phase = 0

    def initialize_simulation(self):
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
            str(SIM_STEP_SIZE),  # Set step size to 5 seconds
        ]
        try:
            traci.start(sumo_cmd)
            print("Available traffic light IDs:", traci.trafficlight.getIDList())
        except traci.exceptions.TraCIException as e:
            print(f"Failed to start SUMO: {e}")
            sys.exit(1)
        self.plexe = Plexe()
        traci.addStepListener(self.plexe)
        self.step_count = 0
        self.topology = {}
        self.departed_vehicles = 0
        self.prev_vehicle_count = 0
        try:
            traci.trafficlight.setRedYellowGreenState(self.junction_id, self.phases[0])
        except traci.exceptions.TraCIException as e:
            print(f"Error setting traffic light state: {e}")
            traci.close()
            sys.exit(1)
        return self.get_observation()

    def take_action(self, action):
        if not 0 <= action < self.action_count:
            raise ValueError(
                f"Action {action} is invalid. Must be in [0, {self.action_count - 1}]."
            )
        self.last_action = action
        if self.step_count % 1 == 0:  # Change phase every step (5 seconds)
            try:
                traci.trafficlight.setRedYellowGreenState(
                    self.junction_id, self.phases[action]
                )
                self.prev_phase = action
            except traci.exceptions.TraCIException as e:
                print(f"Error setting traffic light phase: {e}")
                traci.close()
                sys.exit(1)
        traci.simulationStep()  # Advances 5 seconds

        if self.step_count % (ADD_PLATOON_STEP // 5) == 0:  # Adjusted for 5s steps
            add_platoons(self.plexe, self.topology, self.step_count)

        if self.step_count % 2 == 1:  # Adjusted for communication every 10s
            communicate(self.plexe, self.topology)

        self.step_count += 1
        observation = self.get_observation()
        reward = self.get_reward()
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {
            "total_waiting_time": sum(
                traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList()
            ),
            "total_queue_length": sum(
                traci.lane.getLastStepVehicleNumber(lane) for lane in self.lane_ids
            ),
            "throughput": max(0, self.prev_vehicle_count - traci.vehicle.getIDCount()),
        }
        return observation, reward, terminated, truncated, info

    def get_observation(self):
        queue_lengths = []
        avg_speeds = []
        waiting_times = []
        for lane in self.lane_ids:
            queue_lengths.append(traci.lane.getLastStepVehicleNumber(lane))
            speeds = [
                traci.vehicle.getSpeed(v)
                for v in traci.lane.getLastStepVehicleIDs(lane)
            ]
            avg_speed = np.mean(speeds) if speeds else 0
            avg_speeds.append(avg_speed)
            waiting_time = sum(
                traci.vehicle.getWaitingTime(v)
                for v in traci.lane.getLastStepVehicleIDs(lane)
            )
            waiting_times.append(
                waiting_time / max(1, traci.lane.getLastStepVehicleNumber(lane))
            )  # Normalize per vehicle
        queue_lengths = [min(q, 100) / 100 for q in queue_lengths]
        avg_speeds = [min(s, SPEED) / SPEED for s in avg_speeds]
        waiting_times = [min(w, 100) / 100 for w in waiting_times]
        current_phase = traci.trafficlight.getPhase(self.junction_id)
        phase_one_hot = [
            1 if i == current_phase else 0 for i in range(len(self.phases))
        ]
        return np.array(
            queue_lengths + avg_speeds + waiting_times + phase_one_hot, dtype=np.float32
        )

    def get_reward(self):
        vehicles = traci.vehicle.getIDList()
        total_waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
        total_queue_length = sum(
            traci.lane.getLastStepVehicleNumber(lane) for lane in self.lane_ids
        )
        stopped_vehicles = sum(
            1
            for lane in self.lane_ids
            for v in traci.lane.getLastStepVehicleIDs(lane)
            if traci.vehicle.getSpeed(v) < 0.1
        )
        current_vehicle_count = traci.vehicle.getIDCount()
        throughput = max(0, self.prev_vehicle_count - current_vehicle_count)
        self.prev_vehicle_count = current_vehicle_count
        phase_change_penalty = (
            -0.5 if self.step_count > 0 and self.last_action != self.prev_phase else 0
        )
        reward = (
            -0.5 * total_waiting_time
            - 0.3 * total_queue_length
            - 0.2 * stopped_vehicles
            + 0.3 * throughput
            + phase_change_penalty
        )
        reward = np.clip(reward / 100, -5.0, 5.0)
        return reward

    def get_observation_shape(self):
        return self.observation_shape

    def get_action_count(self):
        return self.action_count

    def close(self):
        if traci.isLoaded():
            traci.close()
