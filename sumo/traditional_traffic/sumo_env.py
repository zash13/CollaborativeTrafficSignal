import gym
import numpy as np
import traci
import os
import sys
from plexe import Plexe, ACC, CACC
from utils import communicate

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

VEHICLE_LENGTH = 4
DISTANCE = 6
LANE_NUM = 12
PLATOON_SIZE = 1
SPEED = 16.6
V2I_RANGE = 200
PLATOON_LENGTH = VEHICLE_LENGTH * PLATOON_SIZE + DISTANCE * (PLATOON_SIZE - 1)
ADD_PLATOON_PRO = 0.3
ADD_PLATOON_STEP = 600
MAX_ACCEL = 2.6
STOP_LINE = 15.0


def add_single_platoon(plexe, topology, step, lane):
    import random

    for i in range(PLATOON_SIZE):
        vid = "v.%d.%d.%d" % (step / ADD_PLATOON_STEP, lane, i)
        routeID = "route_%d" % lane
        traci.vehicle.add(
            vid,
            routeID,
            departPos=str(100 - i * (VEHICLE_LENGTH + DISTANCE)),
            departSpeed=str(5),
            departLane=str(lane % 3),
            typeID="vtypeauto",
        )
        plexe.set_path_cacc_parameters(vid, DISTANCE, 2, 1, 0.5)
        plexe.set_cc_desired_speed(vid, SPEED)
        plexe.set_acc_headway_time(vid, 1.5)
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
                "front": "v.%d.%d.%d" % (step / ADD_PLATOON_STEP, lane, i - 1),
                "leader": "v.%d.%d.0" % (step / ADD_PLATOON_STEP, lane),
            }


def add_platoons(plexe, topology, step):
    import random

    for lane in range(LANE_NUM):
        if random.random() < ADD_PLATOON_PRO:
            add_single_platoon(plexe, topology, step, lane)


class SumoEnv(gym.Env):
    def __init__(self, sumo_config="traditional_traffic.sumo.cfg", gui=False):
        super(SumoEnv, self).__init__()
        self.sumo_config = sumo_config
        self.gui = gui
        # Assume 4 traffic light phases (adjust based on .net.xml)
        self.action_space = gym.spaces.Discrete(4)
        # Observations: queue lengths for 8 lanes (adjust based on network)
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(8,), dtype=np.float32
        )
        self.max_steps = 3600  # 1 hour simulation
        self.step_count = 0
        self.plexe = None
        self.topology = {}
        self.departed_vehicles = 0
        self.prev_vehicle_count = 0

    def reset(self):
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "--duration-log.statistics",
            "--tripinfo-output",
            "output_file.xml",
            "-c",
            self.sumo_config,
        ]
        traci.start(sumo_cmd)
        self.plexe = Plexe()
        traci.addStepListener(self.plexe)
        self.step_count = 0
        self.topology = {}
        self.departed_vehicles = 0
        self.prev_vehicle_count = 0
        return self._get_observation()

    def step(self, action):
        # Apply traffic light phase (replace 'junction_id' with actual ID from .net.xml)
        traci.trafficlight.setPhase("junction_id", action)
        traci.simulationStep()

        if self.step_count % ADD_PLATOON_STEP == 0:
            add_platoons(self.plexe, self.topology, self.step_count)

        if self.step_count % 10 == 1:
            communicate(self.plexe, self.topology)

        self.step_count += 1
        obs = self._get_observation()
        reward = self._calculate_reward()
        done = self.step_count >= self.max_steps
        info = {}
        return obs, reward, done, info

    def _get_observation(self):
        # Get queue lengths for up to 8 lanes (adjust based on .net.xml)
        lanes = traci.lane.getIDList()[:8]  # Select first 8 lanes
        queue_lengths = [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]
        # Normalize to [0, 1] assuming max queue length of 100
        return np.array([min(q, 100) / 100 for q in queue_lengths], dtype=np.float32)

    def _calculate_reward(self):
        # Calculate total waiting time
        vehicles = traci.vehicle.getIDList()
        total_waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicles)
        # Calculate total queue length
        total_queue_length = sum(
            traci.lane.getLastStepVehicleNumber(lane) for lane in traci.lane.getIDList()
        )
        # Calculate throughput (vehicles that left the simulation)
        current_vehicle_count = traci.vehicle.getIDCount()
        throughput = max(
            0, self.prev_vehicle_count - current_vehicle_count
        )  # Approximate departed vehicles
        self.prev_vehicle_count = current_vehicle_count
        # Reward: minimize waiting time and queue length, maximize throughput
        reward = -0.5 * total_waiting_time - 0.3 * total_queue_length + 0.2 * throughput
        # Normalize reward to approximate LunarLander range (-250, 300)
        reward = np.clip(reward / 100, -2.5, 3.0)
        return reward

    def close(self):
        traci.close()
