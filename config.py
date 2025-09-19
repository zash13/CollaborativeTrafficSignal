import os


class Config:
    SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")
    NET_FILE = os.path.join(os.getcwd(), "my_4way.net.xml")
    ROUTE_FILE = os.path.join(os.getcwd(), "my_4way.rou.xml")
    SUMOCFG_FILE = os.path.join(os.getcwd(), "my_4way.sumocfg")
    SIM_STEP = 1.0
    MAX_STEPS = 600
    MAX_VEHICLES = 250
    SPAWN_MIN_INTERVAL = 0.2
    SPAWN_MAX_INTERVAL = 2.0
    DEFAULT_SPEED = 13.89  # m/s (≈50km/h)
    MIN_GREEN_TIME = 10.0
    MAX_WAIT_EXPECTED = 200.0  # Maximum expected waiting time in seconds
    MAX_VEHICLES_PER_LANE = 10
