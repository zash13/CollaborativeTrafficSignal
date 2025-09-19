"""
Simple test for SumoEnv to check if the simulation runs correctly.
No testing libraries, just prints info about environment behavior.
"""

import random
import numpy as np

from sumo_env import SumoEnv, Config


def run_sanity_test(steps=50):
    print("=== SUMO Environment Sanity Test ===")

    # Create environment
    env = SumoEnv(Config)
    obs = env.reset()

    print(f"Initial observation shape: {obs.shape}")
    print(f"First obs (truncated): {obs[:10]}")

    for i in range(steps):
        # Choose random valid action
        n_phases = len(env.phases[env.tls_ids[0]])
        action = random.randint(0, n_phases - 1)

        obs, reward, done, info = env.step(action)

        print(
            f"[STEP {i}] action={action:<2} "
            f"reward={reward:>6.3f} "
            f"active_veh={info['active_vehicles']} "
            f"waiting_total={sum(obs[1::3]):.3f}"
        )

        if done:
            print("Episode finished early (done=True).")
            break

    env.close()
    print("=== Test complete ===")


if __name__ == "__main__":
    run_sanity_test(steps=20)
