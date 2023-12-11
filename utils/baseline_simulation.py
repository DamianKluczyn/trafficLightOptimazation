import traci
from sumolib import checkBinary
from utils.sumo_utils import get_waiting_time
import sys


def run_baseline_simulation(steps=1000):
    """
    Function to run the SUMO simulation without the agent's intervention.
    """
    total_waiting_times_baseline = list()  # List to store total waiting time for each episode

    # Main loop for running the baseline simulation
    traci.start([checkBinary("sumo"), "-c", "../configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])

    step = 0
    total_time = 0

    while step <= steps:
        traci.simulationStep()
        for junction in traci.trafficlight.getIDList():
            controlled_lanes = traci.trafficlight.getControlledLanes(junction)
            waiting_time = get_waiting_time(controlled_lanes)
            total_time += waiting_time
        step += 1

    print("Total waiting time (baseline): ", total_time)
    total_waiting_times_baseline.append(total_time)

    traci.close()
    with open("../simulation_results.txt", "a") as file:
        file.write(f"\nWaiting time for base simulation: {total_time}")
    print(f"Waiting time for base simulation: {total_time}")
    sys.stdout.flush()


if __name__ == "__main__":
    run_baseline_simulation(1000)
