import traci
from sumolib import checkBinary
from utils.sumo_utils import get_waiting_time
import sys
import utils.save_plot as plt


# Function to run the SUMO without agent's intervention to measure the performance without AI control
def run_baseline_simulation(steps=2000):
    # Store total waiting time for each episode
    total_waiting_times_baseline = list()

    # Start the SUMO
    traci.start([checkBinary("sumo"), "-c", "../configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])

    avg_waiting_times_per_step = []
    step = 0
    total_time = 0

    # Run the simulation for the specified number of steps
    while step <= steps:
        traci.simulationStep()
        avg_waiting_time = total_time / (step + 1)
        avg_waiting_times_per_step.append(avg_waiting_time)
        for junction in traci.trafficlight.getIDList():
            controlled_lanes = traci.trafficlight.getControlledLanes(junction)
            waiting_time = get_waiting_time(controlled_lanes)
            total_time += waiting_time
        step += 1

    # Print and record the total waiting time
    print("Total waiting time (baseline): ", total_time)
    total_waiting_times_baseline.append(total_time)

    traci.close()

    # Write the total waiting time to a file
    with open("../simulation_results.txt", "a") as file:
        file.write(f"\nWaiting time for base simulation: {total_time}")
    print(f"Waiting time for base simulation: {total_time}")


    plt.save_base_plot(avg_waiting_times_per_step)

    # Clear internal buffer
    sys.stdout.flush()


if __name__ == "__main__":
    run_baseline_simulation(2000)
