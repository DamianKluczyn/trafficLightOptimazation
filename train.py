import os
import sys
import torch
import numpy as np
import traci
from sumolib import checkBinary
from utils.sumo_utils import get_vehicle_numbers, get_waiting_time, phase_duration
from agent.agent import Agent
import utils.save_plot as plt
import utils.parser as pars


# Check for the SUM_HOME environment variable
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the 'SUMO_HOME' environment variable.")


# Main function to run the SUMO simulation and train the agent
def run_simulation(train=True, model="model", epochs=100, steps=1000):
    # Initialize SUMO with configuration file
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"])

    # Initialize the agent
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=8, fc1_dims=256, fc2_dims=256, batch_size=1024, n_actions=4, junctions=junction_numbers)
    # Store total waiting time for each episode
    total_waiting_times = list()
    best_time = np.inf
    best_epoch = -1

    # If training flag is set to False, load pre-trained model
    if not train:
        checkpoint = torch.load(f'models/{model}.bin', map_location=agent.Q_eval.device)
        agent.Q_eval.load_state_dict(checkpoint['state_dict'])
        best_epoch = checkpoint.get('best_epoch', 0)

    # Print the device being used by the agent's model
    print(agent.Q_eval.device)
    traci.close()

    # Training loop
    for episode in range(epochs):
        if train:
            # Start SUMO for training
            traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        else:
            # Start SUMO with GUI for evaluation
            traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        print(f'epoch: {episode}')

        # Define traffic lights phases
        select_lane = [
            ["YYYYrrrrrrrrrrrr", "GGGGrrrrrrrrrrrr"],
            ["rrrrYYYYrrrrrrrr", "rrrrGGGGrrrrrrrr"],
            ["rrrrrrrrYYYYrrrr", "rrrrrrrrGGGGrrrr"],
            ["rrrrrrrrrrrrYYYY", "rrrrrrrrrrrrGGGG"],
            ["YYYYrrrrYYYYrrrr", "GGGGrrrrGGGGrrrr"],
            ["rrrrYYYYrrrrYYYY", "rrrrGGGGrrrrGGGG"],
            ["YYYYYYYYrrrrrrrr", "GGGGGGGGrrrrrrrr"],
            ["rrrrrrrrYYYYYYYY", "rrrrrrrrGGGGGGGG"],
            ["rrrrYYYYYYYYrrrr", "rrrrGGGGGGGGrrrr"],
        ]

        # Variables for managing simulation steps and traffic light time
        step = 0
        total_time = 0
        min_duration = 3
        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicle_per_lane = dict()
        prev_action = dict()
        all_lanes = list()

        # Initialize variables for each junction
        for junction_number, junction in enumerate(all_junctions):
            traffic_lights_time[junction] = 0
            prev_wait_time[junction] = 0
            prev_vehicle_per_lane[junction_number] = 0
            prev_action[junction_number] = 0
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        # Simulation loop for each step
        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
                # Traffic light control and state updates
                controlled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controlled_lanes)
                total_time += waiting_time
                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controlled_lanes)

                    # Store previous state and current state
                    state = prev_vehicle_per_lane[junction_number]
                    state_ = list(vehicles_per_lane.values())
                    reward = waiting_time * -1
                    prev_vehicle_per_lane[junction_number] = state_
                    agent.store_transition(state, state_, prev_action[junction_number], reward, (step == steps), junction_number)

                    # Select new action
                    lane = agent.choose_action(state_)
                    prev_action[junction_number] = lane
                    phase_duration(junction, 3, select_lane[lane][0])
                    phase_duration(junction, min_duration + 7, select_lane[lane][1])

                    traffic_lights_time[junction] = min_duration + 7

                    if train:
                        agent.learn(junction_number)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
        print("total waiting time: ", total_time)
        total_waiting_times.append(total_time)

        # Check for best time and save model if necessary
        if total_time < best_time:
            best_time = total_time
            best_epoch = episode
            if train:
                agent.save(model, best_epoch)

        # Close the SUMO simulation
        traci.close()
        sys.stdout.flush()

        # Exit the loop if not training
        if not train:
            break

    # Save the plot and the results if training
    if train:
        with open("simulation_results.txt", "a") as file:
            file.write(f"\nSmallest waiting time: {best_time} occurred at epoch: {best_epoch} for model: {model}")
        print(f"Smallest waiting time: {best_time} occurred at epoch: {best_epoch}")
        plt.save_plot(total_waiting_times, model)


# Parse command line arguments and run the simulation
if __name__ == "__main__":
    options = pars.get_options()
    model = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    run_simulation(train=train, model=model, epochs=epochs, steps=steps)

    input('Press Enter to exit...')
