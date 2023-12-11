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


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the 'SUMO_HOME' environment variable.")


last_action_time = 0  # Time when the last action (light change) was taken
current_phase = 0     # Current phase of traffic lights
yellow_phase = {...}  # Dictionary mapping each phase to its corresponding yellow phase
all_phases = {...}    # Dictionary of all traffic light phases including yellow


def run_simulation(train=True, model="model", epochs=100, steps=1000):
    """
    Main function to run the SUMO simulation and train the agent.
    """
    # Initialize SUMO
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"])

    # Initialize the agent
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    agent = Agent(gamma=0.99, epsilon=0.0, lr=0.001, input_dims=8, fc1_dims=256, fc2_dims=256, batch_size=1024, n_actions=4, junctions=junction_numbers)
    total_waiting_times = list()  # List to store total waiting time for each episode
    best_time = np.inf

    # Main loop for training
    if not train:
        agent.Q_eval.load_state_dict(torch.load(f'models/{model}.bin', map_location=agent.Q_eval.device))

    print(agent.Q_eval.device)
    traci.close()
    for episode in range(epochs):
        if train:
            traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        else:
            traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        print(f'epoch: {episode}')

        select_lane = [
            ["YYYYrrrrrrrrrrrr", "GGGGrrrrrrrrrrrr"],
            ["rrrrYYYYrrrrrrrr", "rrrrGGGGrrrrrrrr"],
            ["rrrrrrrrYYYYrrrr", "rrrrrrrrGGGGrrrr"],
            ["rrrrrrrrrrrrYYYY", "rrrrrrrrrrrrGGGG"],
        ]

        step = 0
        total_time = 0
        min_duration = 5
        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicle_per_lane = dict()
        prev_action = dict()
        all_lanes = list()

        for junction_number, junction in enumerate(all_junctions):
            traffic_lights_time[junction] = 0
            prev_wait_time[junction] = 0
            prev_vehicle_per_lane[junction_number] = 0
            prev_action[junction_number] = 0
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
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
                    phase_duration(junction, 6, select_lane[lane][0])
                    phase_duration(junction, min_duration + 10, select_lane[lane][1])

                    traffic_lights_time[junction] = min_duration + 10

                    if train:
                        agent.learn(junction_number)
                else:
                    traffic_lights_time[junction_number] -= 1
            step += 1
        print("total waiting time: ", total_time)
        total_waiting_times.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                agent.save(model)

        traci.close()
        sys.stdout.flush()
        if not train:
            break
    if train:
        plt.save_plot(total_waiting_times, model)


if __name__ == "__main__":
    options = pars.get_options()
    model = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    run_simulation(train=train, model=model, epochs=epochs, steps=steps)
