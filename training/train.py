import os
import sys
import optparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import traci
from sumolib import checkBinary
from utils.sumo_utils import get_vehicle_numbers, get_waiting_time, phaseDuration
from agent.agent import Agent


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the 'SUMO_HOME' environment variable.")


last_action_time = 0  # Time when the last action (light change) was taken
current_phase = 0     # Current phase of traffic lights
yellow_phase = {...}  # Dictionary mapping each phase to its corresponding yellow phase
all_phases = {...}    # Dictionary of all traffic light phases including yellow


def run_simulation(total_episodes=100, steps_per_episode=1000):
    """
    Main function to run the SUMO simulation and train the agent.
    """
    # Initialize SUMO
    traci.start([checkBinary("sumo-gui"), "-c", "path/to/your/config.sumocfg"])

    # Initialize the agent
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=[4], batch_size=32, n_actions=4)
    total_waiting_times = []  # List to store total waiting time for each episode

    # Main loop for training
    for episode in range(total_episodes):
        traci.load(["-c", "path/to/your/config.sumocfg"])
        state = initialize_state()
        total_waiting_time = 0

        for step in range(steps_per_episode):
            action = agent.choose_action(state)
            apply_action_to_sumo(action)
            new_state = get_new_state_from_sumo()
            reward = calculate_reward(state, new_state)
            done = False  # Define your termination condition here
            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            total_waiting_time += get_waiting_time(state)  # Update total waiting time

        total_waiting_times.append(total_waiting_time)  # Store the total waiting time for this episode

        # Save the model after each episode
        torch.save(agent.Q_eval.state_dict(), f'models/model_ep{episode}.pth')

        # Generate plots after each episode
        plt.plot(total_waiting_times)
        plt.xlabel('Episodes')
        plt.ylabel('Total Waiting Time')
        plt.title('Waiting Time Over Episodes')
        plt.savefig(f'plots/waiting_time_ep{episode}.png')
        plt.close()

    traci.close()


def initialize_state():
    """
    Initialize the state for the agent based on the SUMO environment.
    The state could include information like the number of cars on each lane, waiting times, etc.
    """
    controlled_lanes = traci.trafficlight.getControlledLanes("junctionID")  # Adjust "junctionID" as needed
    state = get_vehicle_numbers(controlled_lanes)
    return state


def apply_action_to_sumo(action, junction_id="junctionID"):
    """
    Apply the chosen action to the SUMO environment.
    Incorporates a yellow light phase to ensure safe transitions between light states.
    """
    global last_action_time, current_phase, yellow_phase

    # Define the duration of yellow lights
    yellow_duration = 2

    # Check if it's safe to change the lights (i.e., not in a yellow phase)
    if traci.simulation.getTime() - last_action_time > yellow_duration:
        if action != current_phase:
            # Set yellow phase
            traci.trafficlight.setRedYellowGreenState(junction_id, yellow_phase[current_phase])
            traci.trafficlight.setPhaseDuration(junction_id, yellow_duration)
            last_action_time = traci.simulation.getTime()

            # Schedule the next phase after yellow
            traci.trafficlight.setRedYellowGreenState(junction_id, all_phases[action])
            current_phase = action
    else:
        # If it's still yellow, extend the yellow phase
        traci.trafficlight.setPhaseDuration(junction_id, yellow_duration - (traci.simulation.getTime() - last_action_time))


def get_new_state_from_sumo():
    """
    Get the new state from the SUMO environment after applying an action.
    This could include updated vehicle counts and waiting times.
    """
    controlled_lanes = traci.trafficlight.getControlledLanes("junctionID")
    new_state = get_vehicle_numbers(controlled_lanes)
    return new_state


def calculate_reward(old_state, new_state):
    """
    Calculate the reward based on the old and new state.
    A simple reward function could be based on the reduction of waiting times.
    """
    old_waiting_time = sum(old_state.values())
    new_waiting_time = sum(new_state.values())
    reward = old_waiting_time - new_waiting_time  # Reward is positive if waiting time is reduced
    return reward


if __name__ == "__main__":
    run_simulation()
