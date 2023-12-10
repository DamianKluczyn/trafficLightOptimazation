import traci


def get_vehicle_numbers(lanes):
    """
    Counts the number of vehicles on each lane and returns it as a list.
    :param lanes: List of lane IDs
    :return: List with vehicle counts for each lane
    """
    relevant_lanes = ["-E3_0", "-E3_1", "-E0_0", "-E0_1", "-E1_0", "-E1_1", "-E2_0", "-E2_1"]
    vehicle_counts = []
    for lane in lanes:
        if lane in relevant_lanes and lane not in vehicle_counts:
            vehicle_counts.append(traci.lane.getLastStepVehicleNumber(lane))
    return vehicle_counts

def get_waiting_time(lanes):
    """
    Calculates the total waiting time for vehicles on the given lanes.
    :param lanes: List of lane IDs
    :return: Total waiting time for all vehicles on the lanes
    """
    waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    return waiting_time


def phase_duration(junction, phase_time, phase_state):
    """
    Sets the traffic light phase for a given junction.
    :param junction: Junction ID
    :param phase_time: Duration of the traffic light phase
    :param phase_state: Traffic light phase state
    """
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)
