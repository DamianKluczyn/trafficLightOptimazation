import traci


def get_vehicle_numbers(lanes):
    """
    Counts the number of vehicles on each lane.
    :param lanes: List of lane IDs
    :return: Dictionary with lane IDs as keys and vehicle counts as values
    """
    vehicle_per_lane = {lane: traci.lane.getLastStepVehicleNumber(lane) for lane in lanes}
    return vehicle_per_lane


def get_waiting_time(lanes):
    """
    Calculates the total waiting time for vehicles on the given lanes.
    :param lanes: List of lane IDs
    :return: Total waiting time for all vehicles on the lanes
    """
    waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    """
    Sets the traffic light phase for a given junction.
    :param junction: Junction ID
    :param phase_time: Duration of the traffic light phase
    :param phase_state: Traffic light phase state
    """
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)
