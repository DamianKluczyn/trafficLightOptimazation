import traci


def get_vehicle_numbers(lanes):
    """
    Counts the number of vehicles on each lane and returns it as a list.
    :param lanes: List of lane IDs
    :return: List with vehicle counts for each lane
    """
    vehicle_per_lane = dict()
    for lane in lanes:
        vehicle_per_lane[lane] = 0
        for id in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getLanePosition(id) > 10:
                vehicle_per_lane[lane] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    """
    Calculates the total waiting time for vehicles on the given lanes.
    :param lanes: List of lane IDs
    :return: Total waiting time for all vehicles on the lanes
    """
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
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
