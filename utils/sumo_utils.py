import traci


# Counts the number of vehicles on each lane and returns it as dict
def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for lane in lanes:
        vehicle_per_lane[lane] = 0
        for id in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getLanePosition(id) > 10:
                vehicle_per_lane[lane] += 1
    return vehicle_per_lane


# Calculate the total waiting time for vehicles on the given lane
def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


# Sets the traffic light phase for a given junction
def phase_duration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)
