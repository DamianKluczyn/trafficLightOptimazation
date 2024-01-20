import random


def generate_training_strategies():
    vehicles = []
    vehicles_alt = []
    vehicle_id = 0
    routes = ["-E0 E1", "-E0 E2", "-E0 E3", "-E1 E0", "-E1 E2", "-E1 E3", "-E2 E0", "-E2 E1", "-E2 E3", "-E3 E0", "-E3 E1", "-E3 E2"]

    # Strategia pasywna
    for depart_time in range(0, 601, 3):
        route = routes[random.randrange(0,11)]
        vehicles.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <route edges="{route}"/>
            </vehicle>\n''')
        vehicles_alt.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <routeDistribution last="0">
                    <route cost="14.40" probability="1.00000000" edges="{route}"/>
                </routeDistribution>
            </vehicle>''')
        vehicle_id += 1

    # Strategia agresywna
    for depart_time in range(601, 1201, 1):
        route = routes[random.randrange(0,11)] if vehicle_id % 3 == 0 else "-E0 E2"
        vehicles.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <route edges="{route}"/>
            </vehicle>\n''')
        vehicles_alt.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <routeDistribution last="0">
                    <route cost="14.40" probability="1.00000000" edges="{route}"/>
                </routeDistribution>
            </vehicle>''')
        vehicle_id += 1

    # Strategia mieszana
    for depart_time in range(1201, 1900, 2):
        route = routes[random.randrange(0,11)]
        vehicles.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <route edges="{route}"/>
            </vehicle>\n''')
        vehicles_alt.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <routeDistribution last="0">
                    <route cost="14.40" probability="1.00000000" edges="{route}"/>
                </routeDistribution>
            </vehicle>''')
        vehicle_id += 1

    return [vehicles, vehicles_alt]


def generate_passive_strategies():
    vehicles = []
    vehicles_alt = []
    vehicle_id = 0
    routes = ["-E0 E1", "-E0 E2", "-E0 E3", "-E1 E0", "-E1 E2", "-E1 E3", "-E2 E0", "-E2 E1", "-E2 E3", "-E3 E0",
              "-E3 E1", "-E3 E2"]

    for depart_time in range(0, 1980, 3):
        route = routes[random.randrange(0, 11)]
        vehicles.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <route edges="{route}"/>
            </vehicle>\n''')
        vehicles_alt.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <routeDistribution last="0">
                    <route cost="14.40" probability="1.00000000" edges="{route}"/>
                </routeDistribution>
            </vehicle>''')
        vehicle_id += 1

    return [vehicles, vehicles_alt]


def generate_aggressive_strategies():
    vehicles = []
    vehicles_alt = []
    vehicle_id = 0
    routes = ["-E0 E1", "-E0 E2", "-E0 E3", "-E1 E0", "-E1 E2", "-E1 E3", "-E2 E0", "-E2 E1", "-E2 E3", "-E3 E0",
              "-E3 E1", "-E3 E2"]

    for depart_time in range(0, 1900, 1):
        route = routes[random.randrange(0, 11)] if vehicle_id % 3 == 0 else "-E0 E2"
        vehicles.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <route edges="{route}"/>
            </vehicle>\n''')
        vehicles_alt.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <routeDistribution last="0">
                    <route cost="14.40" probability="1.00000000" edges="{route}"/>
                </routeDistribution>
            </vehicle>''')
        vehicle_id += 1

    return [vehicles, vehicles_alt]


def generate_mixed_strategies():
    vehicles = []
    vehicles_alt = []
    vehicle_id = 0
    routes = ["-E0 E1", "-E0 E2", "-E0 E3", "-E1 E0", "-E1 E2", "-E1 E3", "-E2 E0", "-E2 E1", "-E2 E3", "-E3 E0",
              "-E3 E1", "-E3 E2"]

    for depart_time in range(0, 1900, 2):
        route = routes[random.randrange(0, 11)]
        vehicles.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <route edges="{route}"/>
            </vehicle>\n''')
        vehicles_alt.append(f'''
            <vehicle id="{vehicle_id}" depart="{depart_time}.00">
                <routeDistribution last="0">
                    <route cost="14.40" probability="1.00000000" edges="{route}"/>
                </routeDistribution>
            </vehicle>''')
        vehicle_id += 1

    return [vehicles, vehicles_alt]


#vehicles_xml = generate_training_strategies()
#vehicles_xml = generate_passive_strategies()
#vehicles_xml = generate_aggressive_strategies()
vehicles_xml = generate_mixed_strategies()
xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
{''.join(vehicles_xml[0])}
</routes>
"""
xml_content2 = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
{''.join(vehicles_xml[1])}
</routes>
"""

#file_path = ['../maps/training_map.rou.xml', '../maps/training_map.rou.alt.xml']
#file_path = ['../maps/test_map_passive.rou.xml', '../maps/test_map_passive.rou.alt.xml']
#file_path = ['../maps/test_map_aggressive.rou.xml', '../maps/test_map_aggressive.rou.alt.xml']
file_path = ['../maps/test_map_mixed.rou.xml', '../maps/test_map_mixed.rou.alt.xml']
with open(file_path[0], 'w') as f:
    f.write(xml_content)
with open(file_path[1], 'w') as f:
    f.write(xml_content2)
