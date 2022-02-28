import numpy as np


def get_filter_function(action_filter: str):
    if action_filter == 'change_bus':
        return change_bus_filter
    return None


def change_bus_filter(action) -> bool:
    # Change bus only on one Bus
    count = 0
    for bus in action.change_bus:
        if bus:
            count += 1
    if count > 1:
        return False

    return True