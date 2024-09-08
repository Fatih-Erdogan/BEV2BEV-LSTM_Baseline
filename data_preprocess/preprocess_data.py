import os
import json
import matplotlib.pyplot as plt
import numpy as np

from .data_utils import resize_images, process_and_overwrite_waypoints


def preprocess_bevs_and_waypoints(data_path, target_size, verbose=False):
    """
    iterates over the data folder to:
      overwrite each file by:
          resize the images to the specified size (topdown),
          extract deltas out of the waypoint coordinates (measurements)
    """
    len_total_scenario = sum([os.path.isdir(os.path.join(data_path, sc)) for sc in os.listdir(data_path)])
    idx = 0
    for scenario in os.listdir(data_path):
        scenario_path = os.path.join(data_path, scenario)
        if os.path.isdir(scenario_path):
            idx += 1
            if verbose: print(f"Started working on {idx}/{len_total_scenario}th dataset...")
            for town in os.listdir(scenario_path):
                town_path = os.path.join(scenario_path, town)
                if os.path.isdir(town_path):
                    if verbose: print(f"Started preprocessing for town {town_path}...")
                    for route in os.listdir(town_path):
                        route_path = os.path.join(town_path, route)
                        if os.path.isdir(route_path):
                            topdown_path = os.path.join(route_path, 'topdown')
                            measurements_path = os.path.join(route_path, 'measurements')

                            if os.path.isdir(topdown_path):
                                resize_images(topdown_path, target_size)

                            if os.path.isdir(measurements_path):
                                process_and_overwrite_waypoints(measurements_path)
                    if verbose: print(f"Ended preprocessing for town {town_path}...\n")
            if verbose: print(f"Ended working on {idx}/{len_total_scenario}th dataset!")


def get_delta_waypoints_histogram(data_path, num_of_bins, show_plots=True, save_plots=True, out_path=None):
    delta_x_values = []
    delta_y_values = []

    for scenario_dir in os.listdir(data_path):
        scenario_path = os.path.join(data_path, scenario_dir)
        if os.path.isdir(scenario_path):
            for town_dir in os.listdir(scenario_path):
                town_path = os.path.join(scenario_path, town_dir)
                if os.path.isdir(town_path):
                    for route_dir in os.listdir(town_path):
                        route_path = os.path.join(town_path, route_dir)
                        if os.path.isdir(route_path):
                            measurements_path = os.path.join(route_path, 'measurements')
                            for file in os.listdir(measurements_path):
                                if file.endswith('.json'):
                                    file_path = os.path.join(measurements_path, file)
                                    with open(file_path, 'r') as json_file:
                                        data = json.load(json_file)
                                        waypoints = data.get('waypoints', [])
                                        for waypoint in waypoints:
                                            delta_x_values.append(waypoint[0])
                                            delta_y_values.append(waypoint[1])

    if show_plots:
        plt.figure(figsize=(10, 4))
        plt.hist(delta_x_values, bins=num_of_bins, color='blue', alpha=0.7)
        plt.title('Histogram of Delta X Values')
        plt.xlabel('Delta X')
        plt.ylabel('Frequency')
        if save_plots:
            plt.savefig(os.path.join(out_path, 'delta_x_histogram.png'))
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.hist(delta_y_values, bins=num_of_bins, color='red', alpha=0.7)
        plt.title('Histogram of Delta Y Values')
        plt.xlabel('Delta Y')
        plt.ylabel('Frequency')
        if save_plots:
            plt.savefig(os.path.join(out_path, 'delta_y_histogram.png'))
        plt.show()

    return delta_x_values, delta_y_values


def _create_bins(min_limit, max_limit, num_bins):
    full_delta = max_limit - min_limit
    bin_size = full_delta / num_bins

    limits = list()
    for i in range(num_bins + 1):
        cur_limit = min_limit + (i * bin_size)
        limits.append(cur_limit)
    return limits


def create_xy_bins(min_delta, max_delta, num_of_bins):
    # includes a special bin for the 0 case

    # num_of_bins must be odd bc 0 case
    num_x_bins = num_of_bins[0]
    num_y_bins = num_of_bins[1]
    assert (num_x_bins % 2 == 1) and (num_y_bins % 2 == 1), "Num of bins must be odd!"
    epsilon = 1e-5

    half_num_x_bins = (num_x_bins // 2) - 1  # if 13 -> 5
    half_num_y_bins = (num_y_bins // 2) - 1

    x_left = _create_bins(min_delta, 0, half_num_x_bins)
    x_right = _create_bins(0, max_delta, half_num_x_bins)
    x_left[-1] = 0 - epsilon
    x_right[0] = 0 + epsilon
    x_bin_limits = x_left + x_right
    assert len(x_bin_limits) == num_x_bins - 1

    y_left = _create_bins(min_delta, 0, half_num_y_bins)
    y_right = _create_bins(0, max_delta, half_num_y_bins)
    y_left[-1] = 0 - epsilon
    y_right[0] = 0 + epsilon
    y_bin_limits = y_left + y_right
    assert len(y_bin_limits) == num_x_bins - 1

    return x_bin_limits, y_bin_limits


# another old version
def __create_xy_bins(min_delta, max_delta, num_of_bins):
    # includes a special bin for the 0 case

    # num_of_bins must be odd bc 0 case
    num_x_bins = num_of_bins[0]
    num_y_bins = num_of_bins[1]
    assert (num_x_bins % 2 == 0) and (num_y_bins % 2 == 0), "Num of bins must be odd!"

    x_bin_separations = [min_delta]
    y_bin_separations = [min_delta]
    width = max_delta - min_delta
    step_x = width / num_x_bins
    step_y = width / num_y_bins

    for i in range(1, num_x_bins - 2):
        x_bin_separations.append((i * step_x) + min_delta)

    for i in range(1, num_y_bins - 2):
        y_bin_separations.append((i * step_y) + min_delta)

    x_bin_separations.append(max_delta)
    y_bin_separations.append(max_delta)

    return x_bin_separations, y_bin_separations


# old one, need to work on it
def _create_xy_bins(dx_list, dy_list, num_of_bins, x_edges=None, y_edges=None):
    num_x_bins = num_of_bins[0]
    num_y_bins = num_of_bins[1]
    dx_list = np.array(dx_list)
    dy_list = np.array(dy_list)
    dx_list = np.sort(np.nan_to_num(dx_list))
    dy_list = np.sort(np.nan_to_num(dy_list))

    assert len(dx_list) == len(dy_list)
    if (x_edges is None) or (y_edges is None):
        # it means you will divide the bins exactly containing same number of points
        x_bin_num_points = len(dx_list) // num_x_bins
        y_bin_num_points = len(dy_list) // num_y_bins
        x_bin_separations = list()
        y_bin_separations = list()
        for i in range(1, num_x_bins):
            x_bin_separations.append(dx_list[i * x_bin_num_points])
        for i in range(1, num_y_bins):
            y_bin_separations.append(dy_list[i * y_bin_num_points])

        # after dividing according to the number of points available inside those bins,
    else:
        raise NotImplementedError("Dont know what to do with edges :(")

    return x_bin_separations, y_bin_separations
