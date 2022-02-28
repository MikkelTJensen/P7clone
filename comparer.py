import argparse
import json
import os
from os.path import join
import csv

import grid2op
from lightsim2grid import LightSimBackend
import numpy as np
import matplotlib.pyplot as plt

from evaluater import NUMBER_OF_PROCESSES
from utils.class_getters import *
from utils.nn_utils import *
from utils.paths import get_folders, get_folders_in_folder
from utils.reader import read_json
from utils.result_getters import get_score

VERBOSE = False
NUMBER_OF_SCENARIOS = 10
EPISODE_STEPS = 2016


def compare_score(agent_class: str, reward_class: str, env_name: str, agent_path: str, step_name: str):

    agent_class_ = get_agent_class(agent_class)
    reward_class_ = get_reward_class(reward_class)

    train_path = join(agent_path, 'training_results')

    train_step_path = join(train_path, step_name)
    comp_step_path = join(agent_path, 'comparison_results', step_name)

    if VERBOSE:
        print("\tInitializing backend")
    backend = LightSimBackend()

    with grid2op.make(env_name, reward_class=reward_class_, backend=backend) as env:
        nn_archi = reload_nn_archi(agent_class, train_step_path)
        converters_path = join(agent_path, 'kwargs_converters.json')
        kwargs_converters = read_json(converters_path)
        kwargs_converters['all_actions'] = join(
            train_step_path, 'action_space.npy')

        agent_ = agent_class_(action_space=env.action_space,
                              name=step_name,
                              store_action=NUMBER_OF_PROCESSES == 1,
                              nn_archi=nn_archi,
                              observation_space=env.observation_space,
                              **kwargs_converters)
        agent_.load(train_path)
        os.makedirs(comp_step_path, exist_ok=True)

        if env_name == "l2rpn_neurips_2020_track2_small":
            mix_names = list(env.keys())
            env = env[mix_names[0]]

        score = get_score(agent=agent_,
                          env=env,
                          nb_scenario=NUMBER_OF_SCENARIOS,
                          save_path=comp_step_path)

        file_name = join(comp_step_path, 'scores.csv')

        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(score)

        if VERBOSE:
            print(score)

        return score


def analyze_actions(eval_path):

    # Gets steps paths and deletes the last
    step_paths = get_folders_in_folder(eval_path)

    # All the action dictionaries form eval
    dict_list = []

    # Dict used to merge all the action dicts
    merge_dict = dict()
    action_id_list = []

    # Each action has a list of times for each step
    id_list_list = []

    for step_path in step_paths:
        id_dict_path = join(step_path, 'action_dict.json')
        id_dict = read_json(id_dict_path)
        dict_list.append(id_dict)
        merge_dict.update(id_dict)
        for index, key in enumerate(id_dict):
            if key not in action_id_list:
                action_id_list.append(key)
                id_list_list.append([])
    x_list = range(len(step_paths))

    # For each step
    for step, path in enumerate(step_paths):
        keys = list(dict_list[step].keys())
        norm_list = []
        sum = 0
        for index, key in enumerate(action_id_list):
            if key in keys:
                norm_list.append(dict_list[step][key])
                sum = sum + dict_list[step][key]
            else:
                norm_list.append(0)
        norm_list = [num/sum for num in norm_list]
        for index, key in enumerate(action_id_list):
            id_list_list[index].append(norm_list[index])

    # How high the bars should be placed.
    sum_list = [0] * len(dict_list)
    sum_list = np.array(sum_list)
    id_list_list = np.array(id_list_list)

    for times, action in zip(id_list_list, action_id_list):
        plt.bar(x_list, times, bottom=sum_list, label=action)
        sum_list = np.add(times, sum_list)

    plt.xticks(x_list)
    plt.title('Action Distribution over Training Iterations')
    plt.ylabel('Action Distribution')
    plt.xlabel('Training Iterations')

    if VERBOSE:
        for times, action in zip(id_list_list, action_id_list):
            print("ID: ", action, "Times:", times)
        plt.show()

    agent_path, eval_folder = os.path.split(eval_path)
    comp_path = join(agent_path, 'comparison_results')
    os.makedirs(comp_path, exist_ok=True)
    filename = os.path.join(
        agent_path, 'comparison_results', 'action_analysis')
    plt.savefig(fname=filename)

# The job of this method is to catch general mistakes before comparing
# Like if you have trained/evaluated something.
def run(folder_path: str, should_action: bool, should_score: bool):

    if not should_action and not should_score:
        raise RuntimeError("Action or Score should be compared")

    full_path = join('experiments', folder_path)
    run_path = join(full_path, 'runner.json')

    # Check if the experiment exists
    if not os.path.exists(full_path):
        raise RuntimeError("Experiment folder not found")

    json_data = read_json(run_path)
    made_header = False

    # For each agents
    for agent in json_data['agents']:
        agent_path = join(full_path, agent['folder'])
        # Setup for each steps agent
        train_path = join(agent_path, 'training_results')
        eval_path = join(agent_path, 'evaluation_results')

        step_folders = None
        if should_action:
            if not os.path.exists(eval_path):
                raise RuntimeError(
                    f"'{eval_path}' folder not found, evaluate the Agent first")
            step_folders = get_folders(eval_path)

        if should_action or should_score:
            if not os.path.exists(train_path):
                raise RuntimeError(
                    f"'{train_path}' folder not found, train the Agent first")
            step_folders = get_folders(train_path)

        if len(step_folders) > 1:
            step_index = -2
        else:
            step_index = 0

        if not made_header and should_score:
            header = ['Agent']
            header.extend(range(1, NUMBER_OF_SCENARIOS + 1))
            score_file_path = join(
                full_path, f'score_{step_folders[step_index]}.csv')
            with open(score_file_path, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(header)
            made_header = True

        # This is run once for each agent
        if should_action:
            print('\n', "Analyzing actions for:", agent_path)
            analyze_actions(eval_path=eval_path)

        if should_score:
            print('\n', "Comparing Score for:",
                  agent_path, step_folders[step_index])
            score = compare_score(agent_class=agent['class'],
                                  reward_class=agent['evaluate']['reward'],
                                  env_name=agent['evaluate']['env'],
                                  agent_path=agent_path,
                                  step_name=step_folders[step_index])

            row = [agent['folder']]
            row.extend(score)
            with open(score_file_path, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(row)


# python3 comparer.py --f EXP_FOLDER_NAME --r --a --c
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For comparing agents in an experiment')

    parser.add_argument('--f', metavar='N', type=str, nargs=1,
                        help='Name of the experiment folder', required=True)

    parser.add_argument('--a', default=False, action='store_true',
                        help='Determines if actions should be compared')

    parser.add_argument('--c', default=False, action='store_true',
                        help='Determines if scores should be compared')

    args = parser.parse_args()

    run(args.f[0], args.a, args.c)
