import os
from os.path import join

import grid2op
import numpy as np
from config.legal_parameters import NPZ_LIST
from grid2op.utils import ScoreL2RPN2020
from lightsim2grid import LightSimBackend

from utils.class_getters import *
from utils.nn_utils import *
from utils.paths import get_folders, get_folders_in_folder
from utils.reader import read_json, read_npz


def get_reward(np_episode: int, env_name: str, eval_step_path: str):

    npz_file_name = "rewards"

    if npz_file_name not in NPZ_LIST:
        raise RuntimeError(f"NPZ filename '{npz_file_name}' not recognized")

    agent_reward = []
    eval_step_paths = get_folders_in_folder(eval_step_path)
    npz_file = npz_file_name + '.npz'

    for eval_path in eval_step_paths:
        chron_path = os.path.join(eval_path, npz_file)
        data = read_npz(chron_path)
        agent_reward.append(data)

    return np.array(agent_reward)

# Computes a score for an agent, returns an array


def get_score(agent, env, nb_scenario: int, save_path: str):
    my_score = ScoreL2RPN2020(env,
                              nb_scenario=nb_scenario,
                              env_seeds=[0 for _ in range(nb_scenario)],
                              agent_seeds=[0 for _ in range(nb_scenario)],
                              verbose=True)
    return my_score.get(agent=agent, path_save=save_path)[0]


# Method to get id for actions taken at all singular time steps
def get_actions(np_episode: int, agent, env_name: str, eval_step_path: str, train_step_path: str):

    # Load the action space to get action
    action_space_path = join(train_step_path, 'action_space.npy')
    action_space = np.load(action_space_path)

    action_space_list = []

    id_dict_path = join(eval_step_path, 'action_dict.json')
    id_dict = read_json(id_dict_path)

    id_list = id_dict.keys()
    id_list = [int(key) for key in id_list]

    for id in id_list:
        action_space_list.append(
            agent.action_space.from_vect(action_space[id]))

    npz_file_name = "actions"

    if npz_file_name not in NPZ_LIST:
        raise RuntimeError(f"NPZ filename '{npz_file_name}' not recognized")

    # Folder with the chronics in
    chronics_folder = os.path.join(
        grid2op.get_current_local_dir(), env_name, "chronics")

    chron_fold = get_folders(chronics_folder)

    all_action_list = []

    for chron in range(np_episode):

        # Convert to chron string
        chron_string = chron_fold[chron]

        chron_path = os.path.join(
            eval_step_path, chron_string, npz_file_name + '.npz')

        data = read_npz(chron_path)

        action_list = []

        # This is run for each chronic
        for step in data:
            # Check if the first is Nan meaning it is blackoutet
            if np.isnan(step[0]):
                break

            # Convert the action to a grid object
            act_as = agent.action_space.from_vect(step)
            # append id of the action to list
            for i in range(len(action_space_list)):
                if action_space_list[i] == act_as:
                    action_list.append(id_list[i])

        all_action_list.append(action_list)

    return all_action_list
