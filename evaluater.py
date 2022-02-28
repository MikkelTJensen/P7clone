import json
import os
import tensorflow as tf

import grid2op
from utils.reader import read_json
from utils.paths import get_folders_in_folder
from utils.class_getters import get_reward_class, get_agent_class
from utils.filter_functions import get_filter_function
from utils.nn_utils import reload_nn_archi

from lightsim2grid import LightSimBackend
from grid2op.Runner.runner import Runner
from trainer import FILTER_FUNCTION

VERBOSE = False

NUMBER_OF_PROCESSES = 1
NUMBER_OF_EPISODES = 10
EPISODE_STEPS = 2016

def evaluate(agent_name: str, env_name: str, reward_name: str, agent_path: str):

    print(
        f"\nEvaluating agent '{agent_name}' using environment '{env_name}' and reward '{reward_name}'")
    print(f"For agent: '{agent_path}'")

    # Limit GPU usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Find the folders for agents that should be evaluated
    load_path_base = os.path.join(agent_path, 'training_results')
    folders = get_folders_in_folder(load_path_base)

    agent_class = get_agent_class(agent_name)
    reward_class = get_reward_class(reward_name)

    for folder in folders:

        if VERBOSE:
            print("\tInitializing backend")
        backend = LightSimBackend()

        with grid2op.make(env_name, reward_class=reward_class, backend=backend) as env:
            head, tail = os.path.split(folder)
            agent_step_name = tail
            agent_load_path = head

            save_path = os.path.join(
                *[agent_path, 'evaluation_results', agent_step_name])

            nn_archi = reload_nn_archi(agent_name, folder)

            converters_path = os.path.join(
                *[agent_path, 'kwargs_converters.json'])

            kwargs_converters = read_json(converters_path)
            kwargs_converters['all_actions'] = os.path.join(
                folder, 'action_space.npy')

            filter_function = None
            if FILTER_FUNCTION is not None:
                filter_function = get_filter_function(FILTER_FUNCTION)

            agent = agent_class(action_space=env.action_space,
                                name=agent_step_name,
                                store_action=NUMBER_OF_PROCESSES == 1,
                                filter_action_fun=filter_function,
                                nn_archi=nn_archi,
                                observation_space=env.observation_space,
                                **kwargs_converters)

            agent.load(agent_load_path)

            # Build runner
            runner_params = env.get_params_for_runner()
            runner_params["verbose"] = VERBOSE
            runner = Runner(**runner_params, agentClass=None,
                            agentInstance=agent)

            os.makedirs(save_path, exist_ok=True)

            if VERBOSE:
                print("Running runner\n")
            res = runner.run(path_save=save_path,
                             nb_episode=NUMBER_OF_EPISODES,
                             nb_process=NUMBER_OF_PROCESSES,
                             max_iter=EPISODE_STEPS,
                             pbar=VERBOSE)

            print(f"Evaluation summary of {agent_step_name}:")
            for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                msg_tmp = "chronics at: {}".format(chron_name)
                msg_tmp += "\ttotal score: {:.6f}".format(float(cum_reward))
                msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(
                    nb_time_step, max_ts)
                print(msg_tmp)

            dict_list = []
            total_actions = 0

            filename = os.path.join(save_path, 'actions.txt')
            # saves file with all actions performed
            with open(filename, "w") as action_file:
                for id_, (nb, act, types) in agent.dict_action.items():
                    L = [
                        "Action with ID {} was played {} times".format(
                            id_, nb),
                        "\n{}".format(act),
                        "\n-----------\n"
                    ]
                    action_file.writelines(L)
                    total_actions += nb
                    dict_list.append((id_, nb))

            dict(dict_list)

            filename = os.path.join(save_path, 'action_dict.json')

            # saves id of the performed actions in json file
            with open(filename, "w") as id_file:
                json.dump(dict(dict_list), id_file)
