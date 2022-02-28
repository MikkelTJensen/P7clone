import os
import warnings
import tensorflow as tf

from utils.reader import read_json
from utils.class_getters import get_agent_class, get_reward_class, get_combined_rewards
from utils.nn_utils import init_nn_archi, reload_nn_archi
from utils.filter_functions import get_filter_function

import grid2op
from lightsim2grid import LightSimBackend
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.utils.waring_msgs import _WARN_GPU_MEMORY

VERBOSE = True
FILTER_FUNCTION = "change_bus"


def train(agent_name: str, env_name: str, reward_name: str, agent_path: str, reload_path: str):

    print(
        f"\nTraining agent '{agent_name}' using environment '{env_name}' and reward '{reward_name}'")
    print(f"For agent: '{agent_path}'")

    agent_class = get_agent_class(agent_name)
    reward_class = get_reward_class(reward_name)
    if reward_class == "CombinedScaledReward":
        EpisodeDurationReward, LinesCapacityReward, CloseToOverflowReward, BridgeReward, LinesReconnectedReward = get_combined_rewards()

    # Limit GPU usage
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except AttributeError:
        try:
            physical_devices = tf.config.experimental.list_physical_devices(
                'GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(
                    physical_devices[0], True)
        except Exception:
            warnings.warn(_WARN_GPU_MEMORY)
    except Exception:
        warnings.warn(_WARN_GPU_MEMORY)

    save_path = os.path.join(agent_path, 'training_results')
    load_path = None
    if reload_path is not None:
        # Path example: 'experiments/tenAgents/A1/training_results/DQNPER1024'
        load_path = reload_path
    log_path = os.path.join(agent_path, 'agent_logs')

    if VERBOSE:
        print("\tInitializing backend")
    backend = LightSimBackend()

    if VERBOSE:
        print(f"\tInitializing environment '{env_name}'")

    with grid2op.make(env_name, reward_class=reward_class, backend=backend) as env:
        if reward_class == "CombinedScaledReward":
            cr = env.get_reward_instance()
            cr.addReward("EpisodeDuration", EpisodeDurationReward(), 1.0)
            cr.addReward("LinesCapacity", LinesCapacityReward(), 1.0)
            cr.addReward("CloseToOverflow", CloseToOverflowReward(), 1.0)
            cr.addReward("Bridge", BridgeReward(), 1.0)
            cr.addReward("LinesReconnected", LinesReconnectedReward(), 1.0)
            cr.initialize(env)

        # Load NN architecture
        nn_archi_path = os.path.join(agent_path, 'nn_architecture.json')
        kwargs_archi = read_json(nn_archi_path)
        kwargs_archi['observation_size'] = NNParam.get_obs_size(
            env, kwargs_archi['list_attr_obs'])
        # Load action converters
        converter_path = os.path.join(agent_path, 'kwargs_converters.json')
        kwargs_converters = read_json(converter_path)

        # Get action space size for NN
        filter_function = None
        if FILTER_FUNCTION is not None:
            filter_function = get_filter_function(FILTER_FUNCTION)

        kwargs_archi["action_size"] = agent_class.get_action_size(env.action_space,
                                                                  filter_function,
                                                                  kwargs_converters)

        # Init or reload neural network
        if load_path is not None:
            if VERBOSE:
                print(
                    "\tReloading a model - the architecture parameters will be ignored")
                print(f"{load_path}")
            # Path example: 'experiments/tenAgents/A1/training_results/DQNPER1024'
            nn_archi = reload_nn_archi(agent_name, load_path)
        else:
            if VERBOSE:
                print("\tInitializing neural network architecture")
            nn_archi = init_nn_archi(agent_name, kwargs_archi)

        # Create instance of agent
        baseline = agent_class(action_space=env.action_space,
                               nn_archi=nn_archi,
                               name=agent_name,
                               istraining=True,
                               filter_action_fun=filter_function,
                               verbose=VERBOSE,
                               observation_space=env.observation_space,
                               **kwargs_converters)

        # Init training params and load from .json if an older agent should NOT be reloaded
        if load_path is not None:
            if VERBOSE:
                print("\tReloading an agent model - training parameters will be ignored")
                print(f"{load_path}")
            # All agents must have a load method to load training params
            # Agents must also have a variable 'self._training_params'
            body, head = os.path.split(load_path)
            baseline.name = head
            load_path = body
            baseline.load(load_path)
            training_param = baseline._training_param

        else:
            training_param_path = os.path.join(
                agent_path, 'training_params.json')
            training_param = TrainingParam.from_json(training_param_path)

        if VERBOSE:
            print(f"\tTraining agent '{agent_name}'")
        train_iter = training_param.max_iter + training_param.min_observation

        baseline.train(env,
                       train_iter,
                       save_path=save_path,
                       logdir=log_path,
                       training_param=training_param)
