import os
import argparse

from trainer import train
from evaluater import evaluate

from config.legal_parameters import AGENT_LIST, ENV_LIST, REWARD_LIST
from utils.reader import read_json


def run(experiment_name: str, train_agent: bool, evaluate_agent: bool):

    if not train_agent and not evaluate_agent:
        raise RuntimeError("Agents should either be trained, evaluated or compared")

    experiment_path = os.path.join('experiments', experiment_name)

    # Check if the experiment exists
    if not os.path.exists(experiment_path):
        raise RuntimeError("Experiment folder not found")

    runner_settings_path = os.path.join(experiment_path, 'runner.json')
    runner_json = read_json(runner_settings_path)

    # First check if all entered settings are valid
    for agent in runner_json['agents']:
        # Check if agents are correct
        if agent['class'] not in AGENT_LIST:
            raise RuntimeError(f"Agent '{agent['class']}' not recognized")

        # Check if the agents folder exists
        agent_folder = os.path.join(experiment_path, agent['folder'])
        if not os.path.exists(agent_folder):
            raise RuntimeError(f"No folder found for '{agent}'")

        # Check if reload path exists
        reload_path = agent['reload']
        if reload_path is not None:
            if not os.path.exists(agent['reload']):
                raise RuntimeError(f"Reload folder {reload_path} not found")

        # Check if environments and rewards for training are correct
        if agent['train']['env'] not in ENV_LIST:
            raise RuntimeError(f"Environment '{agent['train']['env']}' not recognized")
        if agent['train']['reward'] not in REWARD_LIST:
            raise RuntimeError(f"Reward '{agent['train']['reward']}' not recognized")

        # Check if environments and rewards for evaluation are correct
        if agent['evaluate']['env'] not in ENV_LIST:
            raise RuntimeError(f"Environment '{agent['evaluate']['env']}' not recognized")
        if agent['evaluate']['reward'] not in REWARD_LIST:
            raise RuntimeError(f"Reward '{agent['evaluate']['reward']}' not recognized")

    # Then train and evaluate all agents
    for agent in runner_json['agents']:
        agent_folder = os.path.join(experiment_path, agent['folder'])
        if train_agent:
            train(agent['class'], agent['train']['env'], agent['train']['reward'],
                  agent_folder, agent['reload'])
        if evaluate_agent:
            evaluate(agent['class'], agent['evaluate']['env'], agent['evaluate']['reward'], agent_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For training and running an agent')

    parser.add_argument('--f', metavar='N', type=str, nargs=1,
                        help='Name of the experiment folder to use', required=True)

    parser.add_argument('--t', default=False, action='store_true',
                        help='Determines if agents are trained or not')

    parser.add_argument('--e', default=False, action='store_true',
                        help='Determines if the agents are evaluated or not')

    args = parser.parse_args()

    run(args.f[0], args.t, args.e)
