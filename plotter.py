import os
import csv
from utils.reader import read_json
from matplotlib import pyplot as plt
from utils.paths import get_folders_in_folder, get_folders

EXP_PATH = 'experiments/threeAgents'
AGENT = 'DoublePER7168'

PLOT_ALL = True
AVERAGE_AGENT_STEPS = True
AVERAGE_AGENT_SCORE = False
AVERAGE_STEPS_LIST = []


def average_agent_score():
    agent_folders = get_folders_in_folder(EXP_PATH)
    agent_folders = [
        f'{folder}/comparison_results/{AGENT}' for folder in agent_folders]
    for i, folder in enumerate(agent_folders):
        print(f'===== A{i + 1} =====')
        folder = os.path.join(folder, 'scores.csv')
        with open(folder) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            average = 0
            for row in csv_reader:
                for num in row:
                    print(f'{float(num):.2f}')
                    average += float(num)
                print(f'AVERAGE: {average/len(row)}')


def average_agent_steps():
    agent_folders = get_folders_in_folder(EXP_PATH)
    agent_folders = [os.path.join(folder, 'evaluation_results')
                     for folder in agent_folders]
    for folder in agent_folders:
        evaluation_folders = get_folders_in_folder(folder)
        add_average_step_to_list(evaluation_folders)
    plot_agents()


def add_average_step_to_list(folders: list):
    average_step_list = []
    for folder in folders:
        steps_taken = []
        chronic_folders = get_folders_in_folder(folder)
        chronic_folders.sort()
        for chronic in chronic_folders:
            json_dict = read_json(chronic + '/episode_meta.json')
            steps_taken.append(json_dict['nb_timestep_played'])
        average_step_list.append(sum(steps_taken) / len(steps_taken))
    AVERAGE_STEPS_LIST.append(average_step_list)


def plot_agents():
    if PLOT_ALL:
        for averages in AVERAGE_STEPS_LIST:
            plt.plot(averages)
            plt.xticks(range(len(averages)), [
                       i + 1 for i, _ in enumerate(averages)])
    else:
        average_of_average = []
        for averages in AVERAGE_STEPS_LIST:
            average_of_average.append(sum(averages) / len(averages))
        plt.plot(average_of_average)

    plt.title('Average Steps Taken over Training Iterations')
    plt.ylabel('Average Steps Taken')
    plt.xlabel('Training Iterations')
    plt.show()


if __name__ == '__main__':
    if AVERAGE_AGENT_STEPS:
        average_agent_steps()
    if AVERAGE_AGENT_SCORE:
        average_agent_score()
