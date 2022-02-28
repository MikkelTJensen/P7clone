import os
import argparse
import shutil

FOLDER_LIST = ["agent_logs",
              "evaluation_results",
              "training_results",
              "comparison_results"]

def main(exp_folder):

    exp_path = os.path.join('experiments', exp_folder)

    # Check if the experiment exists
    if not os.path.exists(exp_path):
        raise RuntimeError("Experiment folder not found")

    # List all files in the exp folder
    a_list = os.listdir(exp_path)

    # Adds the full path, to all
    a_paths = [os.path.join(exp_path,x) for x in a_list]

    # Filter for folders
    a_folders = [x for x in a_paths if os.path.isdir(x)]

    for agent in a_folders:
        for folder in FOLDER_LIST:

            delete_folder = os.path.join(agent, folder)

            if os.path.exists(delete_folder):
                try:
                    shutil.rmtree(delete_folder)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

    print("Deleted folders in", exp_folder)

# python3 deleter.py --f EXP_FOLDER_NAME
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For deleting folder for agents')

    parser.add_argument('--f', metavar='N', type=str, nargs=1,
                        help='Name of the experiment folder', required=True)

    args = parser.parse_args()

    main(args.f[0])
