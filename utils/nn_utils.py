import os


def get_nn_model_path(agent_class: str, agent_name: str, path: str):
    if agent_class == 'DuelQSimple':
        from agents.DuelQSimple.DuelQ_NN import DuelQ_NN
        model_path, _ = DuelQ_NN.get_path_model(path, agent_name)
        return model_path
    elif agent_class == 'DQNPER':
        from agents.DQNPER.DQNPER_NN import DQNPER_NN
        model_path = DQNPER_NN.get_path_model(path, agent_name)
        return model_path
    elif agent_class == 'DoublePER':
        from agents.DoublePER.DoublePER_NN import DoublePER_NN
        model_path = DoublePER_NN.get_path_model(path, agent_name)
        return model_path


def init_nn_archi(agent_name: str, kwargs_archi: dict):
    if agent_name == 'DuelQSimple':
        from agents.DuelQSimple.DuelQ_NNParam import DuelQ_NNParam
        nn_archi = DuelQ_NNParam(**kwargs_archi)
        return nn_archi
    elif agent_name == 'DQNPER':
        from agents.DQNPER.DQNPER_NNParam import DQNPER_NNParam
        nn_archi = DQNPER_NNParam(**kwargs_archi)
        return nn_archi
    elif agent_name == 'DoublePER':
        from agents.DoublePER.DoublePER_NNParam import DoublePER_NNParam
        nn_archi = DoublePER_NNParam(**kwargs_archi)
        return nn_archi


def reload_nn_archi(agent_name: str, model_path: str):
    # Path example: 'experiments/tenAgents/A1/training_results/DQNPER1024'
    if agent_name == 'DuelQSimple':
        from agents.DuelQSimple.DuelQ_NNParam import DuelQ_NNParam
        nn_archi = DuelQ_NNParam.from_json(os.path.join(model_path, "nn_architecture.json"))
        return nn_archi
    elif agent_name == 'DQNPER':
        from agents.DQNPER.DQNPER_NNParam import DQNPER_NNParam
        nn_archi = DQNPER_NNParam.from_json(os.path.join(model_path, "nn_architecture.json"))
        return nn_archi
    elif agent_name == 'DoublePER':
        # model_path is a tuple here - since the target NN model has a path as well
        from agents.DoublePER.DoublePER_NNParam import DoublePER_NNParam
        nn_archi = DoublePER_NNParam.from_json(os.path.join(model_path, "nn_architecture.json"))
        return nn_archi
