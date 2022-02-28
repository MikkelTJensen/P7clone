import os
import copy
import numpy as np

from l2rpn_baselines.utils import DeepQAgent, TrainingParam

from agents.utils.SimpleBuffer import SimpleBuffer

try:
    from grid2op.Chronics import MultifolderWithCache
    _CACHE_AVAILABLE_DEEPQAGENT = True
except ImportError:
    _CACHE_AVAILABLE_DEEPQAGENT = False

DEFAULT_NAME = "DQNPER"


class DQNPER(DeepQAgent):
    def __init__(self,
                 action_space,
                 nn_archi,
                 name=DEFAULT_NAME,
                 store_action=True,
                 istraining=False,
                 filter_action_fun=None,
                 verbose=False,
                 observation_space=None,
                 **kwargs_converters):

        super().__init__(action_space,
                         nn_archi,
                         name,
                         store_action,
                         istraining,
                         filter_action_fun,
                         verbose,
                         observation_space,
                         **kwargs_converters)

        self.__graph_saved = True
        self.__nb_env = 1
        self.temp_state = None
        self.numbers_of_times_saved = 1

    def _init_replay_buffer(self):
        # Initialize the replay buffer for DQNPER
        self.replay_buffer = SimpleBuffer(self._training_param.buffer_size,
                                          self.deep_q,
                                          self._training_param.max_iter,
                                          self._training_param.min_observation)

    def _store_new_state(self, initial_state, action, reward, done, new_state):
        for i_st, act, reward, done, n_st in zip(self.temp_state, action, reward, done, new_state):
            self.replay_buffer.add(i_st, act, reward, done, n_st)

    def _next_move(self, curr_state, epsilon, training_step):
        if curr_state is not None:
            self.temp_state = copy.deepcopy(curr_state)
        # supposes that 0 encodes for do nothing, otherwise it will NOT work (for the observer)
        pm_i, pq_v, q_actions = self.deep_q.predict_movement(curr_state, epsilon, training=True)
        # TODO implement the "max XXX random action per scenarios"
        pm_i, pq_v = self._short_circuit_actions(training_step, pm_i, pq_v, q_actions)
        act = self._convert_all_act(pm_i)
        return pm_i, pq_v, act

    def _train_model(self, training_step):
        # Hack to save the model more efficiently
        self.numbers_of_steps_taken = training_step
        self._training_param.tell_step(training_step)

        if training_step > max(self._training_param.min_observation, self._training_param.minibatch_size) and \
            self._training_param.do_train():

            kwargs = {'batch_size': self._training_param.minibatch_size,
                      'training_step': training_step}

            # train the model
            s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch = \
                self.replay_buffer.sample(kwargs)

            tf_writer = None
            if self.__graph_saved is False:
                tf_writer = self._tf_writer
            loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch,
                                     tf_writer)
            # save learning rate for later
            self._train_lr = self.deep_q._optimizer_model._decayed_lr('float32').numpy()
            self.__graph_saved = True
            if not np.all(np.isfinite(loss)):
                # if the loss is not finite i stop the learning
                return False
            self._losses[training_step:] = np.sum(loss)
        return True

    def save(self, path):
        if path is not None and self.numbers_of_steps_taken > self._training_param.min_observation:
            agent_name = self.name + str(self.numbers_of_steps_taken)
            tmp_me = os.path.join(path, agent_name)
            if not os.path.exists(tmp_me):
                os.mkdir(tmp_me)
            nm_conv = "action_space.npy"
            conv_path = os.path.join(tmp_me, nm_conv)
            if not os.path.exists(conv_path):
                self.action_space.save(path=tmp_me, name=nm_conv)

            self._training_param.save_as_json(tmp_me, name="training_params.json")
            self._nn_archi.save_as_json(tmp_me, "nn_architecture.json")
            self.deep_q.save_network(tmp_me, name=self.name)

            # TODO save the "oversampling" part, and all the other info
            for nm_attr in ["_time_step_lived", "_nb_chosen", "_proba"]:
                conv_path = os.path.join(tmp_me, "{}.npy".format(nm_attr))
                attr_ = getattr(self, nm_attr)
                if attr_ is not None:
                    np.save(arr=attr_, file=conv_path)

    def load(self, path):
        # not modified compare to original implementation
        tmp_me = os.path.join(path, self.name)
        if not os.path.exists(tmp_me):
            raise RuntimeError("The model should be stored in \"{}\". But this appears to be empty".format(tmp_me))
        self._load_action_space(tmp_me)

        # TODO handle case where training param class has been overidden
        self._training_param = TrainingParam.from_json(os.path.join(tmp_me, "training_params.json".format(self.name)))
        self.deep_q = self._nn_archi.make_nn(self._training_param)
        try:
            temp_name = ''.join([i for i in self.name if not i.isdigit()])
            self.deep_q.load_network(tmp_me, name=temp_name)
        except Exception as e:
            raise RuntimeError("Impossible to load the model located at \"{}\" with error \n{}".format(path, e))

        for nm_attr in ["_time_step_lived", "_nb_chosen", "_proba"]:
            conv_path = os.path.join(tmp_me, "{}.npy".format(nm_attr))
            if os.path.exists(conv_path):
                setattr(self, nm_attr, np.load(file=conv_path))
