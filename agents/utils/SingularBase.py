import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as tfko

from l2rpn_baselines.utils.TrainingParam import TrainingParam


class SingularBaseQ(ABC):
    def __init__(self,
                 nn_params,
                 training_param=None,
                 verbose=False):

        self._action_size = nn_params.action_size
        self._observation_size = nn_params.observation_size
        self._nn_archi = nn_params
        self.verbose = verbose

        if training_param is None:
            self._training_param = TrainingParam()
        else:
            self._training_param = training_param

        self._lr = training_param.lr
        self._lr_decay_steps = training_param.lr_decay_steps
        self._lr_decay_rate = training_param.lr_decay_rate

        self._model = None
        self._schedule_model = None
        self._optimizer_model = None
        self._custom_objects = None

    def make_optimizer(self):
        # Create an Adam optimizer
        schedule = tfko.schedules.InverseTimeDecay(self._lr, self._lr_decay_steps, self._lr_decay_rate)
        return schedule, tfko.Adam(learning_rate=schedule)

    @abstractmethod
    def construct_q_network(self):
        raise NotImplementedError("Not implemented")

    def predict_movement(self, data, epsilon, batch_size=None, training=False):
        """
        Predict movement of game controler where is epsilon probability randomly move.'
        """
        if batch_size is None:
            batch_size = data.shape[0]

        # q_actions = self._model.predict(data, batch_size=batch_size)  # q value of each action
        q_actions = self._model(data, training=training).numpy()
        opt_policy = np.argmax(q_actions, axis=-1)
        if epsilon > 0.:
            rand_val = np.random.random(batch_size)
            opt_policy[rand_val < epsilon] = np.random.randint(0, self._action_size, size=(np.sum(rand_val < epsilon)))
        return opt_policy, q_actions[np.arange(batch_size), opt_policy], q_actions

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch, tf_writer=None, batch_size=None):

        if batch_size is None:
            batch_size = s_batch.shape[0]

        # Save the graph just the first time
        if tf_writer is not None:
            tf.summary.trace_on()

        target = self._model(s_batch, training=True).numpy()
        fut_action = self._model(s2_batch, training=True).numpy()

        if tf_writer is not None:
            with tf_writer.as_default():
                tf.summary.trace_export("model-graph", 0)
            tf.summary.trace_off()

        idx = np.arange(batch_size)
        target[idx, a_batch] = r_batch

        nd_batch = ~d_batch

        #get next action and its q-value
        next_a = np.argmax(fut_action, axis=-1)
        fut_Q = fut_action[idx, next_a]

        target[nd_batch, a_batch[nd_batch]] += self._training_param.discount_factor * fut_Q[nd_batch]

        loss = self.train_on_batch(self._model, self._optimizer_model, s_batch, target, imp_batch)

        return loss

    def train_on_batch(self, model, optimizer_model, x, y_true, imp_batch):
        """train the model on a batch of example. This can be overide"""
        loss = model.train_on_batch(x, y_true, imp_batch)
        return loss

    @staticmethod
    def get_path_model(path, name=None):
        # Get the path for the saved model
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)

        return path_model

    def save_network(self, path, name=None, ext="h5"):
        # Saves model at specified path as h5 file
        model_path = self.get_path_model(path, name)
        model_path = model_path + '.' + ext
        self._model.save(model_path)

        if self.verbose:
            print("Succesfully saved network.")

    def load_network(self, path, name=None, ext="h5"):
        # Load the network saved as an .h5 file
        model_path = self.get_path_model(path, name)
        model_path = model_path + '.' + ext

        self.construct_q_network()

        self._model.load_weights(model_path)

        if self.verbose:
            print("Succesfully loaded network.")

    def save_tensorboard(self, current_step):
        """function used to save other information to tensorboard"""
        pass
