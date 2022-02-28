import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import tensorflow as tf

from l2rpn_baselines.DeepQSimple.DeepQ_NN import DeepQ_NN


class DoublePER_NN(DeepQ_NN):

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
        target_next = self._target_model(s2_batch, training=True).numpy()

        idx = np.arange(batch_size)
        target[idx, a_batch] = r_batch
        # update the value for not done episode
        nd_batch = ~d_batch  # update with this rule only batch that did not game over
        next_a = np.argmax(fut_action, axis=-1)  # compute the future action i will take in the next state
        fut_Q = target_next[idx, next_a]  # get its Q value
        target[nd_batch, a_batch[nd_batch]] += self._training_param.discount_factor * fut_Q[nd_batch]
        loss = self.train_on_batch(self._model, self._optimizer_model, s_batch, target, imp_batch)
        return loss

    def train_on_batch(self, model, optimizer_model, x, y_true, imp_batch):
        loss = model.train_on_batch(x, y_true, sample_weight=imp_batch)
        return loss
