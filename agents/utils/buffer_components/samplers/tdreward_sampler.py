import random
import numpy as np
from agents.utils.buffer_components.samplers.sampler import Sampler


class TDRewardSampler(Sampler):

    def __init__(self, buffer, deep_q):
        Sampler.__init__(self, buffer, deep_q)

    def sample(self, kwargs):

        batch_size = kwargs['batch_size']
        buffer_count = kwargs['buffer_count']
        min_observe = kwargs['min_observe']
        training_step = kwargs['training_step']
        training_iter = kwargs['training_iter']
        alpha = kwargs['alpha']
        beta = kwargs['beta']
        epsilon = kwargs['epsilon']

        segment_size = int(buffer_count / batch_size)

        training_step = training_step - min_observe

        batch = []
        exp_list = []
        weights = []
        rewards = []
        total_weight = 0

        alpha = alpha + (1 - alpha) * (training_step / training_iter)
        beta = beta + (1 - beta) * (training_step / training_iter)

        for i, experience in enumerate(self.buffer):

            current_state = [experience[0]]
            next_state = [experience[4]]

            exp_list.append(experience)
            rewards.append(experience[2])

            *_, current_q_value = self.deep_q.predict_movement(np.array(current_state), 0, training=True)
            *_, next_q_value = self.deep_q.predict_movement(np.array(next_state), 0, training=True)
            td_error = np.mean(np.abs(current_q_value, next_q_value))

            weights.append(td_error)
            total_weight += td_error

            if i % segment_size == 0 and i != 0:

                weights = [weight+reward for weight, reward in zip(weights, rewards)]

                # Use alpha for determining if weights are used.
                weights = [(weight+epsilon)**alpha/(total_weight+epsilon)**alpha for weight in weights]

                importance_list = []

                for indx, exp in enumerate(exp_list):
                    importance = ((1/buffer_count)*(1/weights[indx]))**beta
                    _tuple = (exp[0], exp[1], exp[2], exp[3], exp[4], importance)
                    importance_list.append(_tuple)

                random_choice = random.choices(importance_list, weights=weights, k=1)

                batch.append(random_choice[0])

                exp_list = []
                weights = []
                rewards = []
                total_weight = 0

        # Maps each experience in batch in batches of states, actions, rewards and new states
        s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch = list(map(np.array, list(zip(*batch))))
        return s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch
