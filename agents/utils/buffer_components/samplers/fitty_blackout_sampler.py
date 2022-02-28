import random
import numpy as np

from agents.utils.buffer_components.samplers.sampler import Sampler


class FittyBlackoutSampler(Sampler):

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
        sequence_length = kwargs['sequence']

        segment_size = int((buffer_count / batch_size) * sequence_length)

        training_step = training_step - min_observe

        batch = []
        exp_list = []
        weights = []
        total_weight = 0

        alpha = alpha + (1 - alpha) * (training_step / training_iter)
        beta = beta + (1 - beta) * (training_step / training_iter)

        blackout_count = 0
        none_blackout_count = 0

        for i, experience in enumerate(self.buffer):

            current_state = [experience.current_state]
            next_state = [experience.next_state]

            exp_list.append(experience)

            *_, current_q_value = self.deep_q.predict_movement(np.array(current_state), 0, training=True)
            *_, next_q_value = self.deep_q.predict_movement(np.array(next_state), 0, training=True)
            td_error = np.mean(np.abs(current_q_value, next_q_value))
            weights.append(td_error)
            total_weight += td_error

            if i % segment_size == 0 and i != 0:

                # Use alpha for determining if weights are used.
                weights = [(weight+epsilon)**alpha/(total_weight+epsilon)**alpha for weight in weights]

                for indx, exp in enumerate(exp_list):
                    importance = ((1/buffer_count)*(1/weights[indx]))**beta
                    exp.set_importance(importance)
                    exp.set_indx(indx)

                blackout_exp_list = []
                blackout_weight_list = []
                for exp_weight_tuple in zip(exp_list, weights):
                    # checks if next state is blackout
                    if sum([1 for num in exp_weight_tuple[0].next_state if not num == float(0) and not num == float(-1)]) == 0:
                        blackout_exp_list.append(exp_weight_tuple[0])
                        blackout_weight_list.append(exp_weight_tuple[1])

                # There might be no blackouts
                #
                if blackout_exp_list and blackout_count <= none_blackout_count:
                    random_choice = random.choices(blackout_exp_list, weights=blackout_weight_list, k=1)
                    blackout_count += 1
                else:
                    random_choice = random.choices(exp_list, weights=weights, k=1)
                    none_blackout_count += 1

                rel_index = random_choice[0].indx

                for exp in exp_list[rel_index+sequence_length:rel_index+1:-1]:
                    if exp.done:
                        break
                    batch.append(exp)

                batch.append(random_choice[0])

                exp_list = []
                weights = []
                total_weight = 0

        #print(f"BATCH SIZE:{len(batch)}")

        # Maps each experience in batch in batches of states, actions, rewards and new states
        s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch, _ = list(map(np.array, list(zip(*batch))))
        return s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch
