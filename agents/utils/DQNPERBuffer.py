from collections import deque
import numpy as np
import random
import copy


class DQNPERBuffer:
    def __init__(self, buffer_size, deep_q, train_iter, min_observation):

        self.deep_q = deep_q

        self.buffer_size = buffer_size

        self.buffer = deque()
        self.buffer_count = 0

        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 0.0005

        self.min_observe = min_observation
        self.training_iter = train_iter

    def add(self, cur_state, action, reward, done, next_state):
        if np.any(~np.isfinite(cur_state)) or np.any(~np.isfinite(next_state)) or np.any(~np.isfinite(reward)):
            # TODO proper handling of infinite values somewhere !!!!
            raise RuntimeError("Infinite value somewhere in at least one of the state")

        experience = (cur_state, action, reward, done, next_state)
        experience = copy.deepcopy(experience)

        if self.buffer_count < self.buffer_size:
            self.buffer.append(experience)
            self.buffer_count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_count

    def sample(self, batch_size, training_step):

        segment_size = int(self.buffer_count / batch_size)

        training_step = training_step - self.min_observe

        batch = []
        exp_list = []
        weights = []
        total_weight = 0
        experience_sequence_length = 20

        alpha = self.alpha + (1 - self.alpha) * (training_step / self.training_iter)
        beta = self.beta + (1 - self.beta) * (training_step / self.training_iter)

        for i, experience in enumerate(self.buffer):

            current_state = [experience[0]]
            next_state = [experience[4]]

            if experience[3]:
                print("step: ", training_step, "\ncurrent state:", current_state, "\nnext state:", next_state)
            # add sequence before blackout experiences
            #if not np.any(next_state) and not experience[3]:
            #    print("step: ", training_step, "\ncurrent state:", current_state, "\nnext state:", next_state)
                # if training_step < experience_sequence_length:
                #     batch = random.sample(self.buffer, training_step)
                # else:
                #     batch = random.sample(self.buffer, experience_sequence_length)

            exp_list.append(experience)

            *_, current_q_value = self.deep_q.predict_movement(np.array(current_state), 0, training=True)
            *_, next_q_value = self.deep_q.predict_movement(np.array(next_state), 0, training=True)
            td_error = np.mean(np.abs(current_q_value, next_q_value))
            weights.append(td_error)
            total_weight += td_error

            #training
            if i % segment_size == 0 and i != 0:

                # Use alpha for determining if weights are used.
                weights = [(weight+self.epsilon)**alpha/(total_weight+self.epsilon)**alpha for weight in weights]

                importance_list = []

                for indx, exp in enumerate(exp_list):
                    importance = ((1/self.buffer_count)*(1/weights[indx]))**beta
                    _tuple = (exp[0], exp[1], exp[2], exp[3], exp[4], importance, indx)
                    importance_list.append(_tuple)

                random_choice = random.choices(importance_list, weights=weights, k=1)

                experience_indx = random_choice[0][6]

                for i in range(experience_sequence_length,1,-1):
                    batch.append(importance_list[experience_indx-i][:6])

                batch.append(random_choice[0][:6])

                exp_list = []
                weights = []
                total_weight = 0

        # Maps each experience in batch in batches of states, actions, rewards and new states
        s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch = list(map(np.array, list(zip(*batch))))
        return s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch

    def clear(self):
        self.buffer.clear()
        self.buffer_count = 0
