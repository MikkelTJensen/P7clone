import copy

from collections import deque
from agents.utils.buffer_components.buffer import Buffer
from agents.utils.buffer_components.initializers import *

PRIORITIZER = 'simple'
DEPRIORITIZER = 'simple'
SAMPLER = 'sequence'
EXPERIENCE = 'sequence'
SEQUENCE = 16


class SimpleBuffer(Buffer):
    def __init__(self, buffer_size, deep_q, train_iter, min_observation):
        self.deep_q = deep_q

        self.buffer = deque()
        self.buffer_count = 0
        self.buffer_size = buffer_size

        self.min_observe = min_observation
        self.training_iter = train_iter

        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 0.0005

        sampler = get_sampler_class(SAMPLER)
        self.sampler = sampler(self.buffer, self.deep_q)

        prioritizer = get_prioritizer_class(PRIORITIZER)
        self.prioritizer = prioritizer(self.buffer)

        deprioritizer = get_deprioritizer_class(DEPRIORITIZER)
        self.deprioritizer = deprioritizer(self.buffer)

        self.experience = get_experience_class(EXPERIENCE)

    def add(self, cur_state, action, reward, done, next_state):
        experience = self.experience(cur_state, action, reward, done, next_state)
        experience = copy.deepcopy(experience)

        kwargs = {'experience': experience,
                  'sequence': SEQUENCE}

        if self.buffer_count < self.buffer_size:
            self.buffer_count += self.prioritizer.prioritize(kwargs)
        else:
            removed_elements_count = self.deprioritizer.deprioritize(kwargs)
            self.buffer_count -= removed_elements_count
            self.buffer_count += self.prioritizer.prioritize(kwargs)

    def sample(self, kwargs):

        sample_kwargs = {'batch_size': kwargs['batch_size'],
                         'training_step': kwargs['training_step'],
                         'min_observe': self.min_observe,
                         'buffer_count': self.buffer_count,
                         'alpha': self.alpha,
                         'beta': self.beta,
                         'epsilon': self.epsilon,
                         'training_iter': self.training_iter,
                         'sequence': SEQUENCE}

        return self.sampler.sample(sample_kwargs)

    def size(self):
        return self.buffer_count

    def clear(self):
        self.buffer.clear()
        self.buffer_count = 0
