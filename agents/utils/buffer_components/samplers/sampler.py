from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self, buffer, deep_q):
        self.buffer = buffer
        self.deep_q = deep_q

    @abstractmethod
    def sample(self, **kwargs):
        '''
        :return:
            Batches: s_batch, a_batch, r_batch, d_batch, s2_batch, imp_batch
        '''
        pass
