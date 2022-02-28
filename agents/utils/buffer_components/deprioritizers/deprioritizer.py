from abc import ABC, abstractmethod


class Deprioritizer(ABC):
    def __init__(self, buffer):
        self.buffer = buffer

    @abstractmethod
    def deprioritize(self, kwargs) -> int:
        '''
        :param kwargs:
            Whatever you want
        :return:
            Buffer count
        '''
        pass
