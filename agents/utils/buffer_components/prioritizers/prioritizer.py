from abc import ABC, abstractmethod


class Prioritizer(ABC):
    def __init__(self, buffer):
        self.buffer = buffer

    @abstractmethod
    def prioritize(self, **kwargs) -> int:
        '''
        :param kwargs:
            Whatever you want
        :return:
            Buffer count
        '''
        pass
