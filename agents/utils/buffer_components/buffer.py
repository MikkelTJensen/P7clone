from abc import ABC, abstractmethod


class Buffer(ABC):
    @abstractmethod
    def add(self, cur_state, action, reward, done, next_state):
        pass

    @abstractmethod
    def sample(self, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def size(self):
        pass
