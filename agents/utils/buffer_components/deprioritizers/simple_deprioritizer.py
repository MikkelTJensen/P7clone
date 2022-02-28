from agents.utils.buffer_components.deprioritizers.deprioritizer import Deprioritizer


# A deprioritizer that just pops the left most element
class SimpleDeprioritizer(Deprioritizer):
    def __init__(self, buffer):
        Deprioritizer.__init__(self, buffer)

    def deprioritize(self, kwargs) -> int:
        self.buffer.popleft()
        return 1
