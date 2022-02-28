from agents.utils.buffer_components.prioritizers.prioritizer import Prioritizer


# A prioritizer that always adds an experience to the buffer
class SimplePrioritizer(Prioritizer):
    def __init__(self, buffer):
        Prioritizer.__init__(self, buffer)

    def prioritize(self, kwargs) -> int:
        self.buffer.append(kwargs['experience'])
        return 1
