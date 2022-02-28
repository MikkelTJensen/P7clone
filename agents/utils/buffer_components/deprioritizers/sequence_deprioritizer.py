from agents.utils.buffer_components.deprioritizers.deprioritizer import Deprioritizer


class SequenceDeprioritizer(Deprioritizer):
    def __init__(self, buffer):
        Deprioritizer.__init__(self, buffer)

    def deprioritize(self, kwargs) -> int:
        sequence_length = kwargs['sequence']
        return self.sequential_deprioritize(sequence_length)

    def sequential_deprioritize(self, sequence_length):
        count = 0
        removed_count = 0

        remove_list = []

        for experience in reversed(self.buffer):
            if experience.done:
                count = 0
            count += 1
            if count > sequence_length:
                remove_list.append(experience)

        for experience in remove_list:
            removed_count += 1
            self.buffer.remove(experience)

        return removed_count
