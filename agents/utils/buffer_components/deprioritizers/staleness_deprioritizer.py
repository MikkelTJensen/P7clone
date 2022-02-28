from agents.utils.buffer_components.deprioritizers.sequence_deprioritizer import SequenceDeprioritizer


class StalenessDeprioritizer(SequenceDeprioritizer):
    def __init__(self, buffer):
        SequenceDeprioritizer.__init__(self, buffer)
        self.count_deprio = 0

    def deprioritize(self, kwargs) -> int:
        if self.count_deprio <= 2:
            for _ in range(256):
                self.buffer.popleft()
                self.count_deprio += 1
            return 256
        else:
            return self.avg_td_deprioritize()

    def avg_td_deprioritize(self) -> int:
        remove_count = 0
        remove_list = []

        td_error_list = [experience.td_error
                         for experience in self.buffer
                         if experience.td_error is not None]

        td_error_list_length = len(td_error_list)
        #print(td_error_list_length)

        if td_error_list_length == 0:
            return 0

        average_td_error = sum(td_error_list) / td_error_list_length

        sequence = []
        for experience in self.buffer:
            if experience.td_error is None:
                break
            sequence.append(experience)
            if experience.done:
                average_sequence_td_error = sum([experience.td_error for experience in sequence]) / len(sequence)
                if average_sequence_td_error*1.2 <= average_td_error:
                    remove_list.extend(sequence)
                sequence = []

        for experience in remove_list:
            self.buffer.remove(experience)
            remove_count += 1

        #print(remove_count)
        
        if remove_count == 0:
            for _ in range(256):
                self.buffer.popleft()
            return 256

        return remove_count


