from agents.utils.buffer_components.experiences.exp_sequence import SequenceExperience


class StalenessExperience(SequenceExperience):
    def __init__(self, current_state, action, reward, done, next_state):
        SequenceExperience.__init__(self, current_state, action, reward, done, next_state)

        self.use = 0
        self.staleness = 0
        self.age = 0
        self.td_error = None

    def increase_staleness(self):
        self.staleness += 1

    def clear_staleness(self):
        self.staleness = 0

    def increase_use(self):
        self.use += 1

    def clear_use(self):
        self.use = 0

    def birthday(self):
        self.age += 1

    def rebirth(self):
        self.age = 0

    def set_td_error(self, td_error):
        self.td_error = td_error

    def clear_td_error(self):
        self.td_error = 0
