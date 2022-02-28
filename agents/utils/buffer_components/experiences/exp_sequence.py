from agents.utils.buffer_components.experiences.exp_importance import ImportanceExperience


class SequenceExperience(ImportanceExperience):
    def __init__(self, current_state, action, reward, done, next_state):
        ImportanceExperience.__init__(self, current_state, action, reward, done, next_state)

        self.indx = None

    def set_indx(self, indx):
        self.indx = indx
