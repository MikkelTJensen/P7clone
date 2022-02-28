from agents.utils.buffer_components.experiences.experience import Experience


class ImportanceExperience(Experience):
    def __init__(self, current_state, action, reward, done, next_state):
        Experience.__init__(self, current_state, action, reward, done, next_state)

        self.importance = None

    def set_importance(self, importance):
        self.importance = importance
