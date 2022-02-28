from abc import ABC


class Experience(ABC):
    def __init__(self, current_state, action, reward, done, next_state):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state

    def __iter__(self):
        for attribute, value in self.__dict__.items():
            yield value

    def __str__(self):
        return f'''=========================================================================\n''' + \
               f'''CURRENT STATE:\n{[f'{number:.2f}' for number in self.current_state]}\n''' + \
               f'''NEXT STATE:\n{[f'{number:.2f}' for number in self.next_state]}\n''' + \
               f'''ACTION: {self.action}\n''' + \
               f'''REWARD: {self.reward}\n''' + \
               f'''DONE: {self.done}'''

