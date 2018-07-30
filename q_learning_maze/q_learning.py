import numpy as np
import pandas as pd

class QLearning:
    def __init__(self, actions, learning_rate=0.01, discount=0.9, e_greedy=0.01):
        self.actions = actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = e_greedy
        # print(self.q_table)


    def add_state(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state
            ))  # use state as the name of df's index


    def choose_action(self, state):
        self.add_state(state)
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            """
            if the state in the table, pick the action_value of the state
            e.g. state = 2, state_action = df.loc[2, :] = [1,2,3,4,5]
            permutation = [2,3,1,0,4] -> state_action = [3, 4, 2, 1, 5]
            action = state_action.idxmax() = 4
            """
            state_action = self.q_table.loc[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # shuffle
            action = state_action.idxmax()
        return action


    def learn(self, s, a, r, s_):
        self.add_state(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)


