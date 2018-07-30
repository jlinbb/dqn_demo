from env import Maze
from q_learning import QLearning

def update():
    for episode in range(20):
        state = env.reset()
        step_count, done= 0, False
        while not done:
            env.render()
            action = RL.choose_action(str(state))
            state_, reward, done = env.step(action)
            step_count += 1
            RL.learn(str(state), action, reward, str(state_))
            state = state_
        print(' Round over at: {0} round, Total steps: {1} steps'.format(episode, step_count))


if __name__ == '__main__':
    env = Maze()
    agent = QLearning(actions=list(range(env.n_actions)))

    env.after(100, update())
    # env.mainloop()

    print('\n Q Table')
    print(agent.q_table)
    agent.q_table.to_csv('Q_Table.csv')