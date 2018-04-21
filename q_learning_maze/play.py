from env import Maze
from q_learning import QLearning

def update():
    for episode in range(100):
        state = env.reset()
        step_count = 0
        while True:
            env.render()
            action = RL.choose_action(str(state))
            state_, reward, done = env.step(action)
            step_count += 1
            RL.learn(str(state), action, reward, str(state_))
            state = state_
            if done:
                print(' Round over at: {0} round, Total steps: {1} steps'.format(episode, step_count))
                break
    env.distroy()

if __name__ == '__main__':
    env = Maze()
    RL = QLearning(actions=list(range(env.n_actions)))

    env.after(100, update())
    env.mainloop()

    print('\n Q Table')
    print(RL.q_table)