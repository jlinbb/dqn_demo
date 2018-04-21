from dqn_env import Maze
from deep_q_learning import DeepQLearning

def update():
    for episode in range(100):
        state = env.reset()
        step_count = 0
        while True:
            env.render()
            action = dqn.choose_action(state)
            state_, reward, done = env.step(action)
            step_count += 1
            dqn.store_transition(state, action, reward, state_)

            if (step_count > 200) and (step_count % 5 == 0):
                state = state_

            if done:
                print(' Round over at: {0} round, Total steps: {1} steps'.format(episode, step_count))
                break
    env.distroy()

if __name__ == '__main__':
    env = Maze()
    dqn = DeepQLearning(
        env.n_actions,
        env.n_features,
        learning_rate=0.01,
        discount=0.9,
        e_greedy=0.1,
        replace_target_iter=200,
        memory_size=2000,
        output_graph=True
    )

    env.after(100, update())
    env.mainloop()
