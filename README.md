

### 1 Background

Recent years, deep reinforcement learning becomes very popular, which is expected to be an important technology in the future. Deep Q-Network is the first deep reinforcement learning algorithm invented by Google DeepMind. Their articles describe how to get the computer to learn to play the Atari 2600 video game on its own. Atari Breakout is an arcade game developed and published by Atari, Inc. The player’s mission is to make the ball fly and touch the blocks in the above of the racket, hitting and crushing them using the bouncing ball. The player can do three things: top the racket, move the racket to the left or to the right. If you fail, you will be punished, which means missing a ball will make you lose a life. Similarly, rewards are awarded if successful, and points increase when the ball crushing an obstacle. The game combines simple, instant feedback with neural network, which makes machine learns how to play "Breakout" and scores well. The algorithm needs to infer from the change of the score to judge whether the previous action was beneficial or not. It is like training pets. When the pet makes the specified action, we give it some food as a reward, making it even more convinced that as long as that action is done, it will be rewarded. This training is called Reinforcement Learning. DQN does not specify the output and the environment only gives the action of the algorithm the appropriate reward. The algorithm decides what to do and what time to make action by itself. In this project, I am trying to make some research and improvement on DQN algorithm. This report is used to summarize what I have learnt in this project.



### 2 Experiment Environment

Python 3.6

•    gym 0.9.5

•    keras 2.1.6

•    tensorflow 1.5

 

 

### 3   Q-learning

**3.1 Algorithm**

Before learning about the DQN, I firstly did some experiment on Q-learning algorithm. For any finite Markov decision process, Q-learning eventually finds an optimal policy, in the sense that the expected value of the total reward return over all successive steps, starting from the current state, is the maximum achievable [1]. Q-learning algorithm attempts to learn the value of being given state and taking a specific action. The process can be described as the figure below.

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image001.png?raw=true)

Actually, the algorithm stores the action value into a Q-table where the rows indicate the states and the columns are actions the agent can take. The algorithm can be described as the following.

**Algorithm:** 

The update of the Q-value is based on the Bellman Equation:![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image005.png?raw=True). **s** represents the current state. **a** represents the action the agent took under the state **s**. **s’** represents the state after doing action **a**. ![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image007.png?raw=True) represents the discount factor which is used to discount the future reward. It defines how much importance you want to give to the future reward.

**3.2 Maze game**

Using this algorithm, I developed a maze game to understand more about Q learning. The maze environment is very simple which is designed using the same API as **Gym**, including step(), render(), reset() functions. The maze including two red bombs and a green treasure. The agent is represented by yellow circle. When the agent goes to the bombs, it will get negative reward. When the agent goes to the treasure, it will get positive reward.

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image008.png?raw=True)

As this game is very simple which includes very few states, the agent will get best solution in about 20 episodes. After that, I print the Q-table.

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image009.png?raw=True)

In this table, the first column indicates the coordinate of the agent. The next 4 columns are the Q-value of each action, including moving up (0), moving down (1), moving right (2) and moving left (3). According to this Q table, agent is able to choose the right action.

 

### 4   DQN

**4.1 DQN (NIP version)**

| **Layer** | **Input** | **Filter size** | **Stride** | **Num filter** | **Activation** | **Output** |
| --------- | --------- | --------------- | ---------- | -------------- | -------------- | ---------- |
| Conv1     | 84x84x4   | 8x8             | 4          | 32             | ReLU           | 20x20x32   |
| Conv2     | 20x20x32  | 4x4             | 2          | 64             | ReLU           | 9x9x64     |
| Conv3     | 9x9x64    | 3x3             | 1          | 64             | ReLU           | 7x7x64     |
| fc4       | 7x7x64    |                 |            | 512            | ReLU           | 512        |
| fc5       | 512       |                 |            | 18             | Linear         | 18         |

In DQN, to solve the problem that too many states cannot store Q values in a table, the algorithm uses  to approximate , that is, using a function to approximate the Q value. This function uses a neural network to train the parameters to achieve convergence. Of course, this convergence process requires a lot of skills, such as the experience replay. The structure of the network can be summarized as the following graph. 



**Experience Replay** is a biologically inspired mechanism that uses a random sample of prior actions instead of the most recent action to proceed [3]. It stores experiences including the current state, action, reward and the next state, which is necessary training data for Q learning. It helps reduce correlation between experiences when updating DQN's parameters. It reuses past state to avoid catastrophic forgetting, which brings a good performance improvement. Also, using replay buffer can increase learning speed with mini-batch.

  

**4.2 DQN (Nature version)**

In this version, a target network is introduced to further reduce the correlation among training samples.  The whole process can be described as the following graph.

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image033.png?raw=True)



**Target Network** is used to fix parameters of the term ![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image054.png?raw=True) in loss function, which makes the training more stable. It will be updated every C steps.

  

**4.3 Gamma value of DQN**

Using this simple Cart Pole game, I tested the result of different gamma value of DQN.

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image055.gif)

 

| With gamma = 0.9                                             | With gamma = 0.95                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1400 iterations   average live time   is : 195.48   1500 iterations   average live time   is : 199.44   1600 iterations   average live time   is : 197.9   1700 iterations   average live time   is : 200.0   1800 iterations   average live time   is : 200.0   1900 iterations   average live time   is : 200.0   2000 iterations   average live time   is : 198.36 | 1800 iterations   average live time   is : 200.0   1900 iterations   average live time   is : 200.0   2000 iterations   average live time   is : 200.0   2100 iterations   average live time   is : 200.0   2200 iterations   average live time   is : 200.0   2300 iterations   average live time   is : 200.0   2400 iterations   average live time   is : 200.0 |

 

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image056.jpg?raw=True)

When the gamma is larger, the training process is more unstable. In the same time, the big gamma makes the training slower. While when looking at the result, a bigger gamma makes the reward more stable.

  

**4.4 Test on Breakout**

Using this algorithm, I have tried it on the Breakout game using gym environment. I trained it for about 8 hours, 10000 episodes. When testing this agent, it could get about approximately 6 rewards per episode, which is a very bad performance with low training efficiency. 

 

### 5   Multi-threads DQN

**5.1 Algorithm**

Using multi-threads is a method to improve the training efficiency. If different agents explore the game parallel, the network can be trained in a much shorter time which is linear in number of the parallel agents. Here I tried to realize the algorithm invented by DeepMind.



In this algorithm, replay buffer is removed because the training updating is online using the states of different threads, which has very low correlation. The gradient is required to be accumulated by each threads every t steps. And like the DQN algorithm (Nature version), the target network is updated every T steps to provide a stable loss.

**5.2 1st training on Breakout** 

After implementing this algorithm, I have tried to train the multi-threads agent. I set all origin epsilons 1.0 and make them reduce to 0.1 or 0.01 in different threads after 1 million steps and set the gamma value to 0.99. In this first try, I trained it for 25 hours, about 18 million steps using 12 threads. All cores of CPU haven been used. In order to be more efficient, I drop a convolutional layer and a fully connected layer from the origin model.

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image086.png?raw=True) ![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image087.png?raw=True)

The training process is much more efficient than using DQN (Nature version) and still get a good score as first. The agent can get a score of over 40 points in general. 

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image088.png?raw=True)

However, the model failed and could not be trained after it suffered a sudden reward explosion. It was because that the ball broke through the bricks and kept rebounding. So it hit many bricks at the top and got a very high score. This made the Q value much bigger than before and resulted the failure of the model. 

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image090.png?raw=True)

 

 

**5.3 2nd training on Breakout**

After observing the scores, I found that the bricks of different color have different scores. So I clip those scores and set all of them to 1. 

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image091.png?raw=True)

Also, as mentioned before, a smaller gamma will make the model trained a little bit faster, so I set the gamma value from the origin 0.99 to 0.95. Also, agents will care less about the future reward than before, which reduce influence of future’s huge reward. In addition, to further improve the stability of the model, I reset the epsilons. I made all threads’ epsilon start at 0.7 and end at 0.1, 0.01 or 0.3. As, the replay buffer is removed, I believe that different exploration policy can help increase the diversity of training samples. For example, when the epsilon is 0.3, the agent usually gets a lower score than the agent with 0.1 epsilon. This is because that the agent tends to do more random action when the epsilon is big. So when different agents have different exploration policy, some may often get low scores and their state often stay at the game’s beginning, while some are getting higher scores exploring more states. This operation makes the model more stable.

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image092.png?raw=True)

 

After doing these improvements, I tried a second time, training the model for about 35 hours. Its performance continued going up and did not breakdown after it got a very huge reward. 

![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image094.png?raw=True)

When testing the model, I drop the influence of epsilon and it can get a score over 60 in average. The model is tested to be stable and will not breakdown easily after 35-hour training. To get a higher score the model still needs to be trained for a longer time.

**![img](https://github.com/jlinbb/dqn_demo/blob/master/pic/clip_image095.png?raw=True)**

 



### 6   Reference

[1] Francisco S. Melo, *"Convergence of Q-learning: a simple proof"*

[2] Matiisen, Tambet. *"Demystifying Deep Reinforcement Learning | Computational Neuroscience Lab”*

[3] DeepMind. *“Playing Atari with Deep Reinforcement Learning”*

[4] DeepMind. *“Human-level control through deep reinforcement learning”*

[5] DeepMind. *“Asynchronous Methods for Deep Reinforcement Learning”*

 