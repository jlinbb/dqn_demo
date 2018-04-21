## Q-learning Maze

整个DQN的实现分为两部分，第一部分是用纯Q-learning的表格形式训练Agent，第二部分是DQN算法，结合深度学习，用网络进行Q值更新。

### Agent环境

首先，我们需要虚拟一个环境，这里以走迷宫为例。黄色圆圈代表Agent，红色方块表示地雷区域，绿色表示终点。

![maze](https://github.com/jlinbb/jlinbb_image/blob/master/myImage/dqn/maze.png?raw=true)



[GYM](http://gym.openai.com/)是一个通用的强化学习实验环境，是OPEN AI的一个开源项目。为了了解OPEN AI的gym环境，此简易迷宫环境的代码参照了gym中游戏的API，实现了reset、step和render方法。

  

###  Q-Learning实现

纯Q-learnning方法不需要使用深度学习的神经网络，Q-learning通过每一步获取的reward不断更新每个Q(S, A)，使用e-greedy策略选取action。根据以上env环境，在q_learning.py中实现Q-learning算法。

  

#### Agent训练

有了以上的环境env和学习算法q_learning，我们就可以将两部分结合起来训练我们的Agent了。play.py中实现Agent的训练。





## DQN-Maze

### Agent环境

DQN中的Agent环境与上述Q-learning中的maze环境基本一致，增加n_features属性，用来表示replay memory中单个经验集的大小。

   

### DQN实现

在Tensorboard中可视化数据流图如下：

![](https://github.com/jlinbb/jlinbb_image/blob/master/myImage/dqn/graph.png?raw=tru)



## DQN-CartPole

本次试验用DQN实现了CartPole游戏的AI，算法是2013 NIPS版本。游戏中设定了当10轮平均reward达到200之后退出游戏，当训练到1600轮之后，Agent成功达到了200平均奖励，说明该DQN成功的估计了Q值，找到了游戏的策略。

![cartpole](https://github.com/jlinbb/jlinbb_image/blob/master/myImage/dqn/cartpole.png?raw=true)