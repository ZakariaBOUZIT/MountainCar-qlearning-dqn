# MountainCar

#### Problem description : 

A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

#### env :

States = (position,velocity)

Actions = 0,1,2 (drive left, do nothing , drive right)

Rewards = for each step that the car does not reach the goal located at position 0.5 the reward is -1. the goal reward is 0.


#### Agent:

Solution is based on q-learning , our problem has a continuous states so how to adress this problem ? we can think of 2 approaches :

1. discretize the States, so we can use q-learning.
2. use Deep Q-networks which combine q-learning with deep learning by using a deep neural network as an approximation for the Q-function.

____________________________________________________________________________________________________________________________
###### training :

![](https://raw.githubusercontent.com/zackq88/MountainCar-using-qlearning-/main/.train.gif)


###### after learning :

![](https://raw.githubusercontent.com/zackq88/MountainCar-using-qlearning-/main/.solution.gif)
____________________________________________________________________________________________________________________________
There's a part of learning using neural networks only, trained with saved agent games. [CartpoleNN]
