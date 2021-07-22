import gym
import time
import numpy as np

learning_rate = 0.1
gama = 0.95
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 25000
done = False

env=gym.make("MountainCar-v0")
env.reset()

#discretization of states:
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table=np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n] )) #q_table is of size (20,20,3)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

def act(discrete_state):
    if np.random.rand() <= epsilon:
        action = np.random.choice([0,1,2])
    else:
        action = np.argmax(q_table[discrete_state])
    return action



for episode in range(episodes):
    discrete_state = get_discrete_state(env.reset())
    done=False
    while not done:
        action = act(discrete_state)
        new_state, reward, done ,_ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if episode > 2000:
            env.render()

        if not done:
            print('we are in : {} episode '.format(episode))
            max_futur_q = np.max(q_table[new_discrete_state])  # max Q(S(t+1))
            q_table[discrete_state + (action,)] += learning_rate * ( reward + (gama * max_futur_q) - q_table[discrete_state + (action,)] )
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                print('epsilon:{}'.format(epsilon))

        elif new_state[0] >= env.goal_position:  # end game : position >= goal position
            if episode>2000:
                print('**********************************************')
                print('*********************GOAL*********************')
                print('**********************************************')
                print('*** we made it at : {} episode ***'.format(episode))
                time.sleep(2)
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

env.close()