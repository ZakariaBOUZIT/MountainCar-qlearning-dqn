import gym
import time
import numpy as np

learning_rate = 0.1
gama = 0.95
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


for episode in range(episodes):
    discrete_state = get_discrete_state(env.reset())
    done=False

    while not done:
        action = np.argmax(q_table[discrete_state])  #choose action # argmax return the indice of the max Q
        new_state, reward, done ,_ = env.step(action)
        new_discrete_state = get_discrete_state(new_state) #get the new state
        if  0<episode<5 or episode > 3000 :
            env.render()



        if not done:
            print('we are in : {} episode '.format(episode))
            #let's calculate the Q-learning eq
            max_futur_q = np.max(q_table[new_discrete_state])  # max of Q(S(t+1))
            current_q = q_table[discrete_state + (action,)]  # Q(S(t),A(t))
            new_q= current_q + learning_rate * ( reward + (gama * max_futur_q) - current_q )
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:  # end game
            if episode>3000:
                print('**********************************************')
                print('*********************GOAL*********************')
                print('**********************************************')
                print('*** we made it at : {} episode ***'.format(episode))
                time.sleep(3)
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

env.close()