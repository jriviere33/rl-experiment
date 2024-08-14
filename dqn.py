#!/usr/bin/env python
import numpy as np, gym, sys, copy, argparse
import os
import torch
import collections
import tqdm
import matplotlib.pyplot as plt
import random

class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs.float()))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr, logdir=None):
    # Define your network architecture here. It is also a good idea to define any training operations
    # and optimizers here, initialize your variables, or alternately compile your model here.

        self.model = FullyConnectedModel(4, 2)  #Init the model
        self.lr = lr  #Init the learning rate
        self.env = env  #Init the environment
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  #Init the Adam optimizer
        self.criterion = torch.nn.MSELoss()



    def save_model_weights(self, suffix):
    # Helper function to save your model / weights.
        path = os.path.join(self.logdir, "model")
        torch.save(self.model.state_dict(), model_file)
        return path

    def load_model(self, model_file):
    # Helper function to load an existing model.
        return self.model.load_state_dict(torch.load(model_file))

    def load_model_weights(self,weight_file):
    # Optional Helper function to load model weights.
        pass


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

    # Hint: you might find this useful:
    # 		collections.deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = collections.deque(maxlen=self.memory_size)

    def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
        return random.sample(self.buffer, batch_size)

    def append(self, transition):
    # Appends transition to the memory.
        self.buffer.append(transition)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        lr = 0.0005
        self.env = gym.make(environment_name)
        self.Q_w = QNetwork(self.env, lr)
        self.Q_target = QNetwork(self.env, lr)
        self.memory = Replay_Memory()

        self.episodes = 200
        self.epsilon = 0.05
        self.gamma = 0.99

        self.test_episodes = 20
        self.train_rewards = np.array([])
        self.test_rewards = np.array([])

        self.train_loss = []
        


    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.rand(1)[0] < self.epsilon:
            return self.env.action_space.sample()
        else:
            return torch.argmax(q_values).item()

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return torch.argmax(q_values).item()

    def train(self):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        c = 0
        count = 0
        for i in range(self.episodes):
            print(f'episode: {i}')
            state = self.env.reset()
            terminated = False
            full_reward = 0
            train_loss = 0
            inner = 0
            count +=1
            while not terminated:

                inner += 1

                with torch.no_grad():
                    q_w_values = self.Q_w.model(torch.Tensor(state))
                action = self.epsilon_greedy_policy(q_w_values)

                step = self.env.step(action)
                next_state = step[0]
                reward = step[1]
                terminated = step[2]

                full_reward += reward

                trans = (state, action, reward, next_state, terminated)
                self.memory.append(trans)
                mini_batch = self.memory.sample_batch()
                trans_batch = np.array(mini_batch).T
                

                batch_states = torch.Tensor(trans_batch[0].tolist())
                batch_actions = torch.Tensor(trans_batch[1].tolist())
                batch_rewards = torch.Tensor(trans_batch[2].tolist())
                batch_next_states = torch.Tensor(trans_batch[3].tolist())
                batch_terminated = torch.Tensor(trans_batch[4].tolist())

                terminated_loc = torch.where(batch_terminated == True)[0]

                
                
                q_target_values = self.Q_target.model(batch_states)
                q_target_values_next = self.Q_target.model(batch_next_states)
                
                q_target_values[np.arange(len(q_target_values)), batch_actions.tolist()]=batch_rewards+self.gamma*torch.max(q_target_values_next, axis=1).values
                q_target_values[terminated_loc.tolist(), batch_actions[terminated_loc.tolist()].tolist()]=batch_rewards[terminated_loc.tolist()]



                self.Q_w.optimizer.zero_grad()
                loss = self.Q_w.criterion(self.Q_w.model(batch_states), torch.autograd.Variable(q_target_values))
                train_loss += loss
                loss.backward()
                self.Q_w.optimizer.step()

                c+=1
                if c % 50 == 0:
                    self.Q_target.model = copy.deepcopy(self.Q_w.model)
                
                state = next_state

            #if i % int(self.episodes/3) == 0:
            #  	test_video(self, self.env, i)

            if count % 10 == 0:
                self.test()



            print(inner)
            self.train_rewards = np.append(self.train_rewards, full_reward)
            self.train_loss.append(train_loss)

                    

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 20 episodes, by calculating average cumulative rewards (returns) for the 20 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using replay memory.
        for i in range(self.test_episodes):
            full_test = []
            print(f'test episode: {i}')

            state = self.env.reset()
            terminated = False
            full_test_reward = 0
            test_inner = 0

            while not terminated:
                test_inner += 1
                with torch.no_grad():
                    q_w_values = self.Q_w.model(torch.Tensor(state))
                action = self.greedy_policy(q_w_values)

                step = self.env.step(action)
                next_state = step[0]
                reward = step[1]
                terminated = step[2]

                full_test_reward += reward

                state = next_state
            full_test.append(full_test_reward)
            print(f'Test inner: {test_inner}')
        
        self.test_rewards = np.append(self.test_rewards, np.mean(full_test))



    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        state = self.env.reset()
        while len(self.memory.buffer) < 10000:

            action = self.env.action_space.sample()
            step = self.env.step(action)
            next_state = step[0]
            reward = step[1]
            terminated = step[2]

            trans = (state, action, reward, next_state, terminated)
            self.memory.buffer.append(trans)

            if terminated:
                state = self.env.reset()
            
            else:
                state = next_state





# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop.
def test_video(agent, env, epi):
    # Usage:
    # 	you can pass the arguments within agent.train() as:
    # 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    environment_name = args.env

    d_mat = np.array([])
    train_mat = np.array([])

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    for i in range(5):
        print(f'      Trial {i}     ')
        agent = DQN_Agent('CartPole-v0')
        agent.burn_in_memory()
        print(len(agent.memory.buffer))
        agent.train()
        d_mat = np.append(d_mat, agent.test_rewards)
        train_mat = np.append(train_mat, agent.train_rewards)

    d_mat.shape = (5, 20)

    avs = np.mean(d_mat, axis=0)
    maxs = np.max(d_mat, axis=0)
    mins = np.min(d_mat, axis=0)
    ks = np.arange(20)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    
    plt.title('DQN Test Rewards', fontsize = 24)
    plt.savefig('./plots/DQN_Test_11')



    train_mat.shape = (5, 200)

    train_avs = np.mean(train_mat, axis=0)
    train_maxs = np.max(train_mat, axis=0)
    train_mins = np.min(train_mat, axis=0)
    train_ks = np.arange(200)

    plt.fill_between(train_ks, train_mins, train_maxs, alpha=0.1)
    plt.plot(train_ks, train_avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    
    plt.title('DQN Train Rewards', fontsize = 24)
    plt.savefig('./plots/DQN_Train_10')



if __name__ == '__main__':
    main(sys.argv)
