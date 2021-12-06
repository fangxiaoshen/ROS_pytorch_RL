import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
# import matplotlib.pyplot as plt

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from tensorboardX import SummaryWriter

import math, random
from environment_stage_4 import Env
import time
import rospy
from std_msgs.msg import Float32MultiArray
# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10

# env = gym.make('CartPole-v0').unwrapped
# action个数为19，observation为115。
num_state =28
num_action = 5
env=Env(num_action)
torch.manual_seed(seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        # self.fc1.weight.data.normal_(0, 0.1)

        # self.fc2 =nn.Linear(128,128)
        # self.fc2.weight.data.normal_(0,0.1)

        self.action_head = nn.Linear(100, num_action)
        # self.action_head.weight.data.normal_(0, 0.1)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x=F.relu(self.fc2(x))
        # x=F.dropout(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1= nn.Linear(num_state, 100)
        # self.fc11.weight.data.normal_(0, 0.1)

        # self.fc22 =nn.Linear(128,128)
        # self.fc22.weight.data.normal_(0,0.1)

        self.state_value = nn.Linear(100, 1)
        # self.state_value.weight.data.normal_(0,0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x=F.dropout(self.fc22(x))
        value = self.state_value(x)
        return value


class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 128

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        # self.writer = SummaryWriter('../exp')
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.action_loss= 0.
        self.value_loss =0.
        self.load_models =True
        self.load_ep =310
        # self.writer = SummaryWriter('../exp')
        # SummaryWriter 类提供了一个高级 API，用于在给定目录中创建事件文件并向其中添加摘要和事件。该类异步更新文件内容。这允许训练程序直接从训练循环调用方法将数据添加到文件中，而不会减慢训练速度。
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        # Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
        #加载模型
        if self.load_models:
            load_model1 = torch.load("/home/ffd/DRL/PPO/model/310a.pt")
            # load_model2 =torch.load("/home/ffd/QDN/model/210criter.pkl")
            self.actor_net.load_state_dict(load_model1['actor_net'])
            self.critic_net.load_state_dict(load_model1['critic_net'])
            print("load model:",str(self.load_ep))
#选择动作
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0) 
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()
#获取值函数
    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
#保存神经网络参数
    def save_param(self,e):
        state = {'actor_net':self.actor_net.state_dict(),'critic_net':self.critic_net.state_dict(), 'actor_optimizer':self.actor_optimizer.state_dict(), 'critic_optimizer':self.critic_net_optimizer,'epoch':e}
        torch.save(state,"/home/ffd/DRL/PPO/model/"+str(e)+"b.pt")
        # torch.save(self.actor_net.state_dict(),"/home/ffd/QDN/model/"+str(e)+"actor.pkl")
        # torch.save(self.critic_net.state_dict(), "/home/ffd/QDN/model/"+str(e) +"criter.pkl")
#保存训练数据（记忆库）
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

#计算损失并更新
    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.action_loss = torch.max(action_loss)
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.value_loss = torch.max(value_loss)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

    
def main():
    agent = PPO()
    rospy.init_node('turtlebot3_dqn_stage_4')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time =time.time()
    # env=Env()
    for e in range(300):
        state = env.reset()#env.reset()函数用于重置环境
        # if render: env.render()#env.render()函数用于渲染出当前的智能体以及环境的状态
        episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
        done=False
        episode_step=6000


        for t in range(episode_step):
            action, action_prob = agent.select_action(state)
            next_state, reward, done= env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            # if render: env.render()
            agent.store_transition(trans)
            state = next_state
            episode_reward_sum+=reward
            pub_get_action.publish(get_action)
            if e % 10 ==0:
                # dqn.save_model(str(e))
                agent.save_param(e)
            if t >=2500:
                rospy.loginfo("time out!")
                done =True

            if done :
                result.data =[episode_reward_sum,agent.action_loss,agent.value_loss]
                pub_result.publish(result)
                m,s =divmod(int(time.time()- start_time),60)
                h,m =divmod(m,60)
                agent.update(e)
                rospy.loginfo('Ep: %d score: %.2f memory: %d episode_step: %.2f time: %d:%02d:%02d' , e ,episode_reward_sum, agent.counter,t, h, m, s)
                break

if __name__ == '__main__':
    main()
    print("end")
