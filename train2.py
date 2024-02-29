import torch
#torch.set_default_tensor_type('torch.cuda.FloatTensor') # 
import numpy as np
import time
#import cupy as cp
from running_mean_std import RunningMeanStd
from test import evaluate_model
from torch.utils.tensorboard import SummaryWriter
import graph_plot_train_reward
import datetime
import os 
import pandas as pd
import random
import make_adv_length_ant
import gym

class Train2:
    def __init__(self, env, test_env, env_name, n_iterations, agent, epochs, mini_batch_size, epsilon, horizon, device, gamma, lam, seed_id):
        self.env = env
        self.env_name = env_name
        self.test_env = test_env
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations

        self.start_time = 0
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))

        self.running_reward = 0
        
        #finetuning
        self.state_rms_mean, self.state_rms_var = self.agent.load_weights_f()
        
        self.device = device 
        self.gamma = gamma 
        self.lam = lam 
        self.seed_id =seed_id 
        if device == "cuda:0":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, values, log_probs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices], returns[indices], advs[indices], values[indices],\
                  log_probs[indices]

    def train(self, states, actions, advs, values, log_probs, device):

        values = np.vstack(values[:-1])
        device = "cpu"
        if device == "cpu":
            log_probs = np.vstack(log_probs) 
        else:
            log_probs = cp.vstack(log_probs)
        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        actions = np.vstack(actions)
        for epoch in range(self.epochs):
            for state, action, return_, adv, old_value, old_log_prob in self.choose_mini_batch(self.mini_batch_size,
                                                                                               states, actions, returns,
                                                                                               advs, values, log_probs):
                state = torch.Tensor(state).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                return_ = torch.Tensor(return_).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                old_value = torch.Tensor(old_value).to(self.agent.device)
                old_log_prob = torch.Tensor(old_log_prob).to(self.agent.device)

                value = self.agent.critic(state)
                # clipped_value = old_value + torch.clamp(value - old_value, -self.epsilon, self.epsilon)
                # clipped_v_loss = (clipped_value - return_).pow(2)
                # unclipped_v_loss = (value - return_).pow(2)
                # critic_loss = 0.5 * torch.max(clipped_v_loss, unclipped_v_loss).mean()
                critic_loss = self.agent.critic_loss(value, return_)

                new_log_prob = self.calculate_log_probs(self.agent.current_policy, state, action)

                ratio = (new_log_prob - old_log_prob).exp()
                actor_loss = self.compute_actor_loss(ratio, adv)

                self.agent.optimize(actor_loss, critic_loss)

        return actor_loss, critic_loss

    def step(self):
        state = self.env.reset()
        reward_list = []
        
        dt_now = datetime.datetime.now()
        old_day = dt_now.day
        old_hour = dt_now.hour
        old_minute = dt_now.minute
        old_second = dt_now.second
        time_name = str(dt_now.month) +"."+ str(dt_now.day) +"."+ str(dt_now.hour) +"."+ str(dt_now.minute) +"."+ str(dt_now.second)    
        
        dir_name = self.env_name+"_CP"+time_name
        
        os.mkdir("CheckPoint2/"+dir_name) 
        
        path = "./CheckPoint2/"+self.env_name+"_CP"+time_name+"/"+time_name+"result.txt"     
        f = open(path,"a")
        f.writelines("horizon:"+str(self.horizon)+" iteration:"+str(self.n_iterations)+" epoch:"+str(self.epochs)+" batch size:"+str(self.mini_batch_size)+" epsilon:"+str(self.epsilon)+" gamma:"+str(self.gamma)+" lam:"+str(self.lam)+" seed:"+str(self.seed_id))
        f.write("\n\n")
        
        df_adv = pd.read_csv("./Csv_for_attack/a_l_di_005.csv")
        
        for iteration in range(1, 1 + self.n_iterations):
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            
            #adv training
            
            seed=time.time()
            adv_rate=random.random()
            if adv_rate <= 1.0: # in adv rate /1.0:100% /0.5:50% /0.2:20% /
                df = df_adv.sample() # adversarial attack
                adv_list = list(df.iloc[0])
                make_adv_length_ant.set_adv_length(adv_list)          
            else:    
                make_adv_length_ant.set_adv_length([1,1,1,1,1,1,1,1,1,1,1,1])

            #adv training

            self.env = gym.make(self.env_name + "-v2")
            state = self.env.reset()            
            
            self.start_time = time.time()       

            for t in range(self.horizon):
                # self.state_rms.update(state)
                state = np.clip((state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                dist = self.agent.choose_dist(state)
                action = dist.sample().cpu().numpy()[0]
                log_prob = dist.log_prob(torch.Tensor(action))
                value = self.agent.get_value(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state
            # self.state_rms.update(next_state)
            next_state = np.clip((next_state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
            next_value = self.agent.get_value(next_state) * (1 - done)
            values.append(next_value)

            advs = self.get_gae(rewards, values, dones, self.gamma, self.lam)
            states = np.vstack(states)
            actor_loss, critic_loss = self.train(states, actions, advs, values, log_probs,"cpu")##
            # self.agent.set_weights()
            self.agent.schedule_lr()
            eval_rewards = evaluate_model(self.agent, self.test_env, self.state_rms, self.agent.action_bounds)
            reward_list.append(eval_rewards)

            if iteration%1000==0: 
                print(self.env_name,"iteration:",iteration,"reward:",int(eval_rewards))
            
            avr_reward = 0
            var_reward = 0
            tmp_i = 0
            reward_25 = 0
            reward_50 = 0
            reward_75 = 0
            if iteration > 1000:
                tmp_list = reward_list[iteration-1-1000:iteration]
                avr_reward = sum(tmp_list)/1000
                var_reward = np.std(tmp_list)
                tmp_list.sort() 
                reward_25 = np.percentile(tmp_list,25)
                reward_50 = np.percentile(tmp_list,50)
                reward_75 = np.percentile(tmp_list,75)
                        
            if iteration == 100:
                graph_plot_train_reward.plot(reward_list,iteration,time_name,self.env_name)
            if iteration == 1000:
                graph_plot_train_reward.plot(reward_list,iteration,time_name,self.env_name)  
            if iteration == 10000:
                graph_plot_train_reward.plot(reward_list,iteration,time_name,self.env_name)
                
                      
            f.writelines("iteration:"+str(iteration)+" reward:"+str(int(eval_rewards))+" avr:"+str(int(avr_reward))+" var:"+str(int(var_reward)))
            f.writelines(" 25%:"+str(int(reward_25))+" 50%:"+str(int(reward_50))+" 75%:"+str(int(reward_75)))
            f.write("\n")            

            self.state_rms.update(states)
            self.print_logs(iteration, actor_loss, critic_loss, eval_rewards,time_name) 
            
            
        dt_now = datetime.datetime.now() 
        new_day = dt_now.day
        new_hour = dt_now.hour
        new_minute = dt_now.minute
        new_second = dt_now.second
        dlt_time = [0,0,0,0]
        dlt_time[0] = new_day - old_day
        dlt_time[1] = new_hour - old_hour
        dlt_time[2] = new_minute - old_minute
        dlt_time[3] = new_second - old_second
        for i in range(1,4):
            if dlt_time[i] < 0:
                dlt_time[i-1] = dlt_time[i-1] - 1
                if i == 1:
                    dlt_time[i] = dlt_time[i] + 24
                else:                	
                    dlt_time[i] = dlt_time[i] + 60      
        f.write("\n")
        f.writelines("pasted day:"+str(dlt_time[0])+" hour:"+str(dlt_time[1])+" min:"+str(dlt_time[2])+ " second:"+str(dlt_time[3]))
           
        graph_plot_train_reward.plot(reward_list,iteration,time_name,self.env_name) 
          
        return dir_name

    @staticmethod
    #def get_gae(rewards, values, dones, gamma=0.97, lam=0.95):
    def get_gae(rewards, values, dones, gamma, lam):
        #gamma = self.gamma
        #lam = self.lam
        advs = []
        gae = 0

        dones.append(0)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
        return np.vstack(advs)

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution = model(states)
        return policy_distribution.log_prob(actions)

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        return loss

    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards,time_name): ##
        if iteration == 1:
            self.running_reward = eval_rewards
        else:
            self.running_reward = self.running_reward * 0.99 + eval_rewards * 0.01

        if iteration % 100 == 0: 
            print(f"Iter:{iteration}| "
                  f"Ep_Reward:{eval_rewards:.3f}| "
                  f"Running_reward:{self.running_reward:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
            self.agent.save_weights(iteration, self.state_rms,time_name) ##
            
            
