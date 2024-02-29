import gym
import os
import mujoco_py
from agent import Agent
from train2 import Train2
import requests

ENV_NAME = "Ant"

TRAIN_FLAG = True

test_env = gym.make(ENV_NAME + "-v2")
n_states = test_env.observation_space.shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
n_actions = test_env.action_space.shape[0]

n_iterations = 10000
epochs = 10

lr = 3e-6
clip_range = 0.1
mini_batch_size = 256
T = 2048
gamma = 0.975
lam = 0.95


"""clip_range = 0.2
T = 2048
lr = 3e-4
mini_batch_size = 64
gamma = 0.99"""

seed_id = 1

if __name__ == "__main__":
	device = "cpu"
	print(f"number of states:{n_states}\n"
	f"action bounds:{action_bounds}\n"
	f"number of actions:{n_actions}") 
	
	if not os.path.exists(ENV_NAME):
		os.mkdir(ENV_NAME)
		os.mkdir(ENV_NAME + "/logs")
		
	env = gym.make(ENV_NAME + "-v2")
	env.seed(seed_id)
	agent = Agent(n_states=n_states,
				n_iter=n_iterations,
				env_name=ENV_NAME,
				action_bounds=action_bounds,
				n_actions=n_actions,
				lr=lr,
				device=device)

	if TRAIN_FLAG: # train 
		trainer = Train2(env=env,
					test_env=test_env,
					env_name=ENV_NAME,
					agent=agent,
					horizon=T,
					n_iterations=n_iterations,
					epochs=epochs,
					mini_batch_size=mini_batch_size,
					epsilon=clip_range,
					device=device,
					gamma=gamma,
					lam=lam,
					seed_id=seed_id)	
		dir_name = trainer.step()
		

