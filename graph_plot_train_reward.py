import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

def plot(reward_list,n_iterations,time_name,env_name):

	fig2, ax = plt.subplots()
	
	x3 = []
	y3 = []
	
	for i in range(n_iterations):
		x3.append(i)
		y3.append(reward_list[i])

	df2 = pd.DataFrame({
	"timestep":pd.Categorical(x3),
	"reward1":pd.Categorical(y3),
	})
	
	ax.set_xlabel("iteration")
	ax.set_ylabel("reward")
	plt.plot(x3,y3)
	fig2.savefig("./CheckPoint2/"+env_name+"_CP"+time_name+"/"+time_name+"num"+str(n_iterations)+".png")
	#plt.show()
