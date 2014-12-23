import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from forwardBackward import *
def plot_result(er_train,er_test,lik_train,lik_test,title):
	numeric,ax = plt.subplots(2,sharex=True)
	ax[0].plot(er_train,c = "r",label = "Train Gradient")
	ax[0].plot(er_test,c = "g",label = "Test Gradient")
	ax[0].set_ylabel("Gradient")
	ax[1].plot(lik_train,c = "r",label = "Train")
	ax[1].plot(lik_test,c = "g",label = "Test")
	plt.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
	ax[1].set_ylabel("Log(Lik)")
	ax[1].set_xlabel("Iterations")
	#numeric.suptitle(title + "Error and Likelihood")
	numeric.savefig(title + " Error and Likelihood.png")

	#x = range(progress.shape[1])
	#y = range(progress.shape[0])
	#X,Y = np.meshgrid(x,y)
	#surf = plt.figure()
	#ax2 = surf.add_subplot(111, projection='3d')
	#ax2.plot_surface(X,Y,progress,linewidth = 0,cmap = cm.coolwarm)
	#ax2.set_xlabel("Steps")
	return numeric


def trajectoryCompare(examples,steps,model,name):
	m = model.disc
	traj = np.zeros([steps,len(m.bin_info)])
	f,axarr = plt.subplots(2,sharex=True)
	x = range(steps)
	goals = [(ex.goal) for ex in examples]
	policy,log_policy= caus_ent_backward(model.transition,model.reward_f,goals,steps) 
	for example in examples:
		traj[0,:] = example.states[0]
		for i in range(1,example.steps):
			prev_state = m.quantityToState(traj[i-1,:])
			action = np.random.choice(m.tot_actions,p=policy[:,prev_state])
			keys = map(int,model.transition.backward[action + prev_state*m.tot_actions].keys())
			val = model.transition.backward[action + prev_state*m.tot_actions].values()
			next_state =np.random.choice(keys,p=val) 
			#plt.plot(policy[:,prev_state])
			#plt.scatter(action,policy[action,prev_state])
			#plt.show()
			traj[i,:] = m.stateToQuantity(next_state)
			axarr[0].scatter(i,m.stateToQuantity(example.state_numbers[i])[1],color = "red",alpha = 0.3)
			axarr[1].scatter(i,m.stateToQuantity(example.state_numbers[i])[0],color = "red",alpha = 0.3)
		axarr[0].scatter(x,traj[:,1],color = "green",label = "learned",alpha = 0.3)
		axarr[1].scatter(x,traj[:,0],color = "green",alpha = 0.3)		
	axarr[0].set_ylabel("Distance")
	axarr[1].set_ylabel("Angle")
	f.savefig(name+".png")