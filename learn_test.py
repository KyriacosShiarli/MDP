from discretisationmodel import *
from Model import Model
from forwardBackward import *
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
 # ----->Changed : Transition representation, affects forward backward and plots. Anything with transition and children. 

#Aim:
#To create a set of feature expectations using a known cost funtion.
#	i.e import discretisation, create a model create a statistic (the data)
#Then change the reward function and for the same system do gradient descent in order to see if we come to
#the original cost function. The number of steps should be high.

#
#Need to test: likelihood, Convergence instead of steps. Small weight differences to make sure it works. Multiple xamples, averaging etc
# Investigate feature sensitivity of the error surphace. is it alsways the same features or does it depend on anything else?
#effect of transition

def generate_test_statistic(policy,model,start,steps):
	state_freq,dt_states = forward_alt(policy,model.transition,start,steps)
	statistic = 0
	for idx,i in enumerate(state_freq[1::]):
		statistic += i*model.feature_f[idx]
	return statistic,dt_states

def test1(transition_samples):
	def momentum(current,previous,decay):
		new = current + decay*previous
		return new
	w_init = [-1.3,-1.2,-1,-0.8,-0.8,-1.4,-1.5,-3.,-2.,-1.,-0.3,-0.5,-8.,-3.]

	#w_init /=np.linalg.norm(w_init)
	steps = 10
	diff = []
	m = DiscModel();
	model = Model(m,w_init)
	initial_transition = model.transition_f
	policy = caus_ent_backward(model.transition,model.reward_f,3,steps,conv=0.1,z_states = None)
	start_states = [400,45,65,67,87,98,12,34,54,67,54,32,34,56,80,200,100,150]
	#statistics = [generate_test_statistic(policy,model,start_state,steps) for start_state in start_states]
	statistics,dt_states_base =generate_test_statistic(policy,model,start_states,steps)

	model.w =[-1,-1.2,-1,-0.8,-0.8,-4.4,-2,-2.,-3.,-1.,-2.3,-1.5,-4.,-3.]
	#model.w =[-2.,-0.6,-4.,-4.,-3.,-5.,-2.,-0.5,-4.,-0.8,-4.,-3.,-5.]
	#model.w /=np.linalg.norm(model.w)
	model.buildRewardFunction()
	if transition_samples!=1:
		model.buildTransitionFunction(transition_samples,learn = False)
	transition_diff = np.sum(np.absolute(initial_transition - model.transition_f))
	initial_transition = 0
	gamma = 0.04
	iterations = 110
	for i in range(iterations):
		policy2 = caus_ent_backward(model.transition,model.reward_f,1,steps,conv=0.1,z_states = None)
		#gradients = np.array([(statistics[j] - generate_test_statistic(policy,model,start_state,steps)) for j,start_state in enumerate(start_states)])
		state_freq,dt_states_train = generate_test_statistic(policy2,model,start_states,steps) 
		gradients = statistics - state_freq 
		if i ==0:
			image = np.absolute(dt_states_train - dt_states_base)
			gradient = gradients
		else:
			gradient = momentum(gradients,prev,0.8)
			image = np.append(image,np.absolute(dt_states_train - dt_states_base),axis = 1)
		model.w=model.w *np.exp(-gamma*gradient)
		#model.w /=np.linalg.norm(model.w)
		prev = gradient
		gamma = gamma*1.04
	
		model.buildRewardFunction()
		print "Iteration",i
		print "Gradient",gradient
		print "New Weights",model.w
		print "Real weights",w_init
		print "Policy Difference", np.sum(np.sum(np.absolute(policy-policy2)))
		diff.append(np.sum(np.sum(np.absolute(policy-policy2))))
	policy_diff = np.sum(np.sum(np.absolute(policy-policy2)))
	w_diff = np.absolute(w_init - model.w)
	grad = np.sum(np.absolute(gradient))		
	return image, diff,grad,w_diff,transition_diff


def lbfgs(w0,model,policy,statistic,steps):
	
	model.w = w0
	model.reward_f = model.buildRewardFunction()
	p = caus_ent_backward(model.transition,model.reward_f,1,steps,conv=0.1,z_states = None)
	state_freq,dt_states_train = generate_test_statistic(p,model,3,steps) 
	dw = statistic - state_freq
	dw = -dw
	obj = np.sum(np.sum(np.absolute(p-policy)))
	print obj
	return obj,dw
	

def test2():
	w_init = [-1.3,-1.2,-1,-0.8,-0.8,-1.4,-1.5,-3.,-2.,-1.,-0.3,-0.5,-8.]
	steps = 10
	diff = []
	m = DiscModel();
	model = Model(m,w_init)
	pol= caus_ent_backward(model.transition,model.reward_f,3,steps,conv=0.1,z_states = None)
	start_states = [400,45,65,67,87,98,12,34,54,67,54,32,34,56,80,200,100,150]
	#statistics = [generate_test_statistic(policy,model,start_state,steps) for start_state in start_states]
	statistic,dt_states_base =generate_test_statistic(pol,model,start_states,steps)
	w0 = [-1.1,-1.,-1.,-0.4,-1.,-5.,-3.,-1.,-15.,-1.,-1.,-0.6,-5.]
	result = lbfgs(w0,model,pol,statistic,steps)
	bound = [[None,-0.1]]*len(w0)
	
	result = opt.minimize(lbfgs,w0,args = (model,pol,statistic,steps,),jac = True,method = "L-BFGS-B",bounds = bound)
	print result

transitions = [1,2,5,10,20]
results = []
for transition in transitions:
	 results.append(test1(transition))

max_diff = results[0][0].shape[0] * 50 *2

tf_diff = [(result[4]/max_diff) *100 for result in results]
f = plt.figure()
ax = f.add_subplot(111)
colors = ["red","green","blue"]
for i,result in enumerate(results):
	#f ,ax = plt.subplots(1)
	ax.plot(result[1],linewidth = 2,color = colors[i],label = str(np.around(tf_diff[i],2)))
	#ax.imshow(result[0])
	ax.set_ylabel("Policy Difference")
	ax.set_xlabel("Iterations")
	#f.colorbar(plt.imshow(result[0]),orientation = "horizontal")
plt.legend(bbox_to_anchor=(1., 1,0.,-0.06),loc=1)
name = "Policy difference, per transition function"
f.suptitle(name)
f.savefig(name+".png")

	

#Does the algritm work?
#Algoritm seems to work but convergence is slow if only one example is given because the error surphace becomes very skewed depending
#on the starting state of the example. I.e changes in the weights are either over-sensitive or under-sensitive. in other words the whole thing
#gets stuck. In the worst case e.g with small number of steps, a state does not get observed at all and therefore no descend takes place with 
#respect to that state. This should be able to be mitigated by having lengthy examples that start from different states.
#Making sure that all states are visited adequetly.

#Asuming enough steps, do I back up in time steps or until convergence?







