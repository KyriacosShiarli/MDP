from discretisationmodel import *
from forwardBackward import *
from Model import Model
from dataload import *
import numpy as np
from plott import *
from functions import *
def processSet(train,test,model):
	#========================================================================================================
	def extractStatistics(examples,log_policy,state_frequencies):
		feature_avg = 0
		likelihood = 0
		for example in examples:
			feature_avg +=example["feature_sum"]
			for step in xrange(example["steps"]-1):
				likelihood+=max(log_policy[example["action_numbers"][step+1],example["state_numbers"][step]], 
					log_policy[example["action_numbers"][step],example["state_numbers"][step]])
		s,a,f = model.feature_f.shape
		gradient = feature_avg/len(examples) - np.dot(state_frequencies.reshape(s*a),model.feature_f.reshape(s*a,f))
		error = np.sum(np.absolute(gradient))
		likelihood/=len(examples)
		return gradient,error,likelihood
	#======================================================================================================
	steps = train[1]["steps"]
	train_start_states = [(tr["start_state"]) for tr in train]
	test_start_states = [(te["start_state"]) for te in test]
	policy,log_policy=caus_ent_backward(model.transition,model.reward_f,examples[1]["end_state"],steps,3)
	train_state_frequencies,train_sa= forward(policy,model.transition,train_start_states,steps)
	train_statistics = extractStatistics(train,log_policy,train_sa)
	test_state_frequencies,test_sa= forward(policy,model.transition,test_start_states,steps)
	test_statistics = extractStatistics(test,log_policy,test_sa)
	return train_statistics,test_statistics
def learn(mod,test,train,iterations,gamma,dist):
	test_error= []
	train_error = []
	likelihoods_train = []
	likelihoods_test = []
	for iteration in xrange(iterations):
		train_statistics,test_statistics = processSet(train,test,model)
		gradient = train_statistics[0]
		train_error.append(train_statistics[1])
		likelihoods_train.append(train_statistics[2])
		likelihoods_test.append(test_statistics[2])
		test_error.append(test_statistics[1])
		if iteration == 0: gradient = train_statistics[0]
		else: gradient = momentum(gradient,prev,0.6)
		print "THIS IS N",iteration
		print "Old W", mod.w 	
		print "Gradient", train_statistics[0]
		print "Cross Validation Difference", test_statistics[0]
		mod.w=mod.w *np.exp(-gamma*gradient)
		prev = gradient
		if iteration < 25:gamma = gamma/1.03
		else: gamma = gamma*1.03
		print "Train Likelihood", train_statistics[2]
		print "Test Likelihood",test_statistics[2]
		print "new W" , mod.w
		print "------------------------------------------------------------------------"
		mod.buildRewardFunction()
	return train_error,test_error,likelihoods_train,likelihoods_test
# initialise weights
#w = [-1,-5,-0.0291642,-9.21098972,-18.18630619,-1.47331647,-4.51922346,-7.44012957,-4.80524282,-0.06006514,-0.06949205,-1,-1.0064372]
if __name__ == "__main__":
	w = [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]
	#Initialise transition and reward model----------------------------------------------------------------------
	disc_model = DiscModel()
	#w =[-1.,-0.2,-4.,-1.,-0.3,-1.,-1.,-1.,-0.3,-2.,-1.,-0.5,-1.,-3.]
	model = Model(disc_model,w)
	#Load data ->-----------------------------------------------------------------------------------------------
	print "Loading data"
	steps = 3
	examples,distribution = extract_info(disc_model,steps,dist = True)
	#Other Settings----------------------------------------------------------------------------------------------
	iterations =50
	gamma = 0.01
	fol =6
	#Initialise data structures ---------------------------------------------------------------------------------
	grad_train = np.zeros([len(examples)/fol,iterations])
	grad_test = np.zeros([len(examples)/fol,iterations])
	lik_train = np.zeros([len(examples)/fol,iterations])
	lik_test = np.zeros([len(examples)/fol,iterations])
	for idx in xrange(0,len(examples)/fol):
		train,test = getFolds(examples,fol,idx)
		model.w = w
		model.buildRewardFunction()
		train_error,test_error,likelihoods_train,likelihoods_test = learn(model,test,train,iterations,gamma,distribution)
		name = "Fold %s" %idx
		plot_result(train_error,test_error,likelihoods_train,likelihoods_test,name)
		n1 = name+"train" ; n2 = name + "test"
		trajectoryCompare(train,steps,model,n1)
		trajectoryCompare(test,steps,model,n2)


	#f = open("data9.py","w")
	#f.write("\nsum_grad=%s"%str(grad_train))
	#f.write("\nsum_valid=%s"%str(grad_test))
	#f.write("\nlik_train=%s"%str(lik_train))
	#f.write("\nlik_test=%s"%str(lik_test))
	#f.write("\nw_final=%s"%str(mod.w))
		