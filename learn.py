from discretisationmodel import *
from forwardBackward import *
from Model import Model
from dataload import *
import numpy as np
from plott import *
from functions import *

class Results(object):
	def __init__(self,iterations):
		self.test_error= np.zeros(iterations)
		self.train_error = np.zeros(iterations)
		self.train_lik = np.zeros(iterations)
		self.test_lik = np.zeros(iterations)


class Learner(object):
	def __init__(self,model,train_good,test_good,train_bad = None,test_bad = None):
		self.model  = model
		self.train_good  = train_good
		self.train_bad  = train_bad
		self.test_good  = test_good
		self.test_bad  = test_bad
	def __call__(self,iterations,rate,moment,examples_type = "both",processing = "batch"):
		results = self.learn(iterations,gamma,moment,examples_type)
	def processSet(self,train,test):
		#========================================================================================================
		def extractStatistics(examples,log_policy,state_frequencies):
			feature_avg = 0
			likelihood = 0
			for example in examples:
				feature_avg +=example.feature_sum
				for step in xrange(example.steps-1):
					likelihood+=max(log_policy[example.action_numbers[step+1],example.state_numbers[step]], 
						log_policy[example.action_numbers[step],example.state_numbers[step]])
			s,a,f = self.model.feature_f.shape
			gradient = feature_avg/len(examples) - np.dot(state_frequencies.reshape(s*a),self.model.feature_f.reshape(s*a,f))
			test = np.dot(state_frequencies.reshape(s*a),self.model.feature_f.reshape(s*a,f))
			error = np.sum(np.absolute(gradient))
			likelihood/=len(examples)
			return gradient,error,likelihood
		#======================================================================================================
		steps = train[0].steps
		train_start_states = [(tr.start) for tr in train]
		test_start_states = [(te.start) for te in test]
		train_goals = [(tr.goal) for tr in train]

		policy,log_policy=caus_ent_backward(self.model.transition,self.model.reward_f,train_goals,steps,3)
		train_state_frequencies,train_sa= forward(policy,self.model.transition,train_start_states,steps)
		train_statistics = extractStatistics(train,log_policy,train_sa)
		test_state_frequencies,test_sa= forward(policy,self.model.transition,test_start_states,steps)
		test_statistics = extractStatistics(test,log_policy,test_sa)
		return train_statistics,test_statistics
	def learn(self,iterations,rate,moment,examples_type):
		if examples_type == "good" or examples_type == "bad":
			results = Results(iterations)
			for i in xrange(iterations):
				if examples_type == "good":
					assert self.train_good !=None ; assert self.test_good !=None
					[gradient,results.train_error[i],results.train_lik[i]],[cv_diff,results.test_error[i],results.test_lik[i]] = self.processSet(self.train_good,self.test_good)
				elif examples_type == "bad":
					assert self.train_bad !=None ; assert self.test_bad !=None
					[gradient,results.train_error[i],results.train_lik[i]],[cv_diff,results.test_error[i],results.test_lik[i]] = self.processSet(self.train_good,self.test_good)
					rate *= -1
				print "---------THIS IS N: %s---------"%i
				print "Examples are:", examples_type
				print "Train Difference: ", gradient
				print "Train Likelihood: ", results.train_lik[i]
				print "Test Likelihood: ", results.test_lik[i]
				if i != 0: gradient = momentum(gradient,prev,moment)
				self.model.w=self.model.w*np.exp(-rate*gradient) ; print "new W: " , self.model.w
				prev = gradient
				if i < 25:rate = rate/1.03
				else: rate = rate*1.03
				self.model.buildRewardFunction()
			self.results = results
		if examples_type =="both":
			#Initialise data structures
			results_g = Results(iterations)
			results_b = Results(iterations)
			#Make sure the data is there
			assert self.train_good !=None ; assert self.test_good !=None 
			assert self.train_bad !=None ; assert self.test_bad !=None
			#iterate
			for i in xrange(iterations):
				[gradient_g,results_g.train_error[i],results_g.train_lik[i]],[cv_diff,results_g.test_error[i],results_g.test_lik[i]] = self.processSet(self.train_good,self.test_good)
				[gradient_b,results_b.train_error[i],results_b.train_lik[i]],[cv_diff,results_b.test_error[i],results_b.test_lik[i]] = self.processSet(self.train_bad,self.test_bad)
				gradient = gradient_g - gradient_b
				print "---------THIS IS N:---------",i
				print "Examples are :", examples_type
				print "Train Difference: ", gradient_g
				print "Train Likelihood: ", results_g.train_lik[i]
				print "Test Likelihood: ", results_g.test_lik[i]
				if i != 0: gradient = momentum(gradient,prev,moment)
				self.model.w=self.model.w*np.exp(-rate*gradient) ; print "new W: " , self.model.w
				prev = gradient
				if i < 25:rate = rate/1.03
				else: rate = rate*1.03
				self.model.buildRewardFunction()
			self.results_g = results_g
			self.results_b = results_b



# initialise weights
#w = [-1,-5,-0.0291642,-9.21098972,-18.18630619,-1.47331647,-4.51922346,-7.44012957,-4.80524282,-0.06006514,-0.06949205,-1,-1.0064372]
if __name__ == "__main__":
	
	#Initialise transition and reward model----------------------------------------------------------------------
	disc_model = DiscModel()
	w = [-1.,-0.6,-1.8,-1.5,-1.3,-1.2,-1.,-1.,-1.,-1.,-0.7,-1.,-1.,-1,-1,-1.,-1.,-1,-1,-1.,-1.,-0.6,-1,-1,-1]
	#w = [-1.07934796, -1.22879963, -0.86149566, -0.94511367,-1, -0.91669579, -0.79592145, -1.17852044, -1.07693067, -1.16777957, -0.92090362, -0.94316211, -0.89168946, -0.97011397, -1.13973061, -1.49200411, -0.64497652, -1.03916874, -2.46957467, -0.40492803]

	#w = [-1.01214617, -1.068654,   -0.91906589, -0.96951396, -0.9715242,  -0.94190164,-1.11025316, -1.02126335,-1.02232611, -0.95117768, -0.95785595, -1.07361531]
	#w = [-1.02741917, -1.13913975, -0.86524108, -0.92597795, -0.934249,   -0.87620005,-1.2335044,  -1.05616296, -1.07439854, -0.94431928, -0.95122909, -0.92681191,-0.99206743, -1.12693225]
	#w =[-1.,-0.2,-4.,-1.,-0.3,-1.,-1.,-1.,-0.3,-2.,-1.,-0.5,-1.,-3.]
	model = Model(disc_model,w)

	#Load data ->-----------------------------------------------------------------------------------------------
	print "Loading data"
	steps = 70
	examples_good = extract_info(disc_model,steps,examples_type ="good")
	#examples_bad = extract_info(disc_model,steps,examples_type ="bad")
	#Other Settings----------------------------------------------------------------------------------------------
	iterations =10
	gamma = 0.01
	fol =1
	examples_good = examples_good[:2]
	for idx in xrange(0,len(examples_good)/fol):
		train_g,test_g = getFolds(examples_good,fol,idx)
		train_b,test_b = getFolds(examples_good,fol,idx)
		model.w = w
		model.buildRewardFunction()
		n1 = "Fold %s init" %idx
		trajectoryCompare(train_g,steps,model,n1)
		#n1 = "Fold %s init bad" %idx
		#trajectoryCompare(train_b,steps,model,n1)s
		learner = Learner(model,train_g,test_g,train_b,test_b)
		learner(iterations,gamma,0.4,examples_type= "good")
		#name = "Fold %s bad" %idx
		#plot_result(learner.results_b.train_error,learner.results_b.test_error,learner.results_b.train_lik,learner.results_b.test_lik,name)
		#n1 = name+"train" ; n2 = name + "test"
		#trajectoryCompare(train_b,steps,model,n1)	
		#trajectoryCompare(test_b,steps,model,n2)

		name = "Fold %s good" %idx
		#plot_result(learner.results_g.train_error,learner.results_g.test_error,learner.results_g.train_lik,learner.results_g.test_lik,name)
		plot_result(learner.results.train_error,learner.results.test_error,learner.results.train_lik,learner.results.test_lik,name)
		n1 = name+"train" ; n2 = name + "test"
		trajectoryCompare(train_g,steps,model,n1)
		trajectoryCompare(test_g,steps,model,n2)


	#f = open("data9.py","w")
	#f.write("\nsum_grad=%s"%str(grad_train))
	#f.write("\nsum_valid=%s"%str(grad_test))
	#f.write("\nlik_train=%s"%str(lik_train))
	#f.write("\nlik_test=%s"%str(lik_test))
	#f.write("\nw_final=%s"%str(mod.w))
	
		