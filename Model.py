import numpy as np
import math
import RFeatures as rf
from kinematicModels import staticGroupSimple
from Discretisation import *
import time
import scipy.spatial as spat
from dataload import *
class Model(object):
	def __init__(self,discretisation,reward_weights,kin_model= staticGroupSimple):
		self.disc = discretisation#discretisation model
		self.kinematics = kin_model
		self.w = reward_weights
		self.transition_f = self.buildTransitionFunction(50,learn=True)
		self.feature_f = self.buildFeatureFunction()
		self.reward_f = self.buildRewardFunction()
		self.transition_smart()
		self.actionSimilarity()
	def buildTransitionFunction(self,bin_samples,learn =True):
		transition_f = np.zeros([self.disc.tot_actions,self.disc.tot_states,self.disc.tot_states])
		for i in range(self.disc.tot_states):
			for j in range(self.disc.tot_actions):
				bins = self.disc.stateToBins(i)
				for k in range(bin_samples):
					assert bin_samples != 0
					if bin_samples==1:
						samp = False
					else:
						samp = True
					quantity = self.disc.binsToQuantity(bins,sample = samp)
					action = self.disc.indexToAction(j)
					next_quantity = self.kinematics(quantity,action)
					next_state = self.disc.quantityToState(next_quantity)
					transition_f[j,i,next_state] += 1
		if learn ==True:
			trans = self.learnTransitionFunction()
			transition_f =np.add( transition_f , trans)
		row_sums = np.sum(transition_f,axis=2)
		transition_f = transition_f/row_sums[:,:,np.newaxis]
		return transition_f
	def learnTransitionFunction(self):
		examples = extract_info(self.disc,"Full")
		transition = np.zeros([self.disc.tot_actions,self.disc.tot_states,self.disc.tot_states])
		for example in examples:
		#print len(example["action_numbers"])
			for n in xrange(example["steps"]-1):
				action = example["action_numbers"][n+1]
				state = example["state_numbers"][n]
				state_next = example["state_numbers"][n+1]
				transition[action,state,state_next] += 1
		return transition
	def transition_smart(self):
		tot_states = self.transition_f.shape[1]
		tot_actions =self.transition_f.shape[0]
		self.transition_backward = [{} for j in xrange(tot_states*tot_actions)]
		self.transition_forward = [{} for j in xrange(tot_states*tot_actions)]
		idx = np.transpose(np.array(np.nonzero(self.transition_f)))
		for i in idx:
			self.transition_backward[i[0] + i[1]*tot_actions][str(i[2])] = self.transition_f[i[0],i[1],i[2]]
			self.transition_forward[i[0] + i[2]*tot_actions][str(i[1])] = self.transition_f[i[0],i[1],i[2]]
	def buildFeatureFunction(self):
		tot_states = sum(self.disc.states_per_angle)
		for i in range(tot_states):
			quantity = self.disc.binsToQuantity(self.disc.stateToBins(i))
			features = self.disc.quantityToFeature(quantity) #Still hacky. Need to find a solution to his
			if i == 0: feature_f = np.zeros([tot_states,len(features)])
			feature_f[i,:] = features
		return feature_f
	def buildRewardFunction(self):
		reward_f = np.zeros(self.feature_f.shape[0])
		for n,i in enumerate(self.feature_f):	
			reward_f[n] = np.sum(self.w * i)
		return reward_f
	def actionSimilarity(self):
		all_actions = []
		#maxdiff = sum([max(self.actions["linear"]-min(actions["linear"])),max(actions["angular"]-min(actions["angular"]))])
		for i in range(self.disc.tot_actions):
			action = self.disc.indexToAction(i)
			all_actions.append(action)
		dist = spat.distance.squareform(spat.distance.pdist(all_actions,"euclidean"))
		maxx = np.amax(dist)
		s = 0.1
		t = 0.001
		k = np.exp(-dist**2/s**2)
		k[k<t]=0
		self.action_similarity = k



if __name__ == "__main__":
	f = open("data.py","w")
 	x = [3,3,3]
 	f.write("x=%s"%str(x))
	#for i in m.feature_f:
		#print i
	#print d.states_per_angle
	#print m.transition_backward
	#print m.transition_forward
	#print rew
	#print m.transition_f[8,1:30,1:30]
	#st = [1,4,0.0,0.0]
	#action = [0.5,0.44]
	#out = m.kinematics(st,action)
	#print out
