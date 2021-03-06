import math
import numpy as np
from discretisationmodel import *
from RFeatures import *

class featureModel(object):
	def __init__(self,function,state_stuff,action_stuff):
		self.function = function
		self.state_stuff = state_stuff
		self.action_stuff = action_stuff
	def __call__(self,state,action):
		return self.function(self.state_stuff,self.action_stuff,state,action)
	#feature = {"function":binFeatures,
	#			"inputs":[np.array([-math.pi,-3*math.pi/4,-math.pi/2,-math.pi/4,0,math.pi/4,math.pi/2,3*math.pi/4]),np.array([0,0.5,1.5,2.5,3.5])]}
state_disc= [np.array([[-math.pi,-3*math.pi/4],[-3*math.pi/4,-math.pi/2],[-math.pi/2,-math.pi/4],[-math.pi/4,-0.1],[-0.1,0.1],[0.1,math.pi/4],
					[math.pi/4,math.pi/2],[math.pi/2,3*math.pi/4],[3*math.pi/4,math.pi]]),np.array([[0,0.65],[0.65,1.29],[1.29,1.51],[1.51,1.74],[1.74,1.93],[1.93,3]])]
action_disc = [np.array([[-10,-0.2],[-0.2,-0.05],[-0.05,0.05],[0.05,0.2],[0.2,0.3]]),np.array([[0,0.1],[0.1,0.15],[0.15,0.2],[0.2,0.25],[0.25,0.3]])]
	
class DiscModel(object): # Discretisation for non uniform polar discretisation
	def __init__(self,actions = {"linear" :np.array([0,0.1,0.15,0.2,0.25,0.3]),"angular" : np.arange(-0.5,0.5,0.1)},feature =  featureModel(tile_code_features,state_disc,action_disc)):
		distance = np.linspace(0,3,15) # Nine bins whatever the case
		linear = np.array([0,0.1,0.35]) # Linear velocity bins
		target = np.linspace(0,2*math.pi,8) # Targer orientation bins
		angular = np.array([-0.1,0,0.1]) # Angular velocity bins
		angle = np.linspace(-math.pi,math.pi,17)[0:16] # Angle to persons bins
		self.feature = feature
		self.actions = actions#these are in the form of a dictionary
		self.bin_info = [angle,distance]
		self.dist_bins_per_angle = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
		self.get_dims()
		self.statesPerAngle()
		self.tot_states =sum(self.states_per_angle)
		self.tot_actions = len(self.actions["linear"])*len(self.actions["angular"])

	def get_dims(self):
		self.dims = []
		for j in self.bin_info:self.dims.append( len(j))
	def statesPerAngle(self):
		#calculates number of possible states after an orientation is known for all orientiations and returns
		# a vector. This is a direct result of the fact that we have variable length discretisations at different
		# orientations in order to save s
		self.states_per_angle = [] # number of states for each orientation
		for i in range(self.dims[0]):
			self.states_per_angle.append(int(self.dist_bins_per_angle[i] * np.prod(self.dims[2::])))

	def quantityToBins(self,quantity_vector):
		assert len(quantity_vector) == len(self.bin_info)
		bins = []
		for n,k in enumerate(quantity_vector):
			if n == 1: 
				disc_vector = self.bin_info[n][:self.dist_bins_per_angle[bins[0]]]
			else: disc_vector = self.bin_info[n]
			bins.append(discretise(k,disc_vector))
		return map(int,bins)
	def binsToQuantity(self,bins,sample = False):
		quantity = []
		bins = map(int,bins)
		if sample==True:
			for n,k in enumerate(bins):
				if k != (len(self.bin_info[n]) - 1):
					quantity.append(np.random.uniform(self.bin_info[n][k],self.bin_info[n][k+1]))
					#print self.bin_info
				else:
					diff =self.bin_info[n][k] - self.bin_info[n][k-1]  
					quantity.append(np.random.uniform(self.bin_info[n][k], self.bin_info[n][k]+diff))
		else:
			for n,k in enumerate(bins):
				quantity.append(self.bin_info[n][k])
		return quantity
	def binsToState(self,bin_numbers):
		state = sum(self.states_per_angle[0:int(bin_numbers[0])])
		if bin_numbers[1]>self.dist_bins_per_angle[bin_numbers[0]]-1: print "Illegal state based on model";return None
		for i in range(1,len(bin_numbers)):
			if i !=len(bin_numbers)-1:
				state+= bin_numbers[i] * np.prod(self.dims[i+1::])
			else:
				state+=bin_numbers[i]
		return state
		#Determine where you have ones.
	def stateToBins(self,state):
		#first determine direction.
		bins = []
		for i,dim in enumerate(self.states_per_angle):
			state -=  dim 
			if state < 0:
				state = state + dim
				bins.append(i)
				break
		for i in range(2,len(self.dims)):
			bins.append(math.floor(state/np.prod(self.dims[i::])))
			state = state%np.prod(self.dims[i::])
		bins.append(state)
		if bins[1]>self.dist_bins_per_angle[bins[0]] - 1:
			print "Mistake in calculation. REview Function"
			print bins
			return None
		else:
			return bins
	def stateToQuantity(self,state):
		bins = self.stateToBins(state)
		return self.binsToQuantity(bins)
	def quantityToState(self,quantity):
		bins = self.quantityToBins(quantity)
		return self.binsToState(bins)
	def actionToIndex(self,action):
		linear_idx = discretise(action[1],self.actions["linear"])
		angular_idx = discretise(action[0],self.actions["angular"])
		return linear_idx + angular_idx*len(self.actions["linear"])
	def indexToAction(self,index):
		angular_idx = int(math.floor(index/len(self.actions["linear"])))
		linear_idx = int(index%len(self.actions["linear"]))
		return [self.actions["angular"][angular_idx],self.actions["linear"][linear_idx]]
	def quantityToFeature(self,state,action=None):
		feature = self.feature(state,action)
		return feature
