import numpy as np
import math

def discretise(quantity,disc_vector):
	#Returns the bin in which the quantity falls according to the discretisation vector
	typ = len(disc_vector.shape)
	if typ == 1: # 1D discretisation vector of icreasing values
		size = disc_vector.shape[0]
		for i in range(size):
			#assert disc_vector[i]<=disc_vector[i+1] # 1D Discretisation vector should be increasing
			if i == 0 and quantity < disc_vector[i]:
				return i
			elif i != (size-1) and quantity >= disc_vector[i] and quantity < disc_vector[i+1]:
				return i
			elif i == (size - 1):
				return i
	elif typ==2: #2D discretisation to be used for angles
		for i,j in enumerate(disc_vector):
			if quantity >= j[0] and quantity < j[1]:
				return i
def staticGroupSimple1(distance,orientation_group,max_distance=7):
	features = np.zeros(2)
	features[0] = math.exp(-(3-distance)**2)
	features[1] = 0.5*(math.cos(orientation_group)+1)
	return features

def staticGroupBin(state):
	distance = state[1]
	orientation = state[0]
	dist_disc = np.array([0,0.5,1.5,2.5,3.5])
	fd = [0,0,0,0,0]
	bin = discretise(distance,dist_disc)
	fd[bin] = 1
	dist_or = np.array([-math.pi,-3*math.pi/4,-math.pi/2,-math.pi/4,0,math.pi/4,math.pi/2,3*math.pi/4])
	fo = [0,0,0,0,0,0,0,0]
	bin2 = discretise(orientation,dist_or)
	fo[bin2] = 1
	return np.array(fd+fo)

def binFeatures(disctetisation,state,max_distance=7):
	dist_disc = np.array([0,0.5,1.5,2.5,3.5])
	fd = [0,0,0,0,0]
	bin = discretise(distance,dist_disc)
	fd[bin] = 1
	dist_or = np.array([-math.pi,-3*math.pi/4,-math.pi/2,-math.pi/4,0,math.pi/4,math.pi/2,3*math.pi/4])
	fo = [0,0,0,0,0,0,0,0]
	bin2 = discretise(orientation_group,dist_or)
	fo[bin2] = 1
	return feature