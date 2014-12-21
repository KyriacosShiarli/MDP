import numpy as np
import math
from discretisation import *
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

def binFeatures(disctetisation,state):
	out = None
	for n,disc in enumerate(disctetisation):
		temp = np.zeros(len(disc))
		bin = discretise(state[n],disc)
		temp[bin]=1
		if out==None:out = temp
		else: out = np.append(out,temp)
	return out

def tile_code_features(state_disc,action_disc,state,action):
	out = None
	for n,disc in enumerate(state_disc):
		bin = discretise_with_overlap(state[n],disc)
		if np.sum(bin)==0:
			print "Here"
		bin/=np.sum(bin)
		if out==None:out = bin
		else: out = np.append(out,bin)
	if action_disc !=None:
		#print action_disc
		for n,disc in enumerate(action_disc):
			bin = discretise_with_overlap(action[n],disc)
			if np.sum(bin)==0:
				print "Here",bin,action[n],disc
			bin/=np.sum(bin)
			if out==None:out = bin
			else: out = np.append(out,bin)
	return out
def continouous(ignore,state,action):
	return np.array([state[0],state[1],action[0],action[1]])

if __name__ == "__main__":
	out = continouous(1,[4,3,2,1],[1,2])
	print out
