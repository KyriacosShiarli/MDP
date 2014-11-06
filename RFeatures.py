import numpy as np
import math
from Discretisation import discretise

def staticGroupSimple1(distance,orientation_group,max_distance=7):
	features = np.zeros(2)
	features[0] = math.exp(-(3-distance)**2)
	features[1] = 0.5*(math.cos(orientation_group)+1)
	return features

def staticGroupBin(distance,orientation_group,max_distance=7):
	dist_disc = np.array([0,0.5,1.5,2.5,3.5])
	fd = [0,0,0,0,0]
	bin = discretise(distance,dist_disc)
	fd[bin] = 1
	dist_or = np.array([-math.pi,-3*math.pi/4,-math.pi/2,-math.pi/4,0,math.pi/4,math.pi/2,3*math.pi/4])
	fo = [0,0,0,0,0,0,0,0]
	bin2 = discretise(orientation_group,dist_or)
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