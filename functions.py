
from itertools import compress
import math
import numpy as np
from scipy.stats import norm
def getFolds(examples,k,idx):
	ran = range(0,len(examples),k)
	mask = [1]*len(examples)
	mask[ran[idx]:ran[idx]+k] = [0]*k
	test = examples[ran[idx]:ran[idx]+k]
	train = list(compress(examples,mask)) 
	return train,test
def momentum(current,previous,decay):
	new = current + decay*previous
	return new

def angle_half_to_full(inp):
	#transforms vector from half angle mode to full angle mode
	assert inp<=math.pi,"not an angle" ; assert inp>=-math.pi,"not an angle"
	if inp <0:
		inp += 2*math.pi
	return inp

def angle_full_to_half(inp):
	#transforms vector from half angle mode to full angle mode
	assert inp<=2*math.pi,"not an angle" ; assert inp>=0,"not an angle"
	if inp>math.pi:
		inp -= 2*math.pi
	return inp
def polar_to_cartesian(theta,distance):
	x = distance * np.cos(theta)
	y = distance * np.sin(theta)
	return x,y
def cartesian_to_polar(x,y):
	distance = np.sqrt(x*x + y*y)
	theta = np.array([arctan_correct(x[i],y[i]) for i in xrange(len(x))])
	return np.array(zip(theta,distance))
def c_o_m(persons_in_polar):
	#print persons_in_polar[0][:,1]
	cartesian =np.array([np.array(polar_to_cartesian(p[:,0],p[:,1])) for  p in persons_in_polar])
	car_com = np.sum(cartesian,axis = 0)/len(persons_in_polar)
	return cartesian_to_polar(car_com[0],car_com[1])
def arctan_correct(x,y):
	if x<0 and y<0:
		return np.arctan(y/x)- math.pi
	if x<0 and y>0:
		return np.arctan(y/x)+math.pi
	else:
		return np.arctan(y/x)

def subsample(sample_vector,factor,shape="ramp"):
	length = factor
	rv = norm(loc = 0., scale = 0.6)
	le = float(len(sample_vector))
	if (le/factor  - int(le/factor))<0.5 and (le/factor  - int(le/factor))!=0.0 :
		center_idx = np.arange(factor,le-factor,factor)
		new_vec = np.zeros(int(le/factor))
	elif (le/factor  - int(le/factor))==0.0:
		center_idx = np.arange(factor,le,factor)
		new_vec = np.zeros(int(le/factor))
	else:
		center_idx = np.arange(factor,le,factor)
		new_vec = np.zeros(int(le/factor)+1)
	new_vec[0]= sample_vector[0]
	for n, i in enumerate(center_idx):
		for j in range(length):
			new_vec[n+1]+=sample_vector[i-int(length/2)+j] * rv.pdf(-int(length/2)+j)
	return new_vec

if __name__ == "__main__":
	#test =  np.arange(-math.pi,math.pi,0.1)


	#out = map (angle_half_to_full,test)
	#out2 = map (angle_full_to_half,out)
	#print test - out2
	for i in range(100,500):
		print i
		vector = np.arange(0,i,1)
		out = subsample(vector,10)
		print out
