
from itertools import compress

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
