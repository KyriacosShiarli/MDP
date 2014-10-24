import numpy as np
import math
import string
from Discretisation import *
from RFeatures import *
def loadFile():
	f = open("training_samples.txt",'r')
	examples = []
	while True:
		line = f.readline()
		if line == '':
			break
		ex = {"states":[],
			  "actions":[]		
			}
		x = line.split("]]")

		for i in x[0:-1]:
			y = i.replace(' ','')
			y = i.replace('[[','').replace('[','').replace(']','').split(',')[1::]
			y = map(float,y)
			ex["states"].append(y[0:4])
			ex["actions"].append(y[4:6])
		examples.append(ex)
	return examples
def loadFile2():
	f = open("data/training_samples.txt",'r')
	examples = []
	while True:
		line =f.readline()
		if line == '':
			break
		ex = {
			"states":[],
			"actions":[]
			}
		for i in line.replace(' ','').split('||')[1:]:
			al = i.split('|')
			ex["states"].append(map(float,al[0].split(',')))
			ex["actions"].append(map(float,al[1].split(',')))
		examples.append(ex)
	return examples
	
def extract_info(disc_model):

	examples = loadFile2()
	for example in examples:
		example["feature_sum"] = 0
		example["steps"] = len(example["states"])-2 
		for i,state in enumerate(example["states"][1:]):
			bins = disc_model.quantityToBins(state)
			if i == 1:
				example["start_state"] = disc_model.binsToState(bins)
			if i == len(example["states"])-2:
				example["end_state"] = disc_model.binsToState(bins)
				
			discretised = disc_model.binsToQuantity(bins)
			
			example["feature_sum"] += staticGroupSimple(discretised[1],discretised[0])
	return examples
if __name__ =="__main__":
		
			
	m = DiscModel()
	examples = extract_info(m)
	print examples[0]["states"]

