import numpy as np
import math

def staticGroupSimple(distance,orientation_group,max_distance=7,):
	features = np.zeros(2)
	features[0] = math.exp(-(3-distance)**2)
	features[1] = 0.5*(math.cos(orientation_group)+1)
	return features

