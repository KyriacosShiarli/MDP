import numpy as np
import math

def staticGroupSimple(max_distance,distance,orientation_group,orientation_target):
	features = np.zeros(3)
	features[0] = max_distance-distance
	features[1] = math.cos(orientation_group) + 1
	features[2] = -math.cos(orientation_target) +1
	return features

