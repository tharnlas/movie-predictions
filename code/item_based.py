
import numpy as np
from numpy.linalg import norm
from statistics import mean
import math


# project modules
from datafunctions import clean


# m1 := users' ratings for movie 1
# m2 := 1000 users' ratings for movie 2
# i := item i 
# j := item j

def adjusted_cos(m1, m2, users, user_avgs):

	# Get common, non-zero user ratings for movies i and j
	m1, m2, common = clean(m1, m2, users, item_based=True, common_at=True)

	# Check if no users have rated both m1, m2
	if not np.any(common): 
		sim = 0
	else:
		diff_i = m1 - user_avgs[common]
		diff_j = m2 - user_avgs[common]

		# Compute similarity
		if norm(diff_i) == 0 or norm(diff_j) == 0:
			sim = 0
		else:
			sim = np.sum(diff_i * diff_j) / (norm(diff_i) * norm(diff_j))

	return sim


# 0.741 MAE
def pearson(m1, m2, item_avgs):

	diff_m1 = m1 - item_avgs[0]
	diff_m2 = m2 - item_avgs[1]

	# Compute similarity
	if norm(diff_m1) == 0 or norm(diff_m2) == 0:
		sim = 0
	else:
		sim = np.sum(diff_m1 * diff_m2) / (norm(diff_m1) * norm(diff_m2))

	return sim

