
import numpy as np
from numpy.linalg import norm

# Project modules
from datafunctions import clean



# a := active user
# u := similar user
# movies := movies a has rated


def cos_simil(a, u, movies):
    
	# Scaling denom
	s = len(a)

	# Get common, non-zero ratings for a and u
	a, u = clean(a, u, movies)

	# Compute similarity
	if norm(a) == 0 or norm(u) == 0:
		sim = 0
	else:
		sim = a.dot(u) / (norm(a) * norm(u))

	return sim * (min(len(a),s) / s)



# FOR CUSTOM ALGORITHM
# ---------------------------------------------
def adj_cos_simil(a, u, a_avg, u_avg):

	# Compute similarity
	if norm(a-a_avg) == 0 or norm(u-u_avg) == 0:
		sim = 0
	else:
		sim = (a-a_avg).dot(u-u_avg) / (norm(a-a_avg) * norm(u-u_avg))

	return sim