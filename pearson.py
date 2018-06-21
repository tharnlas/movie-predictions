

import numpy as np
from numpy.linalg import norm
from statistics import mean

# Project modules
from datafunctions import clean


# NOTE:
# a := active user
# u := similar user
# movies := movies a has rated



# Calculate pearson correlation coefficient
# -----------------------------------------------------------------------------
def pearson_cc(a, u, u_avg, movies):

    # Scaling denom:
    s = len(a)
    
    # Compute similarity
    a_avg = np.mean(a)

    # Get common, non-zero ratings for a and u
    a, u = clean(a, u, movies)

    # Compute similarity
    if norm(a - a_avg) == 0 or norm(u - u_avg) == 0:
        sim = 0
    else:
        sim = (a - a_avg).dot(u - u_avg) / (norm(a - a_avg) * norm(u - u_avg))    

    # Scale the similarities if only a,u only have a few rated movies in common
    return sim * (min(len(a),s) / s)



# Calculate pearson correl. coeff. w/ iuf
# -----------------------------------------------------------------------------
def pearson_iuf(a, u, u_avg, movies, iufs):
        
    # Scaling denom:
    s = len(a)
    
    # a's avg
    a_avg = np.mean(a)

    # Get common, non-zero ratings for a and u
    # Also get the common item ids
    a, u, common = clean(a, u, movies, common_at=True)

    # Get iufs at common movies (-1 for indexing)
    # Note: don't need to check if common == [], b/c already pre-filtered
    iufs = iufs[common-1]

    # Compute weight
    if norm(a - a_avg) == 0 or norm(u - u_avg) == 0:
        sim = 0
    else:
        numer = np.sum(iufs * (a - a_avg) * (u - u_avg))
        denom = np.sqrt(np.sum(iufs * (a-a_avg)**2) * np.sum(iufs * (u-u_avg)**2))

        sim = numer/denom

    # Scale the similarities if only a,u only have a few rated movies in common
    return sim * (min(len(a),s) / s) 

