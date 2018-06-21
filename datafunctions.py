

import pandas as pd
import numpy as np
from statistics import mean
import math



# Read training and test sets
def read(file, delim):

    data = pd.read_table(file, delimiter=delim, header=None, dtype=int)
    return data.values



# "Clean" the users' vectors by only getting common, non-zero ratings
def clean(a, u, ids, item_based=False, common_at=False):

    # Filter u for the ratings at the given movie ids
    if item_based == False:
        u = [ u[i-1] for i in ids ]
    else:
        u = [ u[i] for i in ids ]
    

    # Remove any ratings at a[i] and u[i] if one or the other = 0
    new_a, new_u = [], []

    # Record the common spots/ids
    spots = []

    for i in range(len(a)):
        if a[i] != 0 and u[i] != 0:
            new_a.append(a[i])
            new_u.append(u[i])
            spots.append(ids[i])

    if common_at == True:
        return (np.array(new_a), np.array(new_u), np.array(spots))
    else:
        return (np.array(new_a), np.array(new_u)) 



# Record index when a distinct user first appears in the test data (for faster numpy indexing)
def index(test_data):

    start = [0]
    current_user = test_data[0,0]
    
    i = 0 
    for row in test_data:
        if row[0] != current_user:
            start.append(i)
            current_user = row[0]
        i += 1
    
    start.append(len(test_data))
    
    return start



# Find averages for each user in training_data
def train_avg(training_data):

    avgs = []
    
    for row in training_data:
        row = row.tolist()                  # convert to list
        row = [e for e in row if e != 0]    # only care about non-zero ratings

        if row == []:                       # compute the mean
            avg = 3
        else:
            avg = mean(row)

        avgs.append(avg)              
    
    return np.array(avgs)


# Find averages w/ Dirichlet smoothing
def train_avg_smoothed(training_data):

    avgs = []
    all_ratings = []

    # Get user mean
    for row in training_data:
        row = row.tolist()
        row = [e for e in row if e != 0]

        if row == []: avg = 3
        else: avg = mean(row)

        all_ratings.extend(row)
        avgs.append((avg,len(row)))

    global_mean = sum(all_ratings)/len(all_ratings)

    # Apply dirichlet smoothing
    beta = 1
    avgs = [ (u_avg[1]/(beta+u_avg[1])*u_avg[0]) + (beta/(beta+u_avg[1])*global_mean) for u_avg in avgs ]

    return np.array(avgs)



# Find iuf for each movie: iuf = log(num_users/num_who_rated_m)
def get_iufs(training_data):

    num_users = 200
    iufs = [ 0 for i in range(1000) ]
    
    for i in range(1000):
        iufs[i] = np.count_nonzero(training_data[:, i])
    
    iufs = [ math.log2(num_users/(1+n)) for n in iufs ]    # add 1 to denom. to adjust for div by 0

    return np.array(iufs)
 