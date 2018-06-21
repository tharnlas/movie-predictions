

# Libraries
import numpy as np
from numpy.linalg import norm
import math

# Project modules
import datafunctions as datafunct
import cosine, pearson, item_based



def predict_userbased(alg, training_data, test_data, output_f, bins):

	# Compute extra data needed to make any predictions
	# --------------------------------------------------------------------------------
	ind = datafunct.index(test_data)   # indices where each distinct user first appears/"starts"
	user_avgs = datafunct.train_avg(training_data)
	iufs = datafunct.get_iufs(training_data)


	# Start prediction stage
	# ---------------------------------------------------------------------------------			
	with open(output_f, 'a') as out_f:
		for n in range(len(ind)-1):

			# active user id
			a_id = test_data[ind[n], 0]                    

			# rows where a rated or didn't rate the item
			rated = test_data[ind[n]:ind[n]+bins,:]   
			unrated = test_data[ind[n]+bins:ind[n+1],:]  

			# active user vector (item ratings)
			a = rated[:,2]
			rated_movies = rated[:,1]

			for m in unrated[:,1]:
				if norm(a - np.mean(a)) < 1:
					p = np.mean(a)
				else:
					# store ids of u's who 1) rated the target movie, 2) rated 1+ movie that a has rated
					user_ids, u_id = [], 0
				
					for u in training_data:
					    if u[m-1] != 0:
					    	if np.count_nonzero(u[rated_movies-1]) != 0:
					    		user_ids.append(u_id)
					    u_id += 1

					if user_ids == []: p = np.mean(a)
					else:
						users, u_avgs = training_data[user_ids], user_avgs[user_ids]
						weights = []

						for u in user_ids:
							if alg == 'cosine':
								sim = cosine.cos_simil(a, training_data[u], rated_movies)
							elif alg == 'pearson_cc' or alg == 'pearson_amp':
								sim = pearson.pearson_cc(a, training_data[u], user_avgs[u], rated_movies) 
							elif alg == 'pearson_iuf':
								sim = pearson.pearson_iuf(a, training_data[u], user_avgs[u], rated_movies, iufs)
							
							weights.append(sim)
				
			 			# Get k nearest neighbors
						weights = np.array(weights)
						k = 40

						top_k = np.argsort(weights)[::-1][0:k]  	# top k indices
						top_weights = weights[top_k]            	# weights for top k
			        	
						if not np.any(top_weights):
							p = np.mean(a)
						else: 
							ratings_at_m = np.array([ u[m-1] for u in users[top_k]])

							# Apply case amplification if requested
							if alg == 'pearson_amp':
								top_weights = top_weights * np.absolute(top_weights)**(1.5)

							if alg == 'cosine':
								p = np.sum(top_weights * ratings_at_m) / np.sum(top_weights)
							elif alg in ['pearson_cc', 'pearson_iuf', 'pearson_amp']:
								weighted_diffs = top_weights * (ratings_at_m - u_avgs[top_k])
								p = np.mean(a) + ( np.sum(weighted_diffs) / np.sum(np.absolute(top_weights)) )

				# Extra measure used to deal with anomalies
				if p < 1: p = 1
				if p > 5: p = 5

				print(a_id, m, int(round(p)), file=out_f)

				
			

def predict_itembased(alg, training_data, test_data, output_f, bins=5):

	# !!! Check that weights list isn't empty (pred. stage)

	# Compute extra data needed to make predictions
	# ------------------------------------------------------------
	ind = datafunct.index(test_data) 
	user_avgs = datafunct.train_avg(training_data)	# original data: users x movies

	training_data = np.transpose(training_data)		# transpose: movies x users
	item_avgs = datafunct.train_avg(training_data)


	# Prediction stage
	# ------------------------------------------------------------
	for n in range(len(ind)-1):

		# active user id
		a_id = test_data[ind[n], 0]                    

		# rows where a rated and didn't rate the item
		rated = test_data[ind[n]:ind[n]+bins,:]   
		unrated = test_data[ind[n]+bins:ind[n+1],:]  

		# a's rated and unrated movies
		rated_movies = rated[:,1]
		unrated_movies = unrated[:,1]

		# 'a' vector (a's ratings)
		a = rated[:,2]

		# avgs for the rated movies
		rated_avgs = item_avgs[rated_movies-1]		

		for m1 in unrated_movies:
			weights = []
			users = np.where(training_data[m1-1] != 0)[0]		# users who rated movie m1 

			for m2 in rated_movies:
				# get similarity score b/w unrated movie m1 and rated movie m2
				if alg == 'adj_cosine':
					sim = item_based.adjusted_cos(training_data[m1-1,users], training_data[m2-1,:], users, user_avgs)
				elif alg == 'pearson':
					sim = item_based.pearson(training_data[m1-1,:], training_data[m2-1,:],(item_avgs[m1-1], item_avgs[m2-1]))
				weights.append(sim)
				
			# Only get positive weights
			weights = np.array(weights)
			pos = np.where(weights > 0)
			pos_weights = weights[pos]
			pos_a = a[pos]

			if not np.any(pos_weights):
				p = item_avgs[m1-1]		# set default to item's avg
			else:
				if alg == 'cosine':
					p = np.sum(pos_weights * pos_a) / np.sum(np.absolute(pos_weights))
				elif alg == 'pearson':
					pos_rated_avgs = rated_avgs[pos]
					weighted_diffs = pos_a - pos_rated_avgs
					p = item_avgs[m1-1] + np.sum(pos_weights * weighted_diffs) / np.sum(np.absolute(pos_weights))

			if p < 1: p = 1
			if p > 5: p = 5

			with open(output_f, 'a') as out_f:
				print(a_id, m1, int(round(p)), file=out_f)






def predict_custom(training_data, test_data, output_f, bins):

	# Compute extra data needed to make any predictions
	# --------------------------------------------------------------------------------
	ind = datafunct.index(test_data)   # indices where each distinct user first appears/"starts"
	user_avgs = datafunct.train_avg(training_data)
	iufs = datafunct.get_iufs(training_data)


	# Start prediction stage
	# ---------------------------------------------------------------------------------			
	with open(output_f, 'a') as out_f:
		for n in range(len(ind)-1):

			# active user id
			a_id = test_data[ind[n], 0]                    

			# rows where a rated or didn't rate the item
			rated = test_data[ind[n]:ind[n]+bins,:]   
			unrated = test_data[ind[n]+bins:ind[n+1],:]  

			# active user vector (item ratings)
			a = rated[:,2]
			rated_movies = rated[:,1]

			new_a = [ 0 for i in range(1000) ]
			for i in range(len(rated_movies)):
				new_a[rated_movies[i]-1] = a[i]
			new_a = np.array(new_a)

			k = 40
			for m in unrated[:,1]:
				p1, p2 = np.mean(a), np.mean(a)
				if norm(a - np.mean(a)) > 1:
					# store ids of u's who 1) rated the target movie, 2) rated 1+ movie that a has rated
					user_ids, u_id = [], 0
				
					for u in training_data:
					    if u[m-1] != 0:
					    	if np.count_nonzero(u[rated_movies-1]) != 0:
					    		user_ids.append(u_id)
					    u_id += 1

					if user_ids != []:
						users, u_avgs = training_data[user_ids], user_avgs[user_ids]
						weights1, weights2 = [], []

						for u in user_ids:
							sim1 = cosine.adj_cos_simil(new_a, training_data[u], np.mean(a), user_avgs[u])
							weights1.append(sim1)
							sim2 = pearson.pearson_iuf(a, training_data[u], user_avgs[u], rated_movies,iufs)
							weights2.append(sim2)
				
						weights1, weights2 = np.array(weights1), np.array(weights2)

						if np.any(weights1):
							# Get top k positive weights
							pos_w1 = weights1 > 0
							weights1 = weights1[pos_w1]
							users1 = users[pos_w1]
							u_avgs1 = u_avgs[pos_w1]
							top_k1 = np.argsort(weights1)[::-1][0:k]

							if np.any(top_k1):
								ratings_at_m1 = np.array([ u[m-1] for u in users1[top_k1] ])
								p1 = np.sum(weights1[top_k1] * ratings_at_m1) / np.sum(weights1[top_k1])
							
						if np.any(weights2):
							# Get top k positive weights
							pos_w2 = weights2 > 0
							weights2 = weights2[pos_w2]
							users2 = users[pos_w2]
							u_avgs2 = u_avgs[pos_w2]
							top_k2 = np.argsort(weights2)[::-1][0:k]

							if np.any(top_k2):
								ratings_at_m2 = np.array([ u[m-1] for u in users2[top_k2] ])
								weighted_diffs = weights2[top_k2] * (ratings_at_m2 - u_avgs2[top_k2])
								p2 = np.mean(a) + ( np.sum(weighted_diffs) / np.sum(np.absolute(weights2[top_k2])) )

				p = (p1 + p2)/2

				# Extra measure used to deal with anomalies
				if p < 1: p = 1
				if p > 5: p = 5

				print(a_id, m, int(round(p)), file=out_f)