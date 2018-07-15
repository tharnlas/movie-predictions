
## Steps to run the algorithms

Run the algorithms from /code/main.py. 

Currently, the user must manually change the following in main.py: 

- The training data (train.txt) and test data (test5.txt, test10.txt, or test20.txt) file names.
- The desired user-based or item-based algorithm. The choices are listed below: 

- **User-Based**: 'cosine' (cosine similarity), 'pearson_cc' (Pearson correlation coefficient), 'pearson_iuf' (Pearson w/ inverse user frequency), 'pearson_amp' (Pearson w/ case amplification)

- **Item-Based**: 'adj_cosine' (adjusted cosine similarity), 'pearson' (Pearson correlation coefficient) 

- The output file name for the predicted ratings.

- The # of known ratings per user in the give test dataset. 

** Example **

``` 
# Read data
training_data = datafunct.read('train.txt', '\t')
test_data = datafunct.read('test5.txt', ' ')

# Predict
pred.predict_userbased('cosine', training_data, test_data,'test5_results.txt', bins=5)
```


## Other Notes

**Training data**: train.txt

**Testing data**: test5.txt, test10.txt, test20.txt

**Brief summary**: This project mirrors the setup of the Netflix Prize competition held from 2006 to 2009. The goal of the competition was to develop an algorithm that predicts user ratings on films with an improved accuracy of 10% over Netflix’s own Cinematch algorithm. Although Netflix’s dataset consisted of well-over 100,000,000 ratings, for this project we were given a smaller training dataset of ratings made by 200 users for 1000 movies. A movie rating was an integer in the 1 to 5 range with 1 meaning “least liked” and 5 meaning “most liked.” A 0 meant that the user had not yet rated that movie. We were also given three test datasets (test5, test10,and test20), each containing data in the form of (user id, movie id, rating) for 100 users. The test datasets provided 5, 10, and 20 known ratings for each user in those files respectively. Our task was to best predict the unknown ratings in the test datasets by designing, implementing, and building intuition about various collaborative filtering algorithms. *For more information, please see the uploaded report*. 