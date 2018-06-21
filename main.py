

# Import project modules
import datafunctions as datafunct
import predict as pred


# Read data
training_data = datafunct.read('train.txt', '\t')
test_data = datafunct.read('test5.txt', ' ')


# Predict

# NOTE: bins := number of known ratings per user in the given test dataset

pred.predict_userbased('cosine', training_data, test_data,'test5_cosine.txt', bins=5)
# pred.predict_itembased('adj_cosine', training_data, test_data, 'test5_item.txt', bins=5)
# pred.predict_custom(training_data, test_data, 'test5_custom.txt', bins=5)