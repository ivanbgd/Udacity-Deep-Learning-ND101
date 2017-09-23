import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
#print(admissions.shape)    # (400, 4)
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = np.mean(data[field]), np.std(data[field])
    data.loc[:, field] = (data[field] - mean)/std

print(data)
print('data shape:', data.shape)

# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*.9), replace=False)
data, test_data = data.loc[sample], data.drop(sample, axis=0)
#print(data)
#print(test_data)
print('data shape:', data.shape)                # (360, 7)
print('test_data shape:', test_data.shape)      # (40, 7)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
print(features)
#print(targets)
print('features shape:', features.shape)    # (360, 6)
print('targets shape:', targets.shape)      # (360, )
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']
#print(features_test)
#print(targets_test)
print('features_test shape:', features_test.shape)  # (40, 6)
print('targets_test shape:', targets_test.shape)    # (40, )


#output_frame = pd.DataFrame(data, columns=['admit', 'gre', 'gpa', 'rank'])
#print(output_frame[:100].to_string(index=True))