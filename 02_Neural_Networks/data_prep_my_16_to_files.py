import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standardize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:, field] = (data[field] - mean)/std

# Split off random 10% of the data for testing
np.random.seed(21)
sample = np.random.choice(data.index, size=int(.9*len(data)), replace=False)
data, test_data = data.loc[sample], data.drop(sample, axis=0)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

#with open('.\\features_full.txt', 'w') as outfile:
#    features.to_string(outfile)
features.to_csv('features.txt', header=None, index=None, sep=' ')
targets.to_csv('targets.txt', header=None, index=None, sep=' ')
features_test.to_csv('features_test.txt', header=None, index=None, sep=' ')
targets_test.to_csv('targets_test.txt', header=None, index=None, sep=' ')


