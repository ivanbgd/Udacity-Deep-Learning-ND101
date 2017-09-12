# gapminder1.py

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Assign the dataframe to this variable
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
X = bmi_life_data[['BMI']]
y = bmi_life_data[['Life expectancy']]

# Make and fit the linear regression model
bmi_life_model = LinearRegression()
bmi_life_model.fit(X, y)

# Make a prediction using the model
LaosBMI = 21.07931
laos_life_exp = bmi_life_model.predict(np.array([LaosBMI]).reshape(-1, 1))
print('Laos life expectancy:', laos_life_exp)   # 60.31
