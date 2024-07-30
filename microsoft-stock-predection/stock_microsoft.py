"""
Created on Fri Jul 26 17:17:29 2024

@author: 7rajk
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('fivethirtyeight')

# Load the data
data = pd.read_csv("MSFT.csv")
print(data.head())

# Plot the closing prices
plt.figure(figsize=(10, 4))
plt.title("Microsoft Stock Prices")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(pd.to_datetime(data["Date"]), data["Close"])
plt.show()  

# Compute the correlation matrix, excluding the Date column
corr_matrix = data.drop(columns=['Date']).corr()
print(corr_matrix)

# Plot the heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Prepare the data for modeling
x = data[["Open", "High", "Low"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

# Create a DataFrame to show the predicted values
pred_data = pd.DataFrame(data={"Predicted Rate": ypred})
print(pred_data.head())
