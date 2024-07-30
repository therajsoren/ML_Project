# Microsoft Stock Price Analysis and Prediction
This project analyzes and predicts Microsoft (MSFT) stock prices using historical data. The steps include data loading, visualization, correlation analysis, and predictive modeling:

Data Loading: Load historical stock prices using pandas and inspect the dataset.

Visualization: Plot closing prices over time using matplotlib to observe trends.

Correlation Analysis: Compute and visualize a correlation matrix using seaborn to understand feature relationships.

Data Preparation: Select 'Open', 'High', and 'Low' as predictors and 'Close' as the target variable, converting them to NumPy arrays.

Train-Test Split: Split the data into training and testing sets (80-20) using train_test_split.

Model Training: Train a Decision Tree Regressor on the training data and predict test data.

Results: Display predicted closing prices in a DataFrame.

# Requirements
    Python  3.12.1
    
# Prerequisites
Ensure you have the following Python libraries installed:
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    
# Install these libraries using pip if they are not already installed:
    pip install numpy pandas matplotlib seaborn scikit-learn

# Download the dataset:
   https://finance.yahoo.com/quote/MSFT/history/?fr=sycsrp_catchall44
# Step-by-Step Guide
1. Import Libraries
Start by importing the necessary libraries:

python
Copy code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

sns.set()
plt.style.use('fivethirtyeight')
2. Load Data
Load the dataset and display the first few rows:

python
Copy code
data = pd.read_csv("MSFT.csv")
print(data.head())
3. Plot Closing Prices
Visualize the historical closing prices of Microsoft stock:

python
Copy code
plt.figure(figsize=(10, 4))
plt.title("Microsoft Stock Prices")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(pd.to_datetime(data["Date"]), data["Close"])
plt.show()
4. Compute Correlation Matrix
Calculate and display the correlation matrix, excluding the Date column:

python
Copy code
corr_matrix = data.drop(columns=['Date']).corr()
print(corr_matrix)
5. Plot Heatmap of Correlation Matrix
Create a heatmap to visualize the correlation matrix:

python
Copy code
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
6. Prepare Data for Modeling
Prepare the features and target variable, then convert them to numpy arrays:

python
Copy code
x = data[["Open", "High", "Low"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)
7. Split Data into Training and Testing Sets
Split the data into training and testing sets:

python
Copy code
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
8. Train the Model
Instantiate and train a Decision Tree Regressor model:

python
Copy code
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
9. Display Predicted Values
Create a DataFrame to display the predicted closing prices and print the first few rows:

python
Copy code
pred_data = pd.DataFrame(data={"Predicted Rate": ypred})
print(pred_data.head())
Execution
To execute the code, run each code block sequentially in a Python environment or script. Ensure that the MSFT.csv file is in the same directory as your script.

Visualizations
Line Plot: Displays the historical closing prices of Microsoft stocks.
Heatmap: Visualizes the correlation matrix of the stock price attributes.
Model
A Decision Tree Regressor is used to predict the closing prices based on the Open, High, and Low prices.

Output
The output includes:

The first few rows of the dataset.
The correlation matrix.
The predicted closing prices for the test set.
By following these steps, you can replicate the analysis and prediction model on your local machine.

write the above in these readme.md file # Microsoft Stock Price Analysis and Prediction
This project analyzes and predicts Microsoft (MSFT) stock prices using historical data. The steps include data loading, visualization, correlation analysis, and predictive modeling:

Data Loading: Load historical stock prices using pandas and inspect the dataset.

Visualization: Plot closing prices over time using matplotlib to observe trends.

Correlation Analysis: Compute and visualize a correlation matrix using seaborn to understand feature relationships.

Data Preparation: Select 'Open', 'High', and 'Low' as predictors and 'Close' as the target variable, converting them to NumPy arrays.

Train-Test Split: Split the data into training and testing sets (80-20) using train_test_split.

Model Training: Train a Decision Tree Regressor on the training data and predict test data.

Results: Display predicted closing prices in a DataFrame.

# Requirements
    Python  3.12.1
  
# Libraries
     numpy pandas matplotlib seaborn scikit-learn

# Install Libraries
     pip install numpy pandas matplotlib seaborn scikit-learn
   
# Download the dataset:
   https://finance.yahoo.com/quote/MSFT/history/?fr=sycsrp_catchall
Download the historical stock prices dataset from Yahoo Finance and save it as MSFT.csv.

Code Implementation:
    1. Import Libraries
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor
        sns.set()
        plt.style.use('fivethirtyeight')
2. Load Data
        data = pd.read_csv("MSFT.csv")
        print(data.head())
3. Plot Closing Prices
        plt.figure(figsize=(10, 4))
        plt.title("Microsoft Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.plot(pd.to_datetime(data["Date"]), data["Close"])
        plt.show()
4. Compute Correlation Matrix
        corr_matrix = data.drop(columns=['Date']).corr()
        print(corr_matrix)
5. Plot Heatmap of Correlation Matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()
6. Prepare Data for Modeling
        x = data[["Open", "High", "Low"]]
        y = data["Close"]
        x = x.to_numpy()
        y = y.to_numpy()
        y = y.reshape(-1, 1)
7. Split Data into Training and Testing Sets
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
8. Train the Model
    model = DecisionTreeRegressor()
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
9. Display Predicted Values
    pred_data = pd.DataFrame(data={"Predicted Rate": ypred})
    print(pred_data.head())
Follow these steps to replicate the analysis and prediction model on your local machine. Ensure that the MSFT.csv file is in the same directory as your script.

# Data
The analysis uses a CSV file named MSFT.csv containing Microsoft stock price data. Ensure this file is located in the same directory as your script. The CSV file should contain the following columns: Date, Open, High, Low, and Close.


