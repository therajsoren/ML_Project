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
    The analysis uses a CSV file named MSFT.csv containing Microsoft stock price data. Ensure this file is located in the same directory as your script. The CSV        file should contain the following columns: Date, Open, High, Low, and Close.


