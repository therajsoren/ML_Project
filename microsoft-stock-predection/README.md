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
  
# Libraries
     numpy pandas matplotlib seaborn scikit-learn

# Install Libraries
     pip install numpy pandas matplotlib seaborn scikit-learn
   
# Download the dataset:
   https://finance.yahoo.com/quote/MSFT/history/?fr=sycsrp_catchall
