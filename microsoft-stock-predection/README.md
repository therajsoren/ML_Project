# Microsoft Stock Price Analysis and Prediction
This project analyzes and predicts Microsoft (MSFT) stock prices using historical data. The main steps include data loading, visualization, correlation analysis, and predictive modeling. Below is a detailed explanation of each step in the implementation process:

Data Loading and Inspection:
    The historical stock price data is loaded from a CSV file using pandas.
    The first few rows of the dataset are printed to inspect the structure and ensure proper loading.
    
Data Visualization:
    The closing prices over time are plotted using matplotlib to visualize the trend and detect any noticeable patterns or anomalies.
    
Correlation Analysis:
    A correlation matrix is computed for numerical columns (excluding the 'Date' column) to understand the relationships between different stock features.
    The correlation matrix is visualized using a heatmap from the seaborn library, highlighting the strength and direction of relationships between variables.
    
Data Preparation for Modeling:
    The features 'Open', 'High', and 'Low' are selected as predictors (X), and 'Close' is chosen as the target variable (Y).
    These features and the target variable are converted to NumPy arrays for compatibility with machine learning models.
    
Train-Test Split:
    The dataset is split into training and testing sets using an 80-20 split ratio. The split is performed using scikit-learn's train_test_split function with a        fixed random seed for reproducibility.
    
Model Training and Prediction:
    A Decision Tree Regressor model from scikit-learn is used to train on the training data.
    The model makes predictions on the test data.
    
Result Visualization:
    The predicted closing prices are stored in a new DataFrame and printed for review.

# Requirements
    Python  3.12.1
  
# Libraries
     numpy pandas matplotlib seaborn scikit-learn

# Install Libraries
     pip install numpy pandas matplotlib seaborn scikit-learn
   
# Download the dataset:
   https://finance.yahoo.com/quote/MSFT/history/?fr=sycsrp_catchall
