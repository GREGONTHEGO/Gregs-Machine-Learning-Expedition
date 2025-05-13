import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from typing import Tuple, List
import sklearn.linear_model

# Download and read the data.
def read_train_data(filename: str) -> pd.DataFrame:
    '''
        read train data and return dataframe
    '''
    df = pd.read_csv(filename)
    return df
pass

def read_test_data(filename: str) -> pd.DataFrame:
    '''
        read test data and return dataframe
    '''
    df = pd.read_csv(filename)
    return df
pass


# Prepare your input data and labels
def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. 
        Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
        may use .dropna, 
    '''
    train_data = df_train.dropna().drop(columns=['y']).to_numpy()
    train_label = df_train.dropna().drop(columns=['x']).to_numpy()

    test_data = df_test.dropna().drop(columns=['y']).to_numpy()
    test_label = df_test.dropna().drop(columns=['x']).to_numpy()

    return train_data, train_label, test_data, test_label
pass

# Implement LinearRegression class
class LinearRegression:   
    def __init__(self, learning_rate=0.01, epoches=1000):        
        self.learning_rate = learning_rate
        self.iterations    = epoches
        self.W = None
        self.b = None
          
    # Function for model training         
    def fit(self, X, Y):
        n_samples, n_features = X.shape
         # weight initialization
        ### YOUR CODE HERE
        self.W = np.zeros(n_features)
        self.b = 0
        ### YOUR CODE HERE   
        
        for _ in range(self.iterations):
            error = 0
            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.W) + self.b

                update = self.learning_rate * (Y[idx] - linear_output)
                
                self.W += update * x_i
                self.b += update

        return self.W, self.b
        # data
        """
        for _ in range(self.iterations):
            # Predict on data and calculate gradients
            y_predicted = np.dot(X, self.W) + self.b
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - Y))
            db = (1/n_samples) * np.sum(y_predicted - Y)

            # Update weights and bias
            self.W += self.learning_rate * dw
            self.b += self.learning_rate * db
        """
        pass

      
    # output      
    def predict(self, X):
        y_predicted = np.dot(X, self.W) + self.b
        return y_predicted
        pass

# Calculate and print the mean square error of your prediction
def MSE(y_test, pred):
    '''
        return the mean square error corresponding to your prediction
    '''
    ### YOUR CODE HERE
    squared_errors = (y_test - pred) ** 2

    # Calculate the mean squared error
    mse = np.mean(squared_errors)

    return mse
    ### YOUR CODE HERE
    pass



if __name__ == "__main__":
   
    data_path_train   = "./train2.csv"
    data_path_test    = "./test.csv"
    df_train, df_test = read_train_data(data_path_train), read_test_data(data_path_test)


    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    r = LinearRegression(learning_rate=0.0001, epoches=10)
    r.fit(train_X, train_y)

    #print?
    print(df_train.head())
    print(df_test.head())

    # Make prediction with test set
    preds = r.predict(test_X)

    # Calculate and print the mean square error of your prediction
    mean_square_error = MSE(test_y, preds[:,0])
    print(mean_square_error) # I added this

    # plot your prediction and labels, you can save the plot and add in the report
    plt.scatter(test_X,test_y, label='data')
    plt.plot(test_X, preds)
    plt.legend()
    plt.show()

    
