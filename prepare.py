import pandas as pd
import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split

    
def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=36)
    test_size2 = test_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size2, random_state=36)
    print("X_train, X_test, X_val, y_train, y_test, y_val")
    print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def telco_prep():
    
    telco = pd.read_csv("kaggle_telco.csv")
    
    #before making total charges a float, eliminate this annoying space
    telco = telco.replace(" ", 0)
    
    telco = telco.replace("No internet service", "No")
    telco = telco.replace("No phone service", "No")
    
    telco = telco.astype({'TotalCharges': np.float})
    
    #to prevent pesky spaces in column names
    telco = telco.replace(" ", "_", regex=True)
    
    return telco
    