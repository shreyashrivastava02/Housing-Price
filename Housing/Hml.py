'''
Created on Jul 17, 2019

@author: shreya0008
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, accuracy_score
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np



def pre_proc(df):
    
    mv = df["total_bedrooms"].mean()
    mv
    df.fillna(0,inplace=True)
    
#     df.drop("longitude" , axis="columns" , inplace=True)
#     df.drop("latitude", axis="columns" , inplace=True)
    
    le= LabelEncoder()
    df["ocean_proximity"]=le.fit_transform(df["ocean_proximity"].astype(str))
    return df
def training(df):
    df = pre_proc(df)
#     model=XGBRegressor()
    model=RandomForestRegressor()
    sc = StandardScaler()
    
    y = df["median_house_value"]
    X = df.drop(["median_house_value"], axis="columns")
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)  
     
    model.fit(X_train, y_train)
    print(model.score(X_test,y_test))  
    
#     

    n_estimators = [int(x) for x in np.linspace(start=1, stop=50, num = 20)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 50, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 3, 5, 10, 12, 15, 20]
    min_samples_leaf = [1, 2, 4,6,8,10]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, cv = 4)
    rf_random.fit(X_train, y_train)
    print(rf_random.best_score_)
    rf_random.best_params_
# rf_random = GridSearchCV(model, random_grid, cv=5)
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(rf_random, file)
    
    print(rf_random.score(X,y))   
    
        
def pred(ob):
    d1 = ob.to_dict()
    df = pd.DataFrame(d1, index=[0])
    df.drop("median_house_value", axis="columns", inplace=True)    
    df = pre_proc(df)    
    dummyrow_filename = "dummyRow.csv"
    dummyrow_filename = os.path.dirname(__file__) + "/" + dummyrow_filename    
    df2 = pd.read_csv(dummyrow_filename)    
    for c1 in df.columns:
        df2[c1] = df[c1]
    pkl_filename = "pickle_model.pkl"
    pkl_filename = os.path.dirname(__file__) + "/" + pkl_filename
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
    pred = model.predict(df2)
    
    return pred
                        
                        
if __name__ == "__main__":
    df = pd.read_csv("housing.csv")     
    training(df)                   