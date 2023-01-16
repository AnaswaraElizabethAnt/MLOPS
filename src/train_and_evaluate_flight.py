import os

import yaml

import pandas as pd

import argparse

from pkgutil import get_data

from get_data import get_data,read_params

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

import joblib

import json

import numpy as np


def eval_metrics(actual, predicted):

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    mae = mean_absolute_error(actual, predicted)

    r2 = r2_score(actual, predicted)



    return rmse, mae, r2




def train_and_evaluate(config_path):
    
    config=  read_params(config_path)

    test_data_path = config['split_data']['test_path']

    train_data_path= config["split_data"]['train_path']

    random_state= config["base"]['random_state']

    model_dir= config["model_dir"]

    # Parameters for Models
    n_estimators_rf= config["estimators"]['RandomForest']["params"]["n_estimators"]
    max_depth_rf= config["estimators"]['RandomForest']["params"]["max_depth"]

    learning_rate_xgb= config["estimators"]['XGBoost']["params"]["learning_rate"]
    n_estimators_xgb= config["estimators"]['XGBoost']["params"]["n_estimators"]
    max_depth_xgb= config["estimators"]['XGBoost']["params"]["max_depth"]


    # Train and test along with the target variable
    target = [config["base"]["target_col"]]

    train= pd.read_csv(train_data_path, sep= ",")

    test= pd.read_csv(test_data_path, sep= ",")



    train_X = train.drop (target,axis = 1)

    test_X = test.drop(target, axis = 1)



    train_y = train[target]

    test_y = test[target]




#### Fiiting model ###

    rf = RandomForestRegressor(n_estimators= n_estimators_rf, max_depth = max_depth_rf,random_state= random_state)

    rf.fit(train_X, train_y)

    xgb = XGBRegressor(n_estimators= n_estimators_xgb, max_depth = max_depth_xgb, learning_rate = learning_rate_xgb, random_state= random_state)

    xgb.fit(train_X, train_y)



##### prediction

    predicted_rf = rf.predict(test_X)

    predicted_xgb = xgb.predict(test_X)




    (rmse_rf, mae_rf, r2_rf)= eval_metrics(test_y,predicted_rf)

    (rmse_xgb, mae_xgb, r2_xgb)= eval_metrics(test_y,predicted_xgb)



    print("RMSE RF: %s",rmse_rf)

    print("MAE RF: %s",mae_rf)

    print("R2 RF: %s",r2_rf)

    print("RMSE XGB: %s",rmse_xgb)

    print("MAE XGB: %s",mae_xgb)

    print("R2 XGB: %s",r2_xgb)



    score_file = config["reports"]["scores"]

    params_file = config["reports"]["params"]



    with open(score_file,"w") as f:

        scores = {

            "rmse_rf" :rmse_rf,

            "mae_rf":mae_rf,

            "r2_rf":r2_rf,

            "rmse_xgb" :rmse_xgb,

            "mae_xgb":mae_xgb,

            "r2_xgb":r2_xgb
        }



        json.dump(scores,f,indent=4)



    with open(params_file,"w") as f:

        params = {

            "n_estimators_rf":n_estimators_rf,

            "max_depth_rf":max_depth_rf,

            "learning_rate_xgb":learning_rate_xgb,

            "n_estimators_xgb":n_estimators_xgb,

            "max_depth_xgb":max_depth_xgb
        }



        json.dump(params,f,indent=4)



    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir,"model_flight.joblib")

    joblib.dump(rf, model_path)
    joblib.dump(xgb, model_path)




if __name__=="__main__":

    args=argparse.ArgumentParser()

    args.add_argument("--config",default="params_flight.yaml")

    parsed_args=args.parse_args()

    train_and_evaluate(config_path=parsed_args.config)