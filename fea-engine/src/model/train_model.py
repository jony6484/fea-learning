# import pandas as pd
# import numpy as np
import os
# import pickle as pk
from sklearn.ensemble import RandomForestRegressor
import json
from src.auxiliary import data_handle





def fit_model(model, X, y):
    if isinstance(model, RandomForestRegressor):
        model.fit(X, y)
        return model


def main():
    
    # hyper parameters
    
    model_params = json.load(open(os.path.join(MODELDIR,'model_params.json')))
    data_path =  os.path.join(DATADIR, "fea_tests_2_5_bar_15_08_2021.csv")
    raw_data = data_handle.read_data(data_path)
    raw_data_X, raw_data_y = data_handle.raw_data_to_X_y(raw_data)
    df_coef = data_handle.preprocess_x_coef(raw_data_X, model_params)
    processed_data = data_handle.make_mX(df_coef, model_params)
    X = processed_data['mX']
    mu_coef = processed_data['mu_coef']
    s_coef = processed_data['s_coef']
    y, mu_y, s_y = data_handle.preprocess_y(raw_data_y, model_params)
    model_params = {**model_params, **dict(zip(['mu_coef', 's_coef','mu_y', 's_y'],[mu_coef, s_coef, mu_y, s_y]))}
    model = RandomForestRegressor(n_estimators=120, max_depth=120)
    fitted_model = fit_model(model, X, y)
    model_path = os.path.join(MODELDIR, "model_and_params.pk")
    data_handle.save_model_to_file([fitted_model, model_params], model_path)
    

    



if __name__ == '__main__':
    MODELDIR = os.path.join(os.getcwd(), 'src', 'model')
    AUXDIR = os.path.join(os.getcwd(), 'src', 'auxiliary')
    DATADIR = os.path.join(os.getcwd(), 'data') 
    main()



