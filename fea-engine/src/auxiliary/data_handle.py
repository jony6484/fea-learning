import pickle as pk
import pandas as pd
import numpy as np

def read_data(path):
    data = pd.read_csv(path)
    return data

def make_cols(cols: list):
    cols_height = cols[cols.str.contains('height')].to_list()
    cols_pressure = cols[cols.str.contains('pressure')].to_list()
    cols_C = cols[cols.str.startswith('C')].to_list()
    return cols_height, cols_pressure, cols_C

def make_coef_cols(df_coef):
    cols = df_coef.columns
    coef_cols = cols[cols.str.startswith('coef')].to_list()
    return coef_cols    


def raw_data_to_X_y(raw_data: pd.DataFrame):
    cols = raw_data.columns
    cols_height, cols_pressure, cols_C = make_cols(cols)
    raw_data_y = raw_data[cols_C]
    raw_data_X = raw_data[cols_height + cols_pressure]
    return raw_data_X, raw_data_y

def preprocess_x_coef(raw_data_X: pd.DataFrame, model_params: dict):
    log_x = model_params['log_x']
    degree = model_params['poly_degree']
    cols = raw_data_X.columns
    cols_height, cols_pressure, _ = make_cols(cols)
    coef_cols = [f'coef {j}' for j in range(degree+1)]
    new_cols_names = ['mu_x', 's_x'] + coef_cols
    def poly_coef(row):
        x = row[cols_height]
        y = row[cols_pressure]
        # Log x
        if  log_x == True:
            x = np.log(x)
        # Norm X
        mu_x = x.mean()
        s_x = x.std()
        x = (x-mu_x)/s_x
        return np.append([mu_x, s_x], np.polyfit(x, y, deg=degree))
    ####################################################################
    df_coef = pd.DataFrame(zip(*raw_data_X.apply(poly_coef, axis=1))).T
    ####################################################################
    df_coef.columns = new_cols_names
    return df_coef

def make_mX(df_coef, model_params: dict):
    norm_coef = model_params['norm_coef']
    coef_cols = make_coef_cols(df_coef)
    mX = df_coef[coef_cols].values
    mu_coef = mX.mean(0)
    s_coef = mX.std(0)
    if norm_coef == True:
        mX = (mX - mu_coef)/s_coef

    return {"mX": mX, "mu_coef": mu_coef, "s_coef": s_coef} 


def preprocess_y(raw_data_y: pd.DataFrame, model_params: dict):
    norm_y = model_params['norm_y']
    mY = raw_data_y.values
    mu_y = mY.mean(0)
    s_y = mY.std(0)
    if norm_y == True:
        mY = (mY - mu_y)/s_y
    return mY, mu_y, s_y


def save_model_to_file(model_and_params, path):
    """
    Save to file
    """
    with open(path, 'wb') as file:
        pk.dump(model_and_params, file)
        
        
 