import numpy as np
import pandas as pd
import pickle as pk
from sklearn.ensemble import RandomForestRegressor
import os, sys
from src.auxiliary import data_handle

MODELDIR = os.path.join(os.getcwd(), 'src', 'model')

def init_model():
    # load the saved model
    path = os.path.join(MODELDIR, "model_and_params.pk")
    if os.path.exists(path):
        model, fit_params = pk.load(open(path, "rb"))
    return model, fit_params

def exp_start_above_zero(x_exp, y_exp):
    # Strat at x=0 and remove first point
    x_exp = x_exp - x_exp.min()
    x_exp = x_exp[1:]
    y_exp = y_exp[1:]
    return x_exp, y_exp
    

def exp_preprocess(x_exp, y_exp, model_params):
    # Unifieng the points distribution
    x_exp, y_exp = dist_unifier(x_exp,y_exp, noise_level_y=0.01)
    
    # Convert experiment data to DataFrame    
    df_inference = pd.DataFrame(np.concatenate([x_exp, y_exp])).T
    df_inference.columns = ['height' + str(ii) for ii in range(len(x_exp))] + ['pressure' + str(ii) for ii in range(len(y_exp))]
    mu_coef = model_params['mu_coef']
    s_coef = model_params['s_coef']
    df_coef = data_handle.preprocess_x_coef(df_inference, model_params)
    # processed_data = data_handle.make_mX(df_coef, model_params, train_mode=False, mu_coef=mu_coef, s_coef=s_coef)
    return df_coef


def eval_poly_data(x_exp, df_coef, model_param):
    x_exp_tag = x_exp.copy()
    if model_param['log_x'] == True:
        x_exp_tag = np.log(x_exp_tag)
    mu_x = df_coef['mu_x'].values[0]
    s_x = df_coef['s_x'].values[0]
    x_exp_tag = (x_exp_tag - mu_x) / s_x
    cols = df_coef.columns
    cols_coef = cols[cols.str.startswith('coef')].to_list()
    coefs = df_coef[cols_coef].values[0]
    y_exp_tag = np.polyval(coefs, x_exp_tag)
    return y_exp_tag

def fit_model(model, X, y):
    if isinstance(model, RandomForestRegressor):
        model.fit(X, y)
        return model
    



def model_predict(model, model_params, X):

    # Predicting C1, C2, C3:
    if isinstance(model, RandomForestRegressor):
        C = model.predict(X).squeeze()
        if model_params['norm_y'] == True:
            C = C*model_params['s_y'] + model_params['mu_y']
        return C


def dist_unifier(x, y, bins=10, noise_level_x=0.05, noise_level_y=0.005):
    # Unifieng the data distribution
    unifier_qty, unifier_bins = np.histogram(x, bins=bins)
    unifier_mul = unifier_qty.max() // unifier_qty
    unifier_mul

    N = len(x)
    new_N = (unifier_qty*unifier_mul).sum()

    output_x = np.array([])
    output_y = np.array([])
    
    # Adding points:
    for i in range(len(unifier_qty)):

        idx_bin_i = (x > unifier_bins[i]) & (x <= unifier_bins[i+1]) 

        output_x = np.hstack([output_x, np.repeat(x[idx_bin_i], unifier_mul[i])])
        output_y = np.hstack([output_y, np.repeat(y[idx_bin_i], unifier_mul[i])])
        
    # Adding noise:
    eps_x = np.random.randn(len(output_x))/output_x.std()
    eps_y = np.random.randn(len(output_y))/output_y.std()
    eps_x *= noise_level_x
    eps_y *= noise_level_y

    output_x += eps_x
    output_y += eps_y
    order_idx = np.argsort(output_x)
    output_x = output_x[order_idx]
    output_y = output_y[order_idx]
    
    # Removing points:
    output_x = output_x[::new_N//N]
    output_y = output_y[::new_N//N]
    
    return output_x, output_y