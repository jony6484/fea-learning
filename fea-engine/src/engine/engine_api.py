import numpy as np
import pandas as pd
import pickle as pk

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.auxiliary import data_handle


import engine

app = FastAPI()

####### Class for typing #######
class ExpData(BaseModel):
    heights: List[float]
    pressures: List[float]

class ExpCoefs(BaseModel):
    coefs: List[float]
    mu_x: float
    s_x: float

####### The Endpoint for predict #######
@app.post("/get-c123")
async def get_c123(exp_coefs: ExpCoefs):
    X = np.array(exp_coefs.coefs)
    norm_coef = app.model_params['norm_coef']
    mu_coef = app.model_params['mu_coef']
    s_coef =app.model_params['s_coef']
    if norm_coef == True:
        X = (X - mu_coef)/s_coef
    c1, c2, c3 = engine.model_predict(model = app.model, model_params=app.model_params, X=X[None, :])
    return {"c1": c1, "c2": c2, "c3": c3}

####### The Endpoint for plot #######
@app.post("/get-coef")
async def get_coef_array(exp_arr: ExpData):
    x_exp = np.array(exp_arr.heights)
    y_exp = np.array(exp_arr.pressures)
    x_exp, y_exp = engine.exp_start_above_zero(x_exp, y_exp)
    df_coef = engine.exp_preprocess(x_exp=x_exp, y_exp=y_exp, model_params=app.model_params)    
    # y_exp_tag = engine.eval_poly_data(x_exp, df_coef, model_param)
    # payload = {'heights': x_exp.tolist(), 'pressures': y_exp_tag.tolist()}
    coef_cols = data_handle.make_coef_cols(df_coef)
    coefs = df_coef[coef_cols].values.squeeze().tolist()
    mu_x = df_coef['mu_x'].values[0]
    s_x = df_coef['s_x'].values[0]
    payload = {'coefs': coefs, 'mu_x': mu_x, 's_x': s_x}
    return payload

####### The Endpoint for model params #######
@app.get("/get-model-params")
async def get_model_params():
    model_params = app.model_params
    model_params = {k:(v.tolist() if type(v) == np.ndarray else v) for (k,v) in model_params.items()}
    return model_params

####### Code begins here #######
def main():
    model, model_params = engine.init_model()
    app.model = model
    app.model_params = model_params
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
if __name__ == "__main__":
    main()

    
    
    



    
    # #####
    # if fit_params['log_x'] == True:
    #     x_exp_log = np.log(x_exp)
    #     x_exp_norm = (x_exp_log - np.mean(x_exp_log)) / np.std(x_exp_log)
    # #####
    # else:
    #     x_exp_norm = (x_exp - np.mean(x_exp)) / np.std(x_exp)

    # coefs_exp = np.polyfit(x_exp_norm, y_exp, deg=fit_params['poly_degree']) # Using Numpy 
    # if fit_params['norm_x'] == True:
    #     coefs_exp = (coefs_exp - fit_params['mu_x'])/fit_params['s_x']   
    # # Predicting C1, C2, C3:
    # C = regressor.predict(coefs_exp[None, :]).squeeze()
    # if fit_params['norm_y'] == True:
    #     C = C*fit_params['s_y'] + fit_params['mu_y']

