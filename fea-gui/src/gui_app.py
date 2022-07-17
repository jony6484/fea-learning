import json, requests
import numpy as np
import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

# API_URL = "http://" + os.environ['API_IP'] + ":8000"
# API_URL = "http://" + 'localhost' + ":8088"
API_URL = "http://" + 'fastapi' + ":8000"

endpoint_coefs = "/get-coef"
endpoint_c123 = "/get-c123"
endpoint_model_params = "/get-model-params"

endpoint_coef_path = API_URL + endpoint_coefs
endpoint_c123_path = API_URL + endpoint_c123
endpoint_model_params_path = API_URL + endpoint_model_params


st.set_page_config(layout="centered")
# Header:
header_cont  = st.container()
# File load:
file_load_cont = st.container()
col_table, col_plot = st.columns(2)
# Send Data:
coef_cont = st.container()
ready_to_send = False
coef_result = None
# Predict:
predict_cont = st.container()


# Session State:
if 'coefs' not in st.session_state:
    st.session_state['coefs'] = None
if 'fig' not in st.session_state:
    st.session_state['fig'] = None
if 'model_params' not in st.session_state:
    st.session_state['model_params'] = None
if 'mu_x' not in st.session_state:
    st.session_state['mu_x'] = None
if 's_x' not in st.session_state:
    st.session_state['s_x'] = None
if 'y_exp_tag' not in st.session_state:
    st.session_state['y_exp_tag'] = None
if 'c123' not in st.session_state:
    st.session_state['c123'] = None
# BG Color:
bgcolor = "#F5F5F5"
st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True)
@st.cache
def get_model_params():
    result = requests.get(endpoint_model_params_path)
    if result.status_code == 200:
        res_dict = json.loads(result.text)
        return res_dict

@st.cache
def send_exp(exp_data: pd.DataFrame):
    heights = exp_data.iloc[:, 0].values.tolist()
    pressures = exp_data.iloc[:, 1].values.tolist()
    payload = {'heights': heights, 'pressures': pressures}
    payload = json.dumps(payload)
    result = requests.post(endpoint_coef_path, data=payload)
    if result.status_code == 200:
        res_dict = json.loads(result.text)
        st.session_state['coefs'] = res_dict['coefs']
        st.session_state['mu_x'] = res_dict['mu_x']
        st.session_state['s_x'] = res_dict['s_x']
@st.cache        
def mk_coef_df():
    coefs = st.session_state['coefs']
    cols = [f'coef {j}' for j in range(len(coefs))]
    df_coefs = pd.DataFrame(coefs).T
    df_coefs.columns = cols
    return df_coefs
@st.cache
def send_coefs():
    coefs = st.session_state['coefs']
    mu_x = st.session_state['mu_x']
    s_x = st.session_state['s_x']
    payload = {'coefs': coefs, 'mu_x': mu_x, 's_x': s_x}
    payload = json.dumps(payload)
    result = requests.post(endpoint_c123_path, data=payload)
    if result.status_code == 200:
        res_dict = json.loads(result.text)
        st.session_state['c123'] = {'c1': res_dict['c1'], 'c2': res_dict['c2'], 'c3': res_dict['c3']}


def mk_exp_plot_figure(x_exp, y_exp):
    fig = go.Figure()
    # fig = px.line(x=x, y=y)
    fig.add_trace(go.Scatter(x=x_exp, y=y_exp, mode='lines', name='exp data'))
    y_exp_tag = st.session_state['y_exp_tag']
    if y_exp_tag is not None:       
        fig.add_trace(go.Scatter(x=x_exp, y=y_exp_tag, mode='lines', name='poly fit')) 
    fig.update_layout(margin=dict(l=1,r=1,b=1,t=1),
                      width=350, height=280,
                      paper_bgcolor=bgcolor, legend=dict(yanchor="bottom", y=0.7, xanchor="right", x=0.98))
    fig.update_xaxes( title_text = "Height [mm]")#), title_font = {"size": 20}, title_standoff = 25)
    fig.update_yaxes( title_text = "Pressure [atm]")#, title_font = {"size": 20}, title_standoff = 25)
    st.write(fig)
    # st.session_state['fig'] = fig

@st.cache
def eval_poly_data(x_exp, coefs, mu_x, s_x, model_params):
    x_exp_tag = x_exp.copy()
    if model_params['log_x'] == True:
        x_exp_tag = np.log(x_exp_tag)
    x_exp_tag = (x_exp_tag - mu_x) / s_x
    y_exp_tag = np.polyval(coefs, x_exp_tag)
    return y_exp_tag


@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

with header_cont:
    st.title('FEA - learning GUI')
    st.session_state['model_params'] = get_model_params()
with file_load_cont:
    st.subheader("Here is where you upload the experiment data:")
    uploaded_file = st.file_uploader("format is a two column csv ONLY", type='csv')
    if uploaded_file is not None:
        exp_data = pd.read_csv(uploaded_file)
        st.text("file is in!!!")
        ############################
        st.subheader("Data Preprocess - making features:")
        data_sent = st.button(label="Compute Coefs", on_click=send_exp, args=[exp_data])
        x_exp=exp_data.iloc[1:,0].values
        y_exp=exp_data.iloc[1:,1].values
        df_coefs = None
        if st.session_state['coefs'] is not None:
            df_coefs = mk_coef_df()
            y_exp_tag = eval_poly_data(x_exp, st.session_state['coefs'], st.session_state['mu_x'], st.session_state['s_x'], st.session_state['model_params'])
            st.session_state['y_exp_tag'] = y_exp_tag
        ############################
        with col_table:
            fig = go.Figure(data=go.Table(header=dict(values=['heights [mm]', 'pressures [atm]']), cells=dict(values=[exp_data.iloc[1:,0], exp_data.iloc[1:,1]])))
            fig.update_layout(margin=dict(l=1,r=1,b=1,t=1), width=350, height=250, paper_bgcolor=bgcolor)    
            st.write(fig)
        with col_plot:
            mk_exp_plot_figure(x_exp, y_exp)
    else:
        st.stop()
        
        
with coef_cont:
    if df_coefs is not None:
        fig_table_coef = go.Figure(data=go.Table(header=dict(values=df_coefs.columns, font_size=18, height=30),
                                      cells=dict(values=df_coefs.values.T, font_size=18, height=30, format=[".3e",".3e",".3e"]))
                        )
        fig_table_coef.update_layout(margin=dict(l=1,r=1,b=1,t=1), height=60, paper_bgcolor=bgcolor)
        st.plotly_chart(fig_table_coef, use_container_width=True)
    else:
        st.stop()
 

with predict_cont:
    st.subheader("Predicting the material Coeficinets:")
    predicted = st.button(label="Predict", on_click=send_coefs)
    if st.session_state['c123'] is not None:
        c123_dict = st.session_state['c123']
        header = list(c123_dict.keys())
        values = list(c123_dict.values())
        fig = go.Figure(data=go.Table(header=dict(values=header, font_size=18, height=30),
                                      cells=dict(values=values, font_size=18, height=30, format=[".3e",".3e",".3e"]))
                        )
        fig.update_layout(margin=dict(l=1,r=1,b=1,t=1),height=60,  paper_bgcolor=bgcolor)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.stop()
    export_df = pd.DataFrame(data=[list(c123_dict.values())], columns=c123_dict.keys())
    csv_data = convert_df(export_df)
    data_export = st.download_button(label='📥 Download CSV', data=csv_data, file_name= 'c123.csv')

