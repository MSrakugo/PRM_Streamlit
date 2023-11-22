#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2023/11/21

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: Satoshi Matsuno. (2023). Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models (PRMs) (v1.1). Zenodo. https://doi.org/10.5281/zenodo.10183974
"""
import glob

import streamlit as st
import pandas as pd
import numpy as np
import openpyxl

import plotly.express as px
import matplotlib.pyplot as plt

import PRM_liblary as prm
import PRM_Predict_liblary as prm_predict
import Library_preprocessing as preprocessing
import PRM_App_Library as App_Library

class Models:
    def __init__(self, model, sc, pca, ICA):
        self.model = model
        self.sc = sc
        self.pca = pca
        self.ICA = ICA

class Model_datas:
    def __init__(self, Score_all, test_error_all, test_data_all):
        self.Score_all = Score_all
        self.test_error_all = test_error_all
        self.test_data_all = test_data_all

class Model_feature_setting: # feature setting
    def __init__(self, ML_algorithm_name, setting_X_raw, setting_X_log, setting_X_product, setting_X_product_log, setting_X_ratio, setting_X_ratio_log, setting_NORMAL_OR_LOG, \
                 setting_PCA, setting_ICA, setting_standard_scaler):
        #Model name LightGBM, NGBoost
        self.ML_algorithm_name = ML_algorithm_name

        #Setting
        self.setting_X_raw = setting_X_raw
        self.setting_X_log = setting_X_log
        self.setting_X_product = setting_X_product
        self.setting_X_product_log = setting_X_product_log

        self.setting_X_ratio = setting_X_ratio
        self.setting_X_ratio_log = setting_X_ratio_log

        #standard_scalerに与える時，X_log or Xを選択
        self.setting_NORMAL_OR_LOG = setting_NORMAL_OR_LOG
        self.setting_PCA = setting_PCA
        self.setting_ICA = setting_ICA
        self.setting_standard_scaler = setting_standard_scaler

#################################################################################### Header
st.header("Appendix. Preprocessing & Visualization")
#################################################################################### Header

#################################################################################### sidebar
#### read example dataset and set download_button
example_data = pd.read_csv("Example_Dataset/Example_dataset(Kelley_2003_Seafloor_altered_basalt).csv", index_col=0)
st.sidebar.download_button(
    label="Example dataset (Quoted from PetDB)",
    data=example_data.to_csv().encode('utf-8'),
    file_name="Example_dataset(Kelley_2003).csv",
    mime='text/csv',
    )

######################## Data read
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)")
uploaded_file_Setting = st.sidebar.expander("Input file Setting")
# File setting
transform_T = uploaded_file_Setting.checkbox("Rotate")
if transform_T:
    st.write('ON')
index_col = uploaded_file_Setting.slider('Input index_columns number', 0, 100, 0)
header = uploaded_file_Setting.slider('Input header number', 0, 100, 0)
# File setting
###### Dataが存在しているかを判定
if uploaded_file is not None:
    raw_data="EXIST"
else:
    raw_data=None
###### Dataが存在しているかを判定
######################## Data read
######################## Output setting
Output_Setting = st.sidebar.expander("Output file Setting")
###### Default setting
DataBase="No_database_info"
SAMPLE_INFO="No_database_info"
###### Default setting
DataBase=Output_Setting.text_input("Write database", "No_database_info")
SAMPLE_INFO=Output_Setting.text_input("Write sample infomation", "No_database_info")
location_info = ["DataBase", "SAMPLE_INFO"]
###### Data read and convert データが存在するときだけ実行
if uploaded_file is not None:
    raw_data = prm.read_Raw_data(uploaded_file, index_col, header, DataBase, SAMPLE_INFO)
    if transform_T == 'ON':
        raw_data = raw_data.T
###### Data read and convert
###### data information updated
if uploaded_file is not None:
    location_info = Output_Setting.multiselect("Choose the location columns", raw_data.columns, ["DataBase", "SAMPLE_INFO"])
###### data information updated
######################## Output setting
#################################################################################### sidebar

#################################################################################### Main
###### Data read
if raw_data is None:
    pass
else:
    try:
        st.write(raw_data)
    except:
        pass
    ###### Data check/preprocessing
    PM, Location_Ref_Data = prm.preprocessing_normalize_output(raw_data, DataBase, SAMPLE_INFO, location_info)
#################################################################################### Main

#################################################################################### Visualization
    ###### Data visualization
    st.subheader("Visualize your data")

    # elements_list_nowの順番で表示するため、元素listの準備
    elements_list_now = PM.columns

    with st.expander("See figures"):
        # select sample
        choice_sample = st.selectbox('Select sample',PM.index,)
        elements_list_now = st.multiselect("Choose the visualize elements", list(elements_list_now), ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'Nd', 'Zr', 'Ti', 'Y', 'Yb','Lu'])

        st.subheader("Spidergram")

        ######################################################################## Spidergram
        ###### figure
        fig, ax = plt.subplots(constrained_layout=True)
        # road data
        pred_data_now = pd.DataFrame(PM.loc[choice_sample]).T.dropna(axis=1)[elements_list_now]
        now_col=pred_data_now.columns
        raw_data_now=pd.DataFrame(PM.loc[choice_sample]).T[now_col]
        #model_score_now=model_score[now_col]
        values = st.slider('Select y axis range in log scale for spidergram',-10.0, 10.0, (-1.0, 3.0))

        # figure control
        fig=prm.Spidergram_simple(raw_data_now, "log", "off","#344c5c", "--", "off", fig, ax)
        # figure control
        # figure setting
        plt.title(choice_sample)
        plt.ylabel("Sample / PM")
        plt.ylim(10**values[0], 10**values[1])
        plt.tick_params(which='both', direction='in',bottom=True, left=True, top=True, right=True)
        plt.tick_params(which = 'major', length = 7.5, width = 2)
        plt.tick_params(which = 'minor', length = 4, width = 1)
        fig.tight_layout()
        # figure setting
        st.pyplot(fig)
        # save and download figure
        fn = "picture/" + str(choice_sample) + '_PM_norm.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )
        ######################################################################## Spidergram
#################################################################################### Visualization

App_Library.page_footer()
