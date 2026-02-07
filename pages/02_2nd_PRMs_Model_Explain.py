#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2024/05/12

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
import pickle

import PRM_liblary as prm
import PRM_Predict_liblary as prm_predict
import Library_preprocessing as preprocessing
import PRM_App_Library as App_Library
import PRM_Explain_Library as prm_ex

######################## Model setting
Model_Setting = st.sidebar.expander("Explain Model Set")
Model_info = st.expander("Model Folder Information")

###### Algorithm folder check
Algorithm_name = None
folder_algorithm_name = glob.glob("0_PRM_Model_Folder/**")
algorithm_list = []
for algorithm_name in folder_algorithm_name:
    algorithm_list.append(algorithm_name.split("/")[1])
Model_info.write("Algorithm list")
Model_info.caption(algorithm_list)

Algorithm_name=Model_Setting.selectbox("Select Algorithm", algorithm_list) # Select Algorithm
if Algorithm_name is not None:
    folder_Model_name = glob.glob("0_PRM_Model_Folder/"+ Algorithm_name +"/**")
###### Algorithm folder check

###### Model folder check
Model_name = None
folder_Model_name = glob.glob("0_PRM_Model_Folder/"+ Algorithm_name +"/**")
Model_list = []
for Model_name in folder_Model_name:
    Model_list.append(Model_name.split("/")[-1]) # / でSplitしたlistの一番最後をappend
Model_info.write("Model list in " + Algorithm_name + " folder")
Model_info.caption(Model_list)

Model_name=Model_Setting.selectbox("Select Model", Model_list) # Select Model
if Model_name is not None:
    folder_Model_name = glob.glob("0_PRM_Model_Folder/"+ Algorithm_name +"/**")
###### Model folder check

###### Input variation check
model_path = "0_PRM_Model_Folder/"+ Algorithm_name +"/" + Model_name ###### DEFINE Model path
folder_input_variation_name = glob.glob(model_path+"/**")
input_variation_list = []
for input_variation in folder_input_variation_name:
    input_variation_list.append(input_variation.split("/")[-1]) # / でSplitしたlistの一番最後をappend
if "Protolith" in input_variation_list:
    input_variation_list.remove("Protolith") #Model_explainをListから削除
if "USE_DATA.xlsx" in input_variation_list:
    input_variation_list.remove("USE_DATA.xlsx") #Model_explainをListから削除
if "0_error_list.xlsx" in input_variation_list:
    input_variation_list.remove("0_error_list.xlsx") #Model_explainをListから削除
Model_info.write("Input variation list in " + Model_name + " folder")
Model_info.table(input_variation_list)
###### Input variation check

Input_name=Model_Setting.selectbox("Select Input", input_variation_list) # Select Model
###### Output variation check
model_path = "0_PRM_Model_Folder/"+ Algorithm_name +"/" + Model_name +"/" + Input_name ###### DEFINE Model path
folder_output_variation_name = glob.glob(model_path+"/**")
output_variation_list = []
for output_variation in folder_output_variation_name:
    output_variation_list.append(output_variation.split("/")[-1]) # / でSplitしたlistの一番最後をappend
if "Model_explain" in output_variation_list:
    output_variation_list.remove("Model_explain") #Model_explainをListから削除
Model_info.write("Output variation list in " + Model_name + " folder")
Model_info.table(output_variation_list)
###### Output variation check

# DEFINE OUTPUT
mobile_elem_all=Model_Setting.multiselect("Choose the Output elements", output_variation_list, output_variation_list)
# DEFINE OUTPUT
# DEFINE good_range_elem
good_range_elem=Model_Setting.multiselect("Choose the Good range elem", output_variation_list, [])
# DEFINE good_range_elem

#Location dataのロード
try:
    Raw_Protolith_location = pd.read_excel("0_PRM_Model_Folder/"+ Algorithm_name +"/" + Model_name+"/Protolith/Location_Ref_Data.xlsx", index_col=0)
except: # 昔のセッティングで0_Protolithの場合があったので、例外処理を追加
    Raw_Protolith_location = pd.read_excel("0_PRM_Model_Folder/"+ Algorithm_name +"/" + Model_name+"/0_Protolith/Location_Ref_Data.xlsx", index_col=0)

TECTONIC_list = Raw_Protolith_location["SAMPLE_INFO"].unique()
st.sidebar.caption("Color Order: ")
st.sidebar.caption(TECTONIC_list)
#################################################################################### sidebar

#################################################################################### Main
path_model = "0_PRM_Model_Folder/"+ Algorithm_name +"/" + Model_name
path = "0_PRM_Model_Folder/"+ Algorithm_name +"/" + Model_name + "/" + Input_name 
###### Data read
Start_Figure_Making = st.button("Start")
if Start_Figure_Making:
    prm_ex.Error_Distribution_Figure(path_model, path, mobile_elem_all, good_range_elem, TECTONIC_list)
    prm_ex.Usual_Scatter_Plot(path_model, path, mobile_elem_all, good_range_elem, TECTONIC_list)
    prm_ex.Usual_Scatter_Plot_norm(path_model, path, mobile_elem_all, good_range_elem, TECTONIC_list)

    if Algorithm_name == "NGBoost":
        prm_ex.NGBoost_Scatter_Plot(path_model, path, mobile_elem_all, good_range_elem, TECTONIC_list)

#################################################################################### Visualization

App_Library.page_footer()
