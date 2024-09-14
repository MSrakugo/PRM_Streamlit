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
import pickle

import PRM_liblary as prm
import PRM_Predict_liblary as prm_predict
import Library_preprocessing as preprocessing
import PRM_App_Library as App_Library
import Library_model_construction as construction_PRM

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
st.header("3rd step: PRM prediction")
st.markdown('> **1.**   : Choose your data file')
st.markdown('> **2.**   : Set model setting')
st.markdown('> **3.**   : Download data')
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
index_col = uploaded_file_Setting.slider('Input index_columns number', 0, 10, 0)
header = uploaded_file_Setting.slider('Input header number', 0, 10, 0)
# File setting
###### Dataが存在しているかを判定
if uploaded_file is not None:
    raw_data="EXIST"
else:
    raw_data=None
###### Dataが存在しているかを判定
######################## Data read

######################## Model setting
Model_Setting = st.sidebar.expander("Model Setting")
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
Model_info.write("Input variation list in " + Model_name + " folder")
Model_info.table(input_variation_list)
###### Input variation check

###### Select element
if raw_data is None:
    pass
else:
    elements_list = ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'Nd', 'Zr', 'Ti', 'Y', 'Yb','Lu', 'SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO', 'K2O', 'TiO2']
    immobile_elem=Model_Setting.multiselect("Choose the immobile elements (Check available models)", elements_list, ["Zr", "Th", "Ti", "Nb"])
    #mobile_elem=Model_Setting.multiselect("Choose the ALL elements (Contain both mobile and immobile elements)", elements_list, ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'Nd', 'Zr', 'Ti', 'Y', 'Yb','Lu','SiO2','MgO', 'Na2O', 'P2O5', 'CaO'])
    mobile_elem=Model_Setting.multiselect("Choose the Mobile elements", elements_list, ['Rb', 'Ba', 'U', 'K', 'La', 'Ce', 'Pb', 'Sr', 'Nd', 'Y', 'Yb','Lu','SiO2','MgO', 'Na2O', 'P2O5', 'CaO', 'TiO2'])

Ratio_flag = Model_Setting.checkbox("Ratio")
###### Select element
###### For ratio model ->Input自身を推定する
######################## Model setting

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

###### For ratio model ->Input自身を推定する
#Caution for Ratio ver240707
#例えば、Ti, Nb, Zr, Y, Thの比 → Zr濃度 を求めることは想定していない（基本的に重複削除の方針）。
#そのため、Ratioデータについては"_"を末尾につけることで、他元素かつPM normalizationに含まれない元素として例外処理する。
#例えば、Ti, Nb, Zr, Y, Thの比 → Zr_ という形として記述する
if Ratio_flag:
    ######## 準備
    # List of elements to check for
    #target_elements = ['Si', 'Al', 'Mg', 'Na', 'P', 'Ca', 'Mn', 'Fe', 'K', 'Ti']
    #rename_elements = ['SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO', 'K2O', 'TiO2']
    ######## 準備
    # 重複元素を_をつけて別元素として記録
    duplicates, duplicates_dict, mobile_elem_all = construction_PRM.check_and_modify_duplicates(mobile_elem, immobile_elem)
    
    # duplicates に含まれている target_elements を major_duplicate としてリスト化->PMで処理してから追加の際に使う ver 240914
    #major_duplicate = [elem for elem in target_elements if elem in duplicates]
    # major_duplicate を target_elements と rename_elements に従ってリネーム
    #renamed_major_duplicate = [rename_elements[target_elements.index(elem)] for elem in major_duplicate]

    duplicates_df = raw_data[duplicates].copy()
    duplicates_df.rename(columns=duplicates_dict, inplace=True)
    duplicates_df = duplicates_df.rename(columns=duplicates_dict)
    duplicates_df.columns = [elem + '_' for elem in duplicates]
    
    # elem_allとWhole_rock_RAWに追加
    mobile_elem = mobile_elem+[elem + '_' for elem in duplicates]
    raw_data = pd.concat([raw_data, duplicates_df], axis=1)

###### モデル推定開始のフラグ
#flag_MODEL_RUN=1
###### モデル推定開始のフラグ
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

    ###### For ratio model -> PMにMajor元素の自己推定するためのデータを追加 ver 240914
    #major_duplicate_name = [elem + '_' for elem in major_duplicate]
    # major_duplicate を target_elements と rename_elements に従ってリネーム
    #renamed_major_duplicate = [rename_elements[target_elements.index(elem)] for elem in major_duplicate]
    #PM[major_duplicate_name] = PM[major_duplicate].copy()
    #PM[renamed_major_duplicate] = PM[renamed_major_duplicate].copy()
    #st.write(major_duplicate_name, renamed_major_duplicate)
    ###### For ratio model -> PMにMajor元素の自己推定するためのデータを追加 ver 240914

    ###### estimation by PRM
    # model folder
    now_model_folder_name = model_path
    st.write(model_path)
    # estimate
    mobile_data_compile, spidergram_data_compile, mobile_data_compile_dist, spidergram_data_compile_dist = prm_predict.predict_protolith(mobile_elem, immobile_elem, PM, Location_Ref_Data, now_model_folder_name)

    # data compile
    protolith_data = spidergram_data_compile.copy()
    protolith_data_ppm = prm.PM_to_ppm(protolith_data)
    element_mobility = PM[protolith_data.columns]/protolith_data

    #################################################### Download
    n = 5
    col_list = [1] * n
    st.subheader("Download PRMs results")
    col1, col2, col3, col4, col5 = st.columns(col_list)

    with col1:
        st.download_button(
            label="Protolith composition predicted by PRM (PM_norm)",
            data=protolith_data.to_csv().encode('utf-8'),
            file_name="Protolith_comp_by_PRM_(PM_norm)_" + DataBase +".csv",
            mime='text/csv',
            )
    with col2:
        st.download_button(
            label="Protolith composition predicted by PRM (ppm)",
            data=protolith_data_ppm.to_csv().encode('utf-8'),
            file_name="Protolith_comp_by_PRM_(ppm)" + DataBase +".csv",
            mime='text/csv',
            )

    with col3:
        st.download_button(
            label="Element mobile predicted by PRM (ppm)",
            data=mobile_data_compile.to_csv().encode('utf-8'),
            file_name="Element_mobile_by_PRM_" + DataBase +".csv",
            mime='text/csv',
            )

    with col4:
        st.download_button(
            label="Element mobility predicted by PRM (Metabasalt / Protolith)",
            data=element_mobility.to_csv().encode('utf-8'),
            file_name="Element_mobility_by_PRM_" + DataBase +".csv",
            mime='text/csv',
            )

    with col5:
        st.download_button(
            label="Predictive uncertanty",
            data=spidergram_data_compile_dist.to_csv().encode('utf-8'),
            file_name="Predictive_uncertanty_" + DataBase +".csv",
            mime='text/csv',
            )
    #################################################### Download
#################################################################################### Main

#################################################################################### Visualization
    ###### Data visualization
    st.subheader("Visualize your data")

    # elements_list_nowの順番で表示するため、元素listの準備
    elements_list_now = elements_list
    elem_use = mobile_elem + immobile_elem
    elem_remove = set(elements_list_now) ^ set(elem_use)
    elements_list_now = [i for i in elements_list_now if i not in elem_remove]

    with st.expander("See figures"):
        # select sample
        choice_sample = st.selectbox('Select sample',spidergram_data_compile.index, )
        elements_list_now = st.multiselect("Choose the visualize elements", elements_list_now, elements_list_now)

        st.subheader("Spidergram")

        raw_comp = st.checkbox("Raw composition")
        prt_comp = st.checkbox("Protolith composition")

        ######################################################################## Spidergram
        ###### figure
        fig, ax = plt.subplots(constrained_layout=True)
        # road data
        pred_data_now = pd.DataFrame(spidergram_data_compile.loc[choice_sample]).T.dropna(axis=1)[elements_list_now]
        now_col=pred_data_now.columns
        raw_data_now=pd.DataFrame(PM.loc[choice_sample]).T[now_col]
        #model_score_now=model_score[now_col]
        values = st.slider('Select y axis range in log scale for spidergram',-10.0, 10.0, (-1.0, 3.0))

        # figure control
        if prt_comp:
            fig=prm.Spidergram_fill_immobile(now_col, immobile_elem, '#ecc06f', 0.18, fig, ax)
        #fig=prm.Spidergram_error(pred_data_now, model_score_now,"log", "on","#f08575", "-", "off", fig, ax)
        if raw_comp:
            fig=prm.Spidergram_simple(raw_data_now, "log", "off","#344c5c", "--", "off", fig, ax)
        if prt_comp:
            fig=prm.Spidergram_simple(pred_data_now,"log", "off","#f08575", "-", "off", fig, ax)
        if prt_comp:
            fig=prm.Spidergram_marker(raw_data_now, immobile_elem, '#f08575', '#344c5c', 'd', 16, fig, ax)
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
        ######################################################################## Mobility
        st.subheader("Element mobility")
        fig, ax = plt.subplots(constrained_layout=True)
        # road data
        pred_mobility_now = raw_data_now/pred_data_now
        values_m = st.slider('Select y axis range in log scale for mobility figure',-10.0, 10.0, (-1.0, 2.0))
        # figure control
        fig=prm.Spidergram_simple(pred_mobility_now,"log", "off","#f08575", "-", "off", fig, ax)
        #fig=prm.Spidergram_error(pred_mobility_now, model_score_now,"log", "off","#f08575", "-", "off", fig, ax)
        fig=prm.Spidergram_marker(pred_mobility_now, immobile_elem, '#f08575', '#344c5c', 'd', 16, fig, ax)
        ax.axhline(y=1, xmin=0, xmax=len(pred_mobility_now.columns)-1, color = "#344c5c", linestyle='--',)
        # figure control
        # figure setting
        plt.title(choice_sample)
        plt.ylabel("Metabasalt/Protolith")
        plt.ylim(10**values_m[0], 10**values_m[1])
        plt.tick_params(which='both', direction='in',bottom=True, left=True, top=True, right=True)
        plt.tick_params(which = 'major', length = 7.5, width = 2)
        plt.tick_params(which = 'minor', length = 4, width = 1)
        fig.tight_layout()
        # figure setting
        st.pyplot(fig)
        # save and download figure
        fn = "picture/" + str(choice_sample) + '_mobility.png'
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="Download image",
                data=img,
                file_name=fn,
                mime="image/png"
            )
        ######################################################################## Mobility
#################################################################################### Visualization

App_Library.page_footer()
