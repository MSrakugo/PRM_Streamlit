#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2024/09/18

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: Satoshi Matsuno. (2023). Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models (PRMs) (v1.1). Zenodo. https://doi.org/10.5281/zenodo.10183974
"""

import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from concurrent.futures import ThreadPoolExecutor

# Date time
import datetime

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import PRM_liblary as prm
import PRM_Predict_liblary as prm_predict
import Library_preprocessing as preprocessing_PRM
import Library_model_construction as construction_PRM
import PRM_App_Library as App_Library

def make_dirs(path):
    import os
    try:
        os.makedirs(path)
    except:
        pass

#################################################################################### sidebar
st.header("1st step: PRMs Construction (Optional)")

################################################################# Preprocessing
##### define path for dataset

Protolith_DATA_ex = st.sidebar.expander("Protolith Data Setting")
data_name = Protolith_DATA_ex.text_input("Write Data Name", "Protolith")
Protolith_DATA_ex.caption("Default data: MORB, VAB, OIB")
own_data = Protolith_DATA_ex.checkbox("Upload your protolith data")
if own_data:
    uploaded_file = [Protolith_DATA_ex.file_uploader("Choose a file (Excel or CSV)")] # 240326修正 ln96のlen機能いらない？
    DataBase = Protolith_DATA_ex.text_input("Write DataBase", "PetDB")
    SAMPLE_INFO = Protolith_DATA_ex.text_input("Write Sample Info", "Protolith")
else: # Default -> Metabsalt (MORB, VAB, OIB)
    uploaded_file = ['0_Model_Construction_Protolith_Composition_Data/PRM_For_Metabasalt/PetDB_Metabasalt_20231126.xlsx']
    DataBase = 'petdb'
    SAMPLE_INFO = 'Protolith'
# Set index and header
index_col = Protolith_DATA_ex.slider('Input index_columns number', 0, 10, 0)
header = Protolith_DATA_ex.slider('Input header number', 0, 10, 7)

'''
Same protocol can do with "0_BASE_Model_Construction.py"
*******For dataset caution*******
1. (Optional) Add "ANALYZED MATERIAL" and write WHOLE ROCK
2. check index col and sample name (if duplicated, the sample will be delete automatically)
'''
################################################################# Preprocessing

###################################################### Model setting
Model_ex = st.sidebar.expander("Model Setting")

today_date = str(datetime.date.today())
Model_ex.caption(today_date)

PRM_construction_Setting = Model_ex.selectbox("Construction Setting", ['Normal', 'Ratio', 'Optional'])
Model_algorithm = Model_ex.selectbox("Model algorithm", ['LightGBM', 'NGBoost'])
Model_Training_Process = Model_ex.selectbox("Model Training Setting", ['Default', 'Optional'])
Minimum_combination_number = Model_ex.slider('Input header number', 1, 10, 4)
if PRM_construction_Setting == "Optional": # ver 240918 Ratioデータに対応するため
    # もし Ratio で推定を行う場合には、Input Outputに重複が存在しても良い形に変更する→Ratioに変更
    PRM_construction_Setting_ratio_check = st.sidebar.checkbox("Ratio")
    if PRM_construction_Setting_ratio_check:
        PRM_construction_Setting = "Ratio"

today_date = today_date + '_' + PRM_construction_Setting + '_' + Model_algorithm+ '_Mincomb_' + str(Minimum_combination_number)
Model_ex.caption("Model Path: " + today_date)

# Model Setting
if PRM_construction_Setting == 'Normal':
    feature_setting, Training_Setting = construction_PRM.Model_All_Setting(PRM_construction_Setting, Model_Training_Process, Model_algorithm, Model_ex)
elif PRM_construction_Setting == 'Ratio':
    feature_setting, Training_Setting = construction_PRM.Model_All_Setting(PRM_construction_Setting, Model_Training_Process, Model_algorithm, Model_ex)
elif PRM_construction_Setting == 'Optional':
    feature_setting, Training_Setting = construction_PRM.Model_All_Setting(PRM_construction_Setting, Model_Training_Process, Model_algorithm, Model_ex)
###################################################### Model setting

###### Data read
Start_Preprocessing = st.sidebar.checkbox("Start Preprocessing")
if Start_Preprocessing:
    ###################################################### Preprocessing
    ###################################################### Data read and check duplicates
    raw_data=pd.DataFrame()
    for num in range(len(uploaded_file)):
        raw_data_num = preprocessing_PRM.read_Raw_data(uploaded_file[num], index_col, header, DataBase, SAMPLE_INFO)
        raw_data = pd.concat([raw_data, raw_data_num], axis=0)
    #duplicate columns delete
    raw_data['index']=raw_data.index
    raw_data = raw_data.drop_duplicates(subset=['index'], keep='first')

    # tectonic setting
    if 'TECTONIC SETTING' in raw_data.columns:
        raw_data['SAMPLE_INFO']=raw_data['TECTONIC SETTING'] #PetDB
    else:
        raw_data['SAMPLE_INFO']=SAMPLE_INFO # 代わりにSAMPLE_INFOを入れておく
    ###################################################### Data read and check duplicates
    #Preprocessing
    Whole_rock_RAW, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1 = preprocessing_PRM.Preprocessing_all(raw_data)
    #pathの名前をつける
    path_name = "0_PRM_Model_Folder/" + Model_algorithm + "/" + today_date + '_ALL/'
    #save data take some secounds # Protolith dataのsave
    #_ = preprocessing_PRM.save_preprocessed_data(path_name, data_name, Whole_rock_RAW, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1)
    print("FIN: Preprocessing")


    """
    Caution when determing the input/output elements
    主要元素の設定を"Ti"などとした場合にエラーが出る　
    -> Applyするときに、Inputの設定で問題が生じるため：具体的にはPMノーマライズ前に元素設定をするため、Tiではエラーが発生する
    ->"TiO2"のように元素を設定するようにすること

    If you set the main element to “Ti” or something similar, you will get an error　
    -> When applying, there is a problem with the Input settings: specifically, because the element is set before PM normalization, an error occurs with Ti
    -> Set the element like “TiO2”.
    """
    ##################### Model Element setting
    #### initial setting-> element
    elem_all = ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'P', 'Nd', 'Zr', 'Ti', 'Y', 'Yb', 'Lu', 'SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO', 'K2O']
    immobile_elem_all = ['Zr', 'Th', 'Ti', 'Nb']

    elem_all = st.multiselect("Choose All elem", Whole_rock_RAW.columns, elem_all)
    immobile_elem_all = st.multiselect("Choose Immobile elem", Whole_rock_RAW.columns, immobile_elem_all)

    # ver 240918 Ratioの時は選択されたElement全てをMobile elementと定義（input自身を推定するモデル）
    if PRM_construction_Setting == 'Ratio':
        mobile_elem_all = elem_all
    else: # ver 240918 その他（NormalとOptional）の時は elem_allからimmobile elemに含まれない元素を探す処理
        mobile_elem_all = [elem for elem in elem_all if elem not in immobile_elem_all]
    #### initial setting-> element
    ##################### Model Element setting

    ############################################################################ For Ratio model ver 240918
    """
    Caution for Ratio ver240707
    例えば、Ti, Nb, Zr, Y, Thの比 → Zr濃度 を求めることは想定していない（基本的に重複削除の方針）。
    そのため、Ratioデータについては"_ (アンダーバー)"を末尾につけることで、他元素かつPM normalizationに含まれない元素として例外処理する。
    例えば、Ti, Nb, Zr, Y, Thの比 → Zr_ という形として記述する

    For example, the ratio of Ti, Nb, Zr, Y, and Th → Zr concentration is not assumed to be obtained (basically, the policy is to delete duplicates).
    For this reason, the “_ (underscore)” at the end of the Ratio data is used to treat it as an exception as an element other than the other elements and an element not included in PM normalization.
    For example, the ratio of Ti, Nb, Zr, Y, and Th is written as Zr_.
    """
    if PRM_construction_Setting == 'Ratio':
        # 重複元素を_をつけて別元素として記録
        duplicates, duplicates_dict, mobile_elem_all = construction_PRM.check_and_modify_duplicates(mobile_elem_all, immobile_elem_all)
        duplicates_df = Whole_rock_RAW[duplicates]
        duplicates_df.rename(columns=duplicates_dict, inplace=True)
        duplicates_df = duplicates_df.rename(columns=duplicates_dict)
        
        # elem_allとWhole_rock_RAWに追加
        elem_all = elem_all+[elem + '_' for elem in duplicates]
        mobile_elem_all = elem_all
        Whole_rock_RAW = pd.concat([Whole_rock_RAW, duplicates_df], axis=1)
    ############################################################################ For Ratio model ver 240918

    #### list elem_allに入っていない要素をWhole_rock_after_Normalize_PM.columnsから見つけ出す
    # 基本的にはMajor元素に対する処理
    columns = Whole_rock_after_Normalize_PM.columns
    # elem_all に含まれている要素で、columns に含まれていない要素を探す
    missing_elements = [elem for elem in elem_all if elem not in columns]
    Whole_rock_after_Normalize_PM[missing_elements] = Whole_rock_RAW[missing_elements].copy() # Majoprを入れる
    #### list elem_allに入っていない要素をWhole_rock_after_Normalize_PM.columnsから見つけ出す

    #### Compile use data
    Protolith_data = Whole_rock_after_Normalize_PM[elem_all].copy() #データをまとめる
    Protolith_data.to_excel(path_name + "USE_DATA.xlsx") # データの出力
    print(Protolith_data[immobile_elem_all].dropna().shape)
    Protolith_location_data=Whole_rock_cannot_Normalize[["DataBase", "SAMPLE_INFO"]]
    #### Compile use data

    #### elem combination listの作成
    element_compile = construction_PRM.combination_list(Minimum_combination_number, immobile_elem_all, mobile_elem_all, elem_all)
    immobile_all_list = element_compile['immobile']
    mobile_all_list = element_compile['mobile']
    #### elem combination listの作成

    print(element_compile.shape)
    print(element_compile)
    st.write("Preprocessing Finished")
    ###################################################### Preprocessing

    ###################################################### Model Element setting
    if st.button('Model Training Active'):
        ###################################################### Model Active
        ############### メモリ節約
        try:
            del Whole_rock_RAW, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1
        except:
            print("DATAs are already del")
        ############### メモリ節約

        # Initialize for error
        error_list_immobile=[]
        error_list_mobile=[]

        for immobile_elem, mobile_elem_list in zip(immobile_all_list, mobile_all_list):
            immobile_elem = list(immobile_elem)

            for mobile_elem in mobile_elem_list:
                mobile_elem = [mobile_elem]
                #try:
                construction_PRM.__main__(path_name, mobile_elem, immobile_elem, Protolith_data, Protolith_location_data, Protolith_location_data, feature_setting, Training_Setting)
                #except:
                error_list_immobile.append(str(immobile_elem))
                error_list_mobile.append(str(mobile_elem))

        error_list=pd.DataFrame([error_list_immobile, error_list_mobile])
        error_list.to_excel(path_name+"/0_error_list.xlsx")
            ################################################################# Main
        ###################################################### Model Active
        
        #save data take some secounds # Protolith dataのsave
        _ = preprocessing_PRM.save_preprocessed_data(path_name, data_name, Whole_rock_RAW, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1)

else:
    pass
#################################################################################### Page control
App_Library.page_footer()
