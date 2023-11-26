 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Begin: Tue Mar  23:33:29 2022
Final update: 2023/11/25

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: Satoshi Matsuno. (2023). Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models (PRMs) (v1.1). Zenodo. https://doi.org/10.5281/zenodo.10183974
"""
def make_dirs(path):
    import os
    try:
        os.makedirs(path)
    except:
        pass

## read self library
import Library_preprocessing as preprocessing_PRM
import Library_model_construction as construction_PRM

## read library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

# Date time
import datetime

################################################################# Preprocessing
##### define path for dataset
data_name = 'Protolith' # INPUT DATA NAME
data_name = '0_' + data_name
uploaded_file = ['0_Model_Construction_Protolith_Composition_Data/PRM_For_Metabasalt/PetDB_Metabasalt.xlsx', ]

index_col = [0, ]
header = [7, ] # petdb
DataBase = 'petdb'
SAMPLE_INFO = 'Protolith'

'''
*******for dataset caution*******
1. Add "ANALYZED MATERIAL" and write WHOLE ROCK
2. check index col and sample name (if duplicated, the sample will be delete automatically)
'''
################################################################# Preprocessing

###################################################### Model setting
#today_date = '231005_for_normal_model'
today_date = str(datetime.date.today())
PRM_construction_Setting = 'Normal' ## raw->normal, ratio->ratio choice=['Normal', 'Ratio', 'Optional']
Model_algorithm = 'LightGBM'  ### 'LightGBM' or 'NGBoost'
Minimum_combination_number = 4
today_date = today_date + '_' + PRM_construction_Setting + '_' + Model_algorithm+ '_Mincomb_' + str(Minimum_combination_number)

# Define Model Setting
if PRM_construction_Setting == 'Normal':
    feature_setting, Training_Setting = construction_PRM.Model_All_Setting(PRM_construction_Setting, Model_algorithm)
elif PRM_construction_Setting == 'Ratio':
    feature_setting, Training_Setting = construction_PRM.Model_All_Setting(PRM_construction_Setting, Model_algorithm)
elif PRM_construction_Setting == 'Optional':
        ##################################### feature_setting
        setting_X_raw = "on"
        setting_X_log = "off"
        setting_X_product = "on"
        setting_X_product_log = "off"
        setting_X_ratio = "on" ##########
        setting_X_ratio_log = "off"
        #standard_scalerに与える時，X_log or Xを選択
        setting_NORMAL_OR_LOG = "off" #
        setting_PCA = "off"#
        setting_ICA = "off"#
        setting_standard_scaler = "off"
        ##################################### feature_setting

        ##################################### Training settings
        n_trials = 100 ### parameter tuning trial number
        Fold_num = 4 ### model numbers (ensemble)
        test_size=0.2 ### test size → 80% training, 20% test
        random_state = 71 ### define random state
        ##################################### Training settings
        ########### Make class and compile setting in class feature_setting
        feature_setting = construction_PRM.Model_feature_setting(Model_algorithm, setting_X_raw, setting_X_log, setting_X_product, setting_X_product_log, setting_X_ratio, setting_X_ratio_log, setting_NORMAL_OR_LOG, setting_PCA, setting_ICA, setting_standard_scaler)
        Training_Setting = construction_PRM.Model_Training_Setting(n_trials, Fold_num, test_size, random_state)
        ########### Make class and compile setting in class feature_setting
###################################################### Model setting

###################################################### Data read and check duplicates
raw_data=pd.DataFrame()
for num in range(len(uploaded_file)):
    raw_data_num = preprocessing_PRM.read_Raw_data(uploaded_file[num], index_col[num], header[num], DataBase, SAMPLE_INFO)
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
_ = preprocessing_PRM.save_preprocessed_data(path_name, data_name, Whole_rock_RAW, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1)
print("FIN: Preprocessing")
################################################################# Preprocessing

################################################################# Main
#### initial setting-> element
# Normal LightGBM
elem_all = ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'P', 'Nd', 'Zr', 'Ti', 'Y', 'Yb', 'Lu', 'SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO', 'K2O']
mobile_elem_all = ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'P', 'Nd', 'Zr', 'Ti', 'Y', 'Yb', 'Lu', 'SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO', 'K2O']
#immobile_elem_all = ['Zr', 'Th', 'Ti', 'Nb', 'La', 'Ce', 'Nd', 'Yb', 'Lu']
immobile_elem_all = ['Zr', 'Th', 'Ti', 'Nb']
#immobile_elem_all = ['Zr', 'Ti',]

# Ratio
#elem_all = ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'P', 'Nd', 'Zr', 'Ti', 'Y', 'Yb', 'Lu', 'SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO_T', 'K2O', 'Cr'] #'Al', 'Cr'
#mobile_elem_all = ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'P', 'Nd', 'Zr', 'Ti', 'Y', 'Yb', 'Lu', 'SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO_T', 'K2O', 'Cr'] #'Al', 'Cr'
#immobile_elem_all = ['Zr', 'Th', 'Ti', 'Nb', 'Al2O3', 'Cr', ]

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

############### メモリ節約
del Whole_rock_RAW, Whole_rock_cannot_Normalize, Whole_rock_after_Normalize_PM, Whole_rock_after_Normalize_C1
############### メモリ節約

# Initialize for error
error_list_immobile=[]
error_list_mobile=[]

for immobile_elem, mobile_elem_list in zip(immobile_all_list, mobile_all_list):
    immobile_elem = list(immobile_elem)

    for mobile_elem in mobile_elem_list:
        mobile_elem = [mobile_elem]
        try:
            construction_PRM.__main__(path_name, mobile_elem, immobile_elem, Protolith_data, Protolith_location_data, Protolith_location_data, feature_setting, Training_Setting)
        except:
            error_list_immobile.append(str(immobile_elem))
            error_list_mobile.append(str(mobile_elem))

error_list=pd.DataFrame([error_list_immobile, error_list_mobile])
error_list.to_excel(path_name+"/0_error_list.xlsx")
################################################################# Main
