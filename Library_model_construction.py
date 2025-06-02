"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2024/05/10

Author: 松野哲士 (Satoshi Matsuno), Tohoku university, Japan
Contact: satoshi.matsuno.p2@dc.tohoku.ac.jp
Citation: Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022).
App Citation: Satoshi Matsuno. (2023). Graphical interface to construct and apply Machine-learning based Protolith Reconstruction Models (PRMs) (v1.1). Zenodo. https://doi.org/10.5281/zenodo.10183974
"""

#plot&calc
#from msvcrt import LK_LOCK
import pandas as pd
import numpy as np
import streamlit as st

#統計処理
import itertools

#pickle
import pickle

#ファイル管理系
import os
import glob

#ランダム
import random
random.seed(0)

#model
import lightgbm as lgb
import ngboost

#ML related
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# plot
import matplotlib.pyplot as plt

def make_dirs(path):
    try:
        os.makedirs(path)
        print("Correctly make dirs")
    except:
        print("Already exist or fail to make dirs")
################################################# Feature engeneering
### Calculate combination ratio/product
def elem_combinations_list(columns):
    num = 0
    first_list = []
    secound_list = []
    for v in itertools.combinations(columns, 2):
        first_list.append(v[0])
        secound_list.append(v[1])
    return first_list, secound_list

def elem_ratio_list(data):
    ratio_data = pd.DataFrame()
    columns = data.columns
    first_list, secound_list = elem_combinations_list(columns)
    for elem1, elem2 in zip(first_list, secound_list):
        ratio_name_1 = elem1 + "/" + elem2
        ratio_name = elem1 + "/" + elem2
        try:
            first_data = data[elem1]
            secound_data = data[elem2]
            ratio_value = first_data/secound_data
            ratio_data[ratio_name_1] = ratio_value
        except:
            print("Check ratio")
            print(ratio_name)
    return ratio_data

def elem_product_list(data):
    ratio_data = pd.DataFrame()
    columns = data.columns
    first_list, secound_list = elem_combinations_list(columns)
    for elem1, elem2 in zip(first_list, secound_list):
        ratio_name = elem1 + "*" + elem2
        try:
            first_data = data[elem1]
            secound_data = data[elem2]
            ratio_value = first_data*secound_data
            ratio_data[ratio_name] = ratio_value
        except:
            print("Check product")
            print(ratio_name)
    return ratio_data
### Calculate combination ratio/product

### combination
def combination_list(Minimum_combination_number, immobile_elem_all, mobile_elem_all, elem_all):
    ################################# immobile combination
    immobile_all_list = []
    for num in range(Minimum_combination_number, len(immobile_elem_all)+1): # immobile combinationを作って追加していく
        # 組み合わせ
        now_list = list(itertools.combinations(immobile_elem_all, num))
        # 追加
        immobile_all_list.append(now_list)
    immobile_all_list = list(itertools.chain.from_iterable(immobile_all_list)) # listを一次元の行列に変更
    # 重複を消去
    result = []
    for line in immobile_all_list:
        if line not in result:
            result.append(line)
    immobile_all_list = result
    ################################# immobile combination

    ################################# immobile combinationの一つずつに、mobile elem listをつける
    mobile_all_list = []
    num = 0
    for im_elem in immobile_all_list: ## immobile elem combination read
        # indexの形でmobile_elemを読み込む
        now_mobile_elem_all = pd.Series(index=mobile_elem_all).index
        for im in im_elem: # mobile elem listにimmbile elemがある場合、mobile elemから抜く
            try: # ある場合はdrop
                now_mobile_elem_all = now_mobile_elem_all.drop(im)
            except:# ない場合はそのまま
                pass
        num = num + len(now_mobile_elem_all) #　数を数える
        mobile_all_list.append(now_mobile_elem_all)  # 各immobile combinationごとにmobile_all_listをまとめる
    ################################# immobile combinationの一つずつに、mobile elem listをつける
    print(num)
    element_compile = pd.concat([pd.Series(immobile_all_list), pd.Series(mobile_all_list)], axis = 1)
    element_compile.columns = ['immobile', 'mobile']
    return element_compile
################################################################# make combination

################################################################# 重複チェック for Ratio elements
def check_and_modify_duplicates(mobile_list, immobile_list):
    # 重複を確認
    duplicates = set(mobile_list) & set(immobile_list)
    # 辞書を作成
    duplicates_dict = {elem: elem + '_' for elem in duplicates}

    # mobile_list の重複している要素を末尾に '_' を付けて変更
    modified_mobile_list = [elem + '_' if elem in duplicates else elem for elem in mobile_list]

    return list(duplicates), duplicates_dict, modified_mobile_list
################################################################# 重複チェック for Ratio elements


################################################# Define class for model construction
class Models: # pickle for model
    def __init__(self, model, sc, pca, ICA):
        self.model = model
        self.sc = sc
        self.pca = pca
        self.ICA = ICA

class Model_datas: # compile result data
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

class Model_Training_Setting: # Model trainig_Setting
    def __init__(self, n_trials, Fold_num, test_size, random_state):
        self.n_trials = n_trials
        self.Fold_num = Fold_num
        self.test_size = test_size
        self.random_state = random_state
################################################# Define class for model construction

################################################# Model All Setting Class Define
def Model_All_Setting_BASE(PRM_construction_Setting, Model_Training_Process, Model_algorithm):
    # PRM_construction_Settingによって、PRMのconstruction_Settingを定義
    # 各Classを作成する
    if PRM_construction_Setting == 'Normal':
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

    elif PRM_construction_Setting == 'Ratio':
        ##################################### feature_setting
        setting_X_raw = "off"
        setting_X_log = "off"
        setting_X_product = "off"
        setting_X_product_log = "off"
        setting_X_ratio = "on" ##########
        setting_X_ratio_log = "off"
        #standard_scalerに与える時，X_log or Xを選択
        setting_NORMAL_OR_LOG = "off" #
        setting_PCA = "off"#
        setting_ICA = "off"#
        setting_standard_scaler = "off"
        ##################################### feature_setting

    elif PRM_construction_Setting == 'Optional':
        ##################################### feature_setting
        setting_X_raw = "off"
        setting_X_log = "off"
        setting_X_product = "off"
        setting_X_product_log = "off"
        setting_X_ratio = "on" ##########
        setting_X_ratio_log = "off"
        #standard_scalerに与える時，X_log or Xを選択
        setting_NORMAL_OR_LOG = "off" #
        setting_PCA = "off"#
        setting_ICA = "off"#
        setting_standard_scaler = "off"

    ##################################### feature_setting

    if Model_Training_Process == 'Optional':
        ##################################### Training settings
        n_trials = 100 ### parameter tuning trial number
        Fold_num = 4 ### model numbers (ensemble)
        test_size=0.2 ### test size → 80% training, 20% test
        random_state = 71 ### define random state
        ##################################### Training settings
    else:
        ##################################### Training settings
        n_trials = 100 ### parameter tuning trial number
        Fold_num = 4 ### model numbers (ensemble)
        test_size=0.2 ### test size → 80% training, 20% test
        random_state = 71 ### define random state
        ##################################### Training settings


    ########### Make class and compile setting in class feature_setting
    feature_setting = Model_feature_setting(Model_algorithm, setting_X_raw, setting_X_log, setting_X_product, setting_X_product_log, setting_X_ratio, setting_X_ratio_log, setting_NORMAL_OR_LOG, \
                    setting_PCA, setting_ICA, setting_standard_scaler)
    Training_Setting = Model_Training_Setting(n_trials, Fold_num, test_size, random_state)
    ########### Make class and compile setting in class feature_setting
    return feature_setting, Training_Setting
################################################# Model All Setting Class Define

def Model_All_Setting(PRM_construction_Setting, Model_Training_Process, Model_algorithm, Model_ex):
    # PRM_construction_Settingによって、PRMのconstruction_Settingを定義
    # 各Classを作成する
    if PRM_construction_Setting == 'Normal':
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

    elif PRM_construction_Setting == 'Ratio':
        ##################################### feature_setting
        setting_X_raw = "off"
        setting_X_log = "off"
        setting_X_product = "off"
        setting_X_product_log = "off"
        setting_X_ratio = "on" ##########
        setting_X_ratio_log = "off"
        #standard_scalerに与える時，X_log or Xを選択
        setting_NORMAL_OR_LOG = "off" #
        setting_PCA = "off"#
        setting_ICA = "off"#
        setting_standard_scaler = "off"
        ##################################### feature_setting

    elif PRM_construction_Setting == 'Optional':
        ##################################### feature_setting
        setting_X_raw_check = Model_ex.checkbox("setting_X_raw")
        if setting_X_raw_check:
            setting_X_raw = "on"
        else:
            setting_X_raw = "off"

        setting_X_log_check = Model_ex.checkbox("setting_X_log")
        if setting_X_log_check:
            setting_X_log = "on"
        else:
            setting_X_log = "off"

        setting_X_product_check = Model_ex.checkbox("etting_X_product")
        if setting_X_product_check:
            setting_X_product = "on"
        else:
            setting_X_product = "off"

        setting_X_product_log_check = Model_ex.checkbox("setting_X_product_log")
        if setting_X_product_log_check:
            setting_X_product_log = "on"
        else:
            setting_X_product_log = "off"

        setting_X_ratio_check = Model_ex.checkbox("setting_X_ratio")
        if setting_X_ratio_check:
            setting_X_ratio = "on"
        else:
            setting_X_ratio = "off"

        setting_X_ratio_log_check = Model_ex.checkbox("setting_X_ratio_log")
        if setting_X_ratio_log_check:
            setting_X_ratio_log = "on"
        else:
            setting_X_ratio_log = "off"

        setting_NORMAL_OR_LOG_check = Model_ex.checkbox("Usetting_NORMAL_OR_LOG_")
        if setting_NORMAL_OR_LOG_check:
            setting_NORMAL_OR_LOG = "on"
        else:
            setting_NORMAL_OR_LOG = "off"

        setting_PCA_check = Model_ex.checkbox("setting_PCA")
        if setting_PCA_check:
            setting_PCA = "on"
        else:
            setting_PCA = "off"

        setting_ICA_check = Model_ex.checkbox("setting_ICA")
        if setting_ICA_check:
            setting_ICA = "on"
        else:
            setting_ICA = "off"

        setting_standard_scaler_check = Model_ex.checkbox("setting_standard_scaler")
        if setting_standard_scaler_check:
            setting_standard_scaler = "on"
        else:
            setting_standard_scaler = "off"

    ##################################### feature_setting

    if Model_Training_Process == 'Optional':
        Model_ex.caption("Model training setting")
        n_trials = int(Model_ex.slider('n_trials', 0, 1000, 100))
        Fold_num = int(Model_ex.slider('Fold_num', 0, 10, 4))
        test_size = int(Model_ex.slider("test_size (%)", 0, 100, 20))/100
        random_state = int(Model_ex.slider('random_state', 0, 1000, 71))
    else:
        ##################################### Training settings
        n_trials = 100 ### parameter tuning trial number
        Fold_num = 4 ### model numbers (ensemble)
        test_size=0.2 ### test size → 80% training, 20% test
        random_state = 71 ### define random state
        ##################################### Training settings


    ########### Make class and compile setting in class feature_setting
    feature_setting = Model_feature_setting(Model_algorithm, setting_X_raw, setting_X_log, setting_X_product, setting_X_product_log, setting_X_ratio, setting_X_ratio_log, setting_NORMAL_OR_LOG, \
                    setting_PCA, setting_ICA, setting_standard_scaler)
    Training_Setting = Model_Training_Setting(n_trials, Fold_num, test_size, random_state)
    ########### Make class and compile setting in class feature_setting
    return feature_setting, Training_Setting
################################################# Model All Setting Class Define

################################################# Feature engeneering
def feature_making(feature_setting, train_x, test_x, cv_x):
        #特徴量の設定
        #X_allに特徴量をまとめていく
        X_all_train = pd.DataFrame()
        X_all_test = pd.DataFrame()
        X_all_cv = pd.DataFrame()

        if feature_setting.setting_X_raw == "on":
            X_all_train = pd.concat([X_all_train, train_x], axis = 1)
            X_all_test = pd.concat([X_all_test, test_x], axis = 1)
            X_all_cv = pd.concat([X_all_cv, cv_x], axis = 1)
        elif feature_setting.setting_X_log == "on":
            #LOG変換
            train_x_log = train_x.apply(lambda x:np.log10(x))
            test_x_log = test_x.apply(lambda x:np.log10(x))
            cv_x_log = cv_x.apply(lambda x:np.log10(x))

            X_all_train = pd.concat([X_all_train, train_x_log], axis = 1)
            X_all_test = pd.concat([X_all_test, test_x_log], axis = 1)
            X_all_cv = pd.concat([X_all_cv, cv_x_log], axis = 1)

        #Product log変換するかもTest
        if feature_setting.setting_X_product == "on":
            if feature_setting.setting_X_product_log == "on":
                train_x_product = elem_product_list(train_x).apply(lambda x:np.log10(x))
                test_x_product = elem_product_list(test_x).apply(lambda x:np.log10(x))
                cv_x_product = elem_product_list(cv_x).apply(lambda x:np.log10(x))
            else:
                train_x_product = elem_product_list(train_x)
                test_x_product = elem_product_list(test_x)
                cv_x_product = elem_product_list(cv_x)
            X_all_train = pd.concat([X_all_train, train_x_product], axis = 1)
            X_all_test = pd.concat([X_all_test, test_x_product], axis = 1)
            X_all_cv = pd.concat([X_all_cv, cv_x_product], axis = 1)

        #Ratio log変換するかもTest
        if feature_setting.setting_X_ratio == "on":
            if feature_setting.setting_X_ratio_log == "on":
                train_x_ratio = elem_ratio_list(train_x).apply(lambda x:np.log10(x))
                test_x_ratio = elem_ratio_list(test_x).apply(lambda x:np.log10(x))
                cv_x_ratio = elem_ratio_list(cv_x).apply(lambda x:np.log10(x))
            else:
                train_x_ratio = elem_ratio_list(train_x)
                test_x_ratio = elem_ratio_list(test_x)
                cv_x_ratio = elem_ratio_list(cv_x)
            X_all_train = pd.concat([X_all_train, train_x_ratio], axis = 1)
            X_all_test = pd.concat([X_all_test, test_x_ratio], axis = 1)
            X_all_cv = pd.concat([X_all_cv, cv_x_ratio], axis = 1)

        #Standard_scaler&次元削減の初期設定
        #PCA, ICAの変数を決める。Xのlen(変数)で決める
        n_components=len(train_x.columns)
        # PCA・ICAを一元素でやっても意味がないので， 1の時はやらない
        if n_components != 1:
            #PCA, ICAの名前の準備
            PCA_index = []
            ICA_index = []
            for num in range(n_components):
                PCA_index.append("PC"+ str(num + 1))
                ICA_index.append("IC"+ str(num + 1))

            sc = StandardScaler()
            pca = PCA(n_components=n_components, whiten=True)
            ICA = FastICA(n_components=n_components, random_state=30, whiten=True)

            #Standard_scaler&次元削減 #indexを指定するのをわすれない
            #LOG変換されたもの or Not
            if feature_setting.setting_NORMAL_OR_LOG == "on":
                sc.fit(train_x_log)
                s_scaler_train = pd.DataFrame(sc.transform(train_x_log), index = train_x.index)
                s_scaler_test = pd.DataFrame(sc.transform(test_x_log), index = test_x.index)
                s_scaler_cv = pd.DataFrame(sc.transform(cv_x_log), index = cv_x.index)
            else:
                sc.fit(train_x)
                s_scaler_train = pd.DataFrame(sc.transform(train_x), index = train_x.index)
                s_scaler_test = pd.DataFrame(sc.transform(test_x), index = test_x.index)
                s_scaler_cv = pd.DataFrame(sc.transform(cv_x), index = cv_x.index)

            #PCA
            if feature_setting.setting_PCA == "on":
                #Fit & Predict
                pca.fit(s_scaler_train)
                pca_train = pd.DataFrame(pca.fit_transform(s_scaler_train), index = train_x.index, columns=PCA_index)
                pca_test = pd.DataFrame(pca.transform(s_scaler_test), index = test_x.index, columns=PCA_index)
                pca_cv = pd.DataFrame(pca.transform(s_scaler_cv), index = cv_x.index, columns=PCA_index)
                #Concat
                X_all_train = pd.concat([X_all_train, pca_train], axis = 1)
                X_all_test = pd.concat([X_all_test, pca_test], axis = 1)
                X_all_cv = pd.concat([X_all_cv, pca_cv], axis = 1)
            #ICA
            if feature_setting.setting_ICA == "on":
                ICA.fit(s_scaler_train)
                ICA_train = pd.DataFrame(ICA.fit_transform(s_scaler_train), index = train_x.index, columns=ICA_index)
                ICA_test = pd.DataFrame(ICA.transform(s_scaler_test), index = test_x.index, columns=ICA_index)
                ICA_cv = pd.DataFrame(ICA.transform(s_scaler_cv), index = cv_x.index, columns=ICA_index)
                #Concat
                X_all_train = pd.concat([X_all_train, ICA_train], axis = 1)
                X_all_test = pd.concat([X_all_test, ICA_test], axis = 1)
                X_all_cv = pd.concat([X_all_cv, ICA_cv], axis = 1)
        else :
            sc = 0
            pca = 0
            ICA = 0
        # PCA・ICAを一元素でやっても意味がないので，　1の時はやらない

        #再度代入設定
        train_x = X_all_train
        test_x = X_all_test
        cv_x = X_all_cv

        return train_x, test_x, cv_x, sc, pca, ICA
################################################# Feature engeneering

################################################# Model construction
def model_construction(feature_setting, params, train_x_valid, train_y_valid, cv_x, cv_y):
    #Name read
    Model_algorithm = feature_setting.ML_algorithm_name
    #####################モデルごとに推定をして行く
    if Model_algorithm == "LightGBM":
        # Datasetに入れて学習させる
        train_set = lgb.Dataset(train_x_valid, train_y_valid)
        val_set = lgb.Dataset(cv_x, cv_y, reference=train_set)
        # Datasetに入れて学習させる

        ### 学習曲線を記録
        evals_result = {}
        ###　　学習曲線を記録

        ########### model construct & train
        if params == 0:
            print("param-0")
            params = {
                # 基本パラメータ
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'seed': 42,
            }
            model = lgb.train(params, train_set, valid_sets=[train_set, val_set], evals_result=evals_result, valid_names=['train', 'eval'], verbose_eval=True)
        else:
            print("param-1")
            model = lgb.train(params, train_set, valid_sets=[train_set, val_set], evals_result=evals_result, valid_names=['train', 'eval'], verbose_eval=True)
        ########### model construct & train

    elif Model_algorithm == "NGBoost":
        if params == 0:
            Base_model = DecisionTreeRegressor(criterion='squared_error')
            params = {
                "Base":Base_model,
                "random_state":0,
                "verbose":False,
                "Dist":ngboost.distns.Normal,
            }
            model = ngboost.NGBRegressor(**params)
            model.fit(train_x_valid, train_y_valid, X_val=cv_x, Y_val=cv_y,)
        else:
            Base_model = DecisionTreeRegressor(criterion='squared_error',\
                max_depth = params['max_depth'],
                min_samples_leaf = params['min_samples_leaf'],
                min_samples_split = params['min_samples_split'],
                )
            params = {
                "Base":Base_model,
                "random_state":0,
                "verbose":False,
                "Dist":ngboost.distns.Normal,
                "minibatch_frac": params['minibatch_frac']
            }
            model = ngboost.NGBRegressor(**params)
            model.fit(train_x_valid, train_y_valid, X_val=cv_x, Y_val=cv_y,)
        evals_result = [model.evals_result['train']['LOGSCORE']]
    #####################モデルごとに推定をして行く
    return model, evals_result



def model_predict(feature_setting, model, X_data):
    #Name read
    Model_algorithm = feature_setting.ML_algorithm_name
    #####################モデルごとに推定をして行く
    if Model_algorithm == "LightGBM":
        Y_pred =model.predict(X_data) # predict Y
        Y_pred_dist=0 # sigmaは出力できないので０を入れておく

    elif Model_algorithm == "NGBoost":
        Y =model.pred_dist(X_data)# predict Y
        Y_pred = Y.params['loc']# predict Y
        Y_pred_dist = Y.params['scale']# sigma

    return Y_pred, Y_pred_dist


def predict_cv(params, feature_setting, Fold_num, train_x, train_y, test_x, test_y):

    ############ model append list
    models = []
    sc_s = []
    pca_s = []
    ICA_s = []
    cv_indexes = []
    ############ model append list

    # score append
    test_score_append = []
    train_score_append = []
    cv_score_append = []

    # score dist append
    test_score_dist_append = []
    train_score_dist_append = []
    cv_score_dist_append = []

    #predict append
    test_predict_append = []
    train_predict_append = []
    cv_predict_append = []

    #predicted dist
    test_predict_dist_append = []
    train_predict_dist_append = []
    cv_predict_dist_append = []

    # learning curve append
    evals_result_append = {} #dictionary type
    num_models = 0

    #cross_varidationの設定
    kf = KFold(n_splits = Fold_num, shuffle = True, random_state=5)

    #cross_validation
    for train_index, cv_index in kf.split(train_x):

        num_models = num_models + 1

        #Cross_validationによる分離
        train_x_valid, cv_x = train_x.iloc[train_index], train_x.iloc[cv_index]
        train_y_valid, cv_y = train_y.iloc[train_index], train_y.iloc[cv_index]
        test_x_valid = test_x
        test_y_valid = test_y
        #Cross_validationによる分離

        ###########################################特徴量の設定
        train_x_valid, test_x_valid, cv_x, sc, pca, ICA = feature_making(feature_setting, train_x_valid, test_x_valid, cv_x)
        ###########################################特徴量の設定

        ###########################################モデルの作成
        model, evals_result = model_construction(feature_setting, params, train_x_valid, train_y_valid, cv_x, cv_y)
        ###########################################モデルの作成

        ########### model append
        models.append(model)
        sc_s.append(sc)
        pca_s.append(pca)
        ICA_s.append(ICA)
        evals_result_append[num_models] = evals_result #dictionary type
        ########### model append

        #predict
        predicted_train_y, predicted_dist_train_y = model_predict(feature_setting, model, train_x_valid) ## model.predict(example)
        predicted_test_y, predicted_dist_test_y = model_predict(feature_setting, model, test_x_valid)
        predicted_cv_y, predicted_dist_cv_y = model_predict(feature_setting, model, cv_x)

        #predict append
        train_predict_append.append(predicted_train_y)
        test_predict_append.append(predicted_test_y)
        cv_predict_append.append(predicted_cv_y)
        cv_indexes.append(cv_y.index)

        #predict dist append
        train_predict_dist_append.append(predicted_dist_train_y)
        test_predict_dist_append.append(predicted_dist_test_y)
        cv_predict_dist_append.append(predicted_dist_cv_y)

        #score myself
        #実装したスコアを計算
        train_score = np.average(np.sqrt((predicted_train_y - train_y_valid)**2))
        test_score = np.average(np.sqrt((predicted_test_y - test_y_valid)**2))
        cv_score = np.average(np.sqrt((predicted_cv_y - cv_y)**2))

        #スコアを加えていく
        train_score_append.append(train_score)
        test_score_append.append(test_score)
        cv_score_append.append(cv_score)

        #Dist スコアを加えていく ## https://www.statology.org/averaging-standard-deviations/
        train_score_dist_append.append(np.sqrt(np.average((predicted_dist_train_y)**2)))
        test_score_dist_append.append(np.sqrt(np.average((predicted_dist_test_y)**2)))
        cv_score_dist_append.append(np.sqrt(np.average((predicted_dist_cv_y)**2)))

    #テストデータに対する予測値の平均をとる
    pred_test = np.mean(test_predict_append, axis = 0)
    test_predict_dist_append = pd.DataFrame(test_predict_dist_append)**2 # ２乗
    test_predict_dist_append = test_predict_dist_append.mean()
    pred_test_dist = test_predict_dist_append**(1/2)

    #scoreを平均して返す
    scores = [np.mean(train_score_append), np.mean(test_score_append), np.mean(cv_score_append)]
    scores_dist = [np.mean(train_score_dist_append), np.mean(test_score_dist_append), np.mean(cv_score_dist_append)]

    feature_name = train_x_valid.columns

    return models, sc_s, pca_s, ICA_s, scores, scores_dist, pred_test, pred_test_dist, feature_name, evals_result_append
    ################################################# Model construction


################################################################################################ Parameter tuning

## global 変数　→ここにスコアを格納（直接持っていくのが難しいため） 'train', 'cv', 'test'
score_holder_list = []
study_optuna_score =[]
before_score = 10**10
evals_result_append_best = {}

def score_holder_define():
    #初期化
    global score_holder_list
    score_holder_list=[]

########### if文入れると遅くなるので、改良する必要あり→そこまで影響ないかも？
def objective_define(feature_setting, Fold_num, train_x, train_y, test_x, test_y):
    def objective(trial):
        ### parameter selection for model
        Model_algorithm = feature_setting.ML_algorithm_name
        if Model_algorithm == "LightGBM":
            params = {
                # 基本パラメータ
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'seed': 42,

                # 学習用基本パラメータ
                'num_leaves' : trial.suggest_int('num_leaves', 8, 128),
                'max_depth' : trial.suggest_int('max_depth', 2, 10),
                'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 75, 500),

                #'max_bin' : trial.suggest_int('max_bin', 100, 300),
                #'learning_rate' : trial.suggest_loguniform('learning_rate', 0.05, 0.5),
                #'lambda_l1' : trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                #'lambda_l2' : trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                #'feature_fraction' : trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                #'bagging_fraction' : trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                #'bagging_freq' : trial.suggest_int('bagging_freq', 1, 7),
            }
        elif Model_algorithm == "NGBoost":
            params = {
                "max_depth" : trial.suggest_int('max_depth', 2, 10),
                "min_samples_leaf":trial.suggest_int('min_samples_leaf', 8, 128),
                "min_samples_split": trial.suggest_int('min_samples_split', 75, 500),
                "minibatch_frac": trial.suggest_float('minibatch_frac', 1E-16, 1.0)
            }

        #try:
        models, sc_s, pca_s, ICA_s, scores, scores_dist, pred_test, pred_test_dist, feature_name, evals_result_append\
        = predict_cv(params, feature_setting, Fold_num, train_x, train_y, test_x, test_y)
        # scoreを格納
        score_holder_list = scores
        cv_mean_score = scores[2]
        #except:
            #たまにICA convert ERRORがでるので、逃げの処理
        #    cv_mean_score=10*10

        global before_score
        global evals_result_append_best
        if before_score > cv_mean_score:
            evals_result_append_best = evals_result_append
            before_score = cv_mean_score
        return cv_mean_score

    return objective


def print_score(train_mean_score, test_mean_score, cv_mean_score):
    #print_score
    print("トレーニング・平均スコア　：　" + str(train_mean_score))
    print("テスト・平均スコア　：　" + str(test_mean_score))
    print("CV・平均スコア　：　" + str(cv_mean_score))

################################################################################################ Parameter tuning

################################################################################################ Model construction Main + Pickle

#################################################モデルの定義
def predict_model(elem, X_use, y, path_all_share, path_figure_all, feature_setting, Model_Training_Setting):
    #Scoreのデータを全て入れる
    Score_all = pd.Series([], dtype=pd.StringDtype())
    Score_all.name = elem

    print(X_use)

    ############################################元素の決定 Data read
    #データの代入
    X_elem = X_use
    #主要元素はRawで評価する
    #目的関数はLOG空間上で評価したいので，初めからLOG
    y_elem = y[elem].apply(lambda x:np.log10(x)).copy()
    ############################################元素の決定 Data reads

    ##################################### Training settings
    n_trials = Model_Training_Setting.n_trials ### parameter tuning trial number
    Fold_num = Model_Training_Setting.Fold_num ### model numbers (ensemble)
    test_size= Model_Training_Setting.test_size ### test size → 80% training, 20% test
    random_state = Model_Training_Setting.random_state ### define random state
    Model_algorithm = feature_setting.ML_algorithm_name
    ##################################### Training settings

    ##################################### model_featureのsettingを保存
    #pathとフォルダの準備
    path_pkl = "/Model_setting.pkl"
    #pickle
    file = path_all_share + path_pkl
    pickle.dump(feature_setting, open(file, 'wb'))
    ##################################### model_featureのsettingを保存

    ############################################トレーニングデータとテストデータを分ける
    train_x, test_x, train_y, test_y = train_test_split(X_elem, y_elem, test_size=test_size, random_state=random_state, shuffle = True)
    ############################################トレーニングデータとテストデータを分ける

    #################################################################################################################################################### model construct
    ##################################### Initial model construct: CrossVaridationによるアンサンブルモデル作成，Predict，Score
    params = 0
    models, sc_s, pca_s, ICA_s, scores, scores_dist, pred_test, pred_test_dist, feature_name, evals_result_append \
    = predict_cv(params, feature_setting, Fold_num, train_x, train_y, test_x, test_y)
    ##################################### Print score
    #print_score
    print("########### Default")
    #トレーニングの平均スコアを格納
    Score_all["Default_Train_mean"] = scores[0]
    Score_all["Default_Train_Dist_mean"] = scores_dist[0]
    Score_all["Default_Test_mean"] = scores[1]
    Score_all["Default_Test_Dist_mean"] = scores_dist[1]
    Score_all["Default_CV_mean"] = scores[2]
    Score_all["Default_CV_Dist_mean"] = scores_dist[2]
    print_score(scores[0], scores[1], scores[2])
    print_score(scores_dist[0], scores_dist[1], scores_dist[2])

    ##################################### Print score
    ##################################### Initial model construct: CrossVaridationによるアンサンブルモデル作成，Predict，Score

    ########################################################################### Optuna optimize
    # train, CV, and test score compile in this global dataframe 初期化
    score_holder_define()
    global before_score # 初期化
    before_score = 10**10  # 初期化
    global evals_result_append_best
    evals_result_append_best={}
    ############################optunaによるベイズ最適化
    study = optuna.create_study() #load_if_exists=False 再度学習する場合，設定を変えている場合がほとんどだから。
    study.optimize(objective_define(feature_setting, Fold_num, train_x, train_y, test_x, test_y), n_trials=n_trials)

    #最適化の過程を可視化
    study_optuna_score = pd.DataFrame(data=score_holder_list, columns=['train', 'cv', 'test'], index=range(0, n_trials))
    study_optuna_score.plot()
    plt.savefig(path_all_share + "/Score_training_optimize.pdf", bbox_inches='tight')
    plt.close()
    plt.show()

    #trialを保存
    study.trials_dataframe().to_excel(path_all_share + "/Optune_trials.xlsx")
    study_optuna_score.to_excel(path_all_share + "/Optune_trials_ALL_score.xlsx")
    ###best_paramをread
    best_params = study.best_params
    ############################optunaによるベイズ最適化

    #########################optunaによるベイズ最適化されたparamで計算
    models, sc_s, pca_s, ICA_s, scores, scores_dist, pred_test, pred_test_dist, feature_name, evals_result_append\
    = predict_cv(best_params, feature_setting, Fold_num, train_x, train_y, test_x, test_y)
    #########################optunaによるベイズ最適化されたparamで計算
    ########################################################################### Optuna optimize

    ##################################### Print score
    #print_score
    print("########### Optune_tuned")
    #トレーニングの平均スコアを格納
    Score_all["Optuna_Train_mean"] = scores[0]
    Score_all["Optuna_Train_Dist_mean"] = scores_dist[0]
    Score_all["Optuna_Test_mean"] = scores[1]
    Score_all["Optuna_Test_Dist_mean"] = scores_dist[1]
    Score_all["Optuna_CV_mean"] = scores[2]
    Score_all["Optuna_CV_Dist_mean"] = scores_dist[2]
    print_score(scores[0], scores[1], scores[2])
    print_score(scores_dist[0], scores_dist[1], scores_dist[2])
    ##################################### Print score
    #########score 保存
    Score_all.to_excel(path_all_share + "/Score_all.xlsx")
    ########################################################################### Model saving
    path_dir_model = path_all_share +"/Models"
    make_dirs(path_dir_model)
    for fold_num in range(Fold_num):
        model_now = models[fold_num]
        sc_now = sc_s[fold_num]
        pca_now = pca_s[fold_num]
        ICA_now = ICA_s[fold_num]

        class_model = Models(model_now, sc_now, pca_now, ICA_now)
        #pathとフォルダの準備
        path_pkl = "/"+ str(fold_num) +".pkl"
        #pickle
        file = path_dir_model + path_pkl
        pickle.dump(class_model, open(file, 'wb'))
    ########################################################################### Model saving
    model_all = Models(models, sc_s, pca_s, ICA_s)
    #################################################################################################################################################### model construct

    ################################################################パラメータ・特徴量
    print("パラメータ・特徴量")
    #importance
    num_importance = 1
    importance_compile_gain = pd.DataFrame()
    importance_compile_split = pd.DataFrame()
    evals_result_compile = pd.DataFrame()

    for best_model in model_all.model:
        if Model_algorithm == 'LightGBM':
            #################################### learning curve
            lgb.plot_metric(evals_result_append_best[num_importance])
            #path
            path_now_fig_name = "/learning_curve_" + str(num_importance) + ".pdf"
            #save
            plt.savefig(path_figure_all + path_now_fig_name, bbox_inches='tight')
            plt.close()
            plt.show()

            #DataFrameに入れる
            evals_result_compile['learning_curve_'+str(num_importance)+'_train'] = evals_result_append_best[num_importance]['train']['rmse']
            evals_result_compile['learning_curve_'+str(num_importance)+'_eval'] = evals_result_append_best[num_importance]['eval']['rmse']
            #################################### learning curve

            #################################### gain
            importances = pd.Series(best_model.feature_importance(importance_type="gain"), index = feature_name)
            #DataFrameに入れる
            importance_compile_gain['model_'+str(num_importance)] = importances
            #importance取得
            lgb.plot_importance(best_model, importance_type="gain", figsize=(12, 12))
            #path
            path_now_fig_name = "/importance_model_gain_" + str(num_importance) + ".pdf"
            #save
            plt.savefig(path_figure_all + path_now_fig_name, bbox_inches='tight')
            plt.close()
            plt.show()
            #################################### gain

            #################################### split
            importances = pd.Series(best_model.feature_importance(importance_type="split"), index = feature_name)
            #DataFrameに入れる
            importance_compile_split['model_'+str(num_importance)] = importances
            #importance取得
            lgb.plot_importance(best_model, importance_type="split", figsize=(12, 12))
            #path
            path_now_fig_name = "/importance_model_split_" + str(num_importance) + ".pdf"
            #save
            plt.savefig(path_figure_all + path_now_fig_name, bbox_inches='tight')
            plt.close()
            plt.show()
            #################################### split
            #名前change
            num_importance = num_importance + 1
        elif Model_algorithm == 'NGBoost':
            pass
    # save importance
    importance_compile_gain.to_excel(path_all_share + "/feature_importance_gain.xlsx")
    importance_compile_split.to_excel(path_all_share + "/feature_importance_split.xlsx")
    evals_result_compile.to_excel(path_all_share + "/learning_curve_result_compile.xlsx")
    ################################################################パラメータ・特徴量

    ########################################関数の定義
    #Scoreのデータを全て入れる #上で定義済み
    #test_誤差データ
    test_error_all = pd.DataFrame()
    #test_y raw
    if Model_algorithm == 'NGBoost':
        pred_test_dist.index = test_y.index # indexを指定してやる
    #######################テストデータの誤差計算
    #誤差の計算
    score_error_freq = (pd.Series(pred_test).apply(lambda x: 10**x).values / pd.Series(test_y).apply(lambda x: 10**x).values)*100
    #誤差を格納していく
    test_error_all[elem] = pd.Series(score_error_freq, index = test_y.index)
    test_error_all.to_excel(path_all_share + "/test_error_all.xlsx")

    #######################testデータ
    #この際のtest_yを代入していく
    test_data_all = pd.concat([test_y.apply(lambda x: 10**x), test_x], axis = 1)
    test_data_all["RAW"] = test_y.apply(lambda x: 10**x)
    test_data_all["predict"] = pd.Series(pred_test, index = test_y.index).apply(lambda x: 10**x)
    if Model_algorithm == 'NGBoost':
        test_data_all["predict_Dist"] = pred_test_dist
    test_data_all.to_excel(path_all_share + "/test_data_all.xlsx")

    test_data_all.plot.scatter(x='RAW', y='predict', figsize=(3, 3), alpha=0.05)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(path_all_share + "/test_data_scatter.pdf", bbox_inches='tight')
    plt.close()
    plt.show()

    if Model_algorithm == 'NGBoost': ### error bar付きの散布図
        # figure
        test_data_all = pd.concat([test_y, test_x], axis = 1)
        test_data_all["RAW"] = test_y
        test_data_all["predict"] = pd.Series(pred_test, index = test_y.index)
        test_data_all["predict_Dist"] = pred_test_dist
        test_data_all.to_excel(path_all_share + "/test_data_all_dist.xlsx")

        plt.figure(figsize=(3, 3))
        plt.errorbar(test_data_all["RAW"], test_data_all["predict"], yerr = test_data_all["predict_Dist"], capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w', alpha=0.05)
        plt.savefig(path_all_share + "/test_data_scatter_dist.pdf", bbox_inches='tight')
        plt.close()
        plt.show()

    return model_all, Score_all, test_error_all, test_data_all

################################################################################################ Model construction Main + Pickle

################################################################################################ Main
def __main__(path_name, mobile_elem, immobile_elem, Protolith_data, Protolith_loc_data, Protolith_loc_data_raw, feature_setting, Training_Setting):
    ########################################################元素の数を計算
    #不動元素
    #一回全て代入
    use_element = mobile_elem + immobile_elem
    #######################################################ディレクトリの作成
    #pathとフォルダの準備
    path_1 = path_name
    path_2 = str(immobile_elem).strip("[").strip("]").strip("'")
    path_3 = "/" + str(mobile_elem).strip("[").strip("]").strip("'")
    #全体のpathの設定
    path_all_share = path_1 + path_2 + path_3

    ############ディレクトリが存在するかをチェック
    directry_exist = glob.glob(path_1 + "/*/*", recursive=True)
    flag_for_folder_exist = 0
    for folder_name in directry_exist:
        if path_all_share == folder_name:
            print("exist")
            flag_for_folder_exist = 1
            break
        else:
            pass

    ########### 何回でも作り直すか（0）、一度作ったモデルはそのままか（1） exist=>Default=1
    # 上記で確認しているので、ここで下を外せば必ず作ることになる。
    #flag_for_folder_exist = 0
    ###########

    ##########################################################################モデルの作成
    if flag_for_folder_exist == 0:
        ########################################################サンプル数を計算
        index_names = Protolith_data[use_element].dropna().index
        Sample_info = Protolith_loc_data.T[index_names].T
        Sample_info_num = Sample_info.groupby('SAMPLE_INFO').size()
        Sample_info_num_sum = Sample_info.groupby('SAMPLE_INFO').size().sum()
        Sample_DataBase_num = Sample_info.groupby('DataBase').size()
        ########################################################サンプル数を計算

        #######################################################データinfoの保存
        #Check point
        Data_info = pd.Series([], dtype=pd.StringDtype())
        print("####################")
        print("Element number : " + str(len(use_element)))
        print("Mobile elem : " + str(mobile_elem))
        print("Mobile elem : " + str(len(mobile_elem)))
        print("Immobile elem : " + str(immobile_elem))
        print("Immobile elem : " + str(len(immobile_elem)))
        print("####################")
        print("Sample_Tectonic_Setting : ")
        print(Sample_info_num)
        print("Sample_num : " + str(Sample_info_num_sum))
        print("Sample_DataBase : ")
        print(Sample_DataBase_num)
        print("####################")

        Data_info["Element number"] = len(use_element)
        Data_info["Mobile elem"] = mobile_elem
        Data_info["Mobile elem num"] =  len(mobile_elem)
        Data_info["Immobile elem : "] = immobile_elem
        Data_info["Immobile elem num:"] = len(immobile_elem)
        Data_info = pd.concat([Data_info, Sample_info_num])
        Data_info["SUM num "] = Sample_info_num_sum
        Data_info = pd.concat([Data_info, Sample_DataBase_num])

        #use_elemの全体ディレクトリの作成
        make_dirs(path_all_share)
        #図を保存するディレクトリの作成
        path_4 = "/Figure"
        path_figure_all = path_all_share + path_4
        make_dirs(path_figure_all)
        #######################################################データinfoの保存
        Data_info.to_excel(path_all_share + "/Data_Info.xlsx")
        #######################################################データinfoの保存


        #######################################################使うデータの整理
        #目的yと入力データXを設定
        data___ = Protolith_data[use_element].dropna()
        print(Protolith_data[use_element].shape)
        print(data___.shape)
        #X_use
        X_use = data___[immobile_elem]
        #y 目的変数
        y = data___[mobile_elem]
        #Location_data
        sample_name_list = y.index
        #Protolith Location Dataのread
        Protolith_loc_data = Protolith_loc_data_raw.T[sample_name_list].T
        Protolith_loc_data["Sample_name"] = Protolith_loc_data.index
        #######################################################使うデータの整理

        #今回の流体移動元素の定義
        elem = mobile_elem[0]
        ########################################MODEL
        model_all, Score_all, test_error_all, test_data_all = predict_model(elem, X_use, y,  path_all_share, path_figure_all, feature_setting, Training_Setting)
        ########################################MODEL  fin

        ############################################dataの保存
        class_data = Model_datas(Score_all, test_error_all, test_data_all)
        #pathとフォルダの準備
        path_3 = "/Used_all_data.pkl"
        path_folder = path_all_share + path_3
        #pickle
        file = path_folder
        pickle.dump(class_data, open(file, 'wb'))
        #######################################################モデルの作成fin
################################################################################################ Main
