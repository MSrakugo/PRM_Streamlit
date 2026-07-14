"""
Begin: Tue Mar  1 23:06:08 2022
Final update: 2026/02/08

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


ADDED_EVALUATION_OUTPUT_FILES = (
    "Model_evaluation_scores_log10.xlsx",
    "Optuna_trials.xlsx",
    "Optuna_trials_all_scores.xlsx",
    "Optuna_training_scores.pdf",
    "Early_stopping_summary.xlsx",
    "learning_curve_result_compile_default.xlsx",
    "learning_curve_result_compile_optuna.xlsx",
    "learning_curve_default_<fold>.pdf",
    "learning_curve_optuna_<fold>.pdf",
    "learning_curve_selected_<fold>.pdf",
)


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


def make_ratio_duplicate_columns(data, ratio_element_names):
    """Create Ratio target aliases from already-preprocessed columns."""
    ratio_element_names = [
        element_name
        for element_name in ratio_element_names
        if element_name.endswith("_")
    ]
    source_element_names = [
        element_name.removesuffix("_")
        for element_name in ratio_element_names
    ]

    duplicate_columns = data[source_element_names].copy()
    duplicate_columns.columns = ratio_element_names
    return duplicate_columns


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

class Model_Optimization_Setting:
    """Shared, replaceable search ranges and early-stopping settings."""

    def __init__(
        self,
        lightgbm_search_space=None,
        ngboost_search_space=None,
        lightgbm_max_boost_rounds=3000,
        lightgbm_early_stopping_rounds=100,
        ngboost_max_estimators=2000,
        ngboost_early_stopping_rounds=50,
    ):
        self.lightgbm_search_space = {
            "max_depth": (2, 10),
            "num_leaves": (4, 256),
            "learning_rate": (0.005, 0.1),
            "min_data_in_leaf_fraction": (0.0, 0.15),
            "min_data_in_leaf_minimum": 20,
            "feature_fraction": (0.5, 1.0),
            "bagging_fraction": (0.5, 1.0),
            "lambda_l2": (1e-8, 100.0),
        }
        self.ngboost_search_space = {
            "max_depth": (1, 8),
            "learning_rate": (0.005, 0.1),
            "min_samples_leaf_fraction": (0.0, 0.15),
            "min_samples_leaf_minimum": 8,
            "minibatch_frac": (0.5, 1.0),
            "col_sample": (0.5, 1.0),
        }
        if lightgbm_search_space is not None:
            self.lightgbm_search_space.update(lightgbm_search_space)
        if ngboost_search_space is not None:
            self.ngboost_search_space.update(ngboost_search_space)

        self.lightgbm_max_boost_rounds = int(lightgbm_max_boost_rounds)
        self.lightgbm_early_stopping_rounds = int(lightgbm_early_stopping_rounds)
        self.ngboost_max_estimators = int(ngboost_max_estimators)
        self.ngboost_early_stopping_rounds = int(ngboost_early_stopping_rounds)
        if min(
            self.lightgbm_max_boost_rounds,
            self.lightgbm_early_stopping_rounds,
            self.ngboost_max_estimators,
            self.ngboost_early_stopping_rounds,
        ) <= 0:
            raise ValueError("Iteration and early-stopping settings must be positive")


class Model_Training_Setting: # Model trainig_Setting
    def __init__(self, n_trials, Fold_num, test_size, random_state, optimization_setting=None):
        self.n_trials = n_trials
        self.Fold_num = Fold_num
        self.test_size = test_size
        self.random_state = random_state
        self.optimization_setting = optimization_setting or Model_Optimization_Setting()
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
        X_all_test = pd.DataFrame() if test_x is not None else None
        X_all_cv = pd.DataFrame()

        if feature_setting.setting_X_raw == "on":
            X_all_train = pd.concat([X_all_train, train_x], axis = 1)
            if test_x is not None:
                X_all_test = pd.concat([X_all_test, test_x], axis = 1)
            X_all_cv = pd.concat([X_all_cv, cv_x], axis = 1)
        elif feature_setting.setting_X_log == "on":
            #LOG変換
            train_x_log = train_x.apply(lambda x:np.log10(x))
            test_x_log = test_x.apply(lambda x:np.log10(x)) if test_x is not None else None
            cv_x_log = cv_x.apply(lambda x:np.log10(x))

            X_all_train = pd.concat([X_all_train, train_x_log], axis = 1)
            if test_x_log is not None:
                X_all_test = pd.concat([X_all_test, test_x_log], axis = 1)
            X_all_cv = pd.concat([X_all_cv, cv_x_log], axis = 1)

        #Product log変換するかもTest
        if feature_setting.setting_X_product == "on":
            if feature_setting.setting_X_product_log == "on":
                train_x_product = elem_product_list(train_x).apply(lambda x:np.log10(x))
                test_x_product = elem_product_list(test_x).apply(lambda x:np.log10(x)) if test_x is not None else None
                cv_x_product = elem_product_list(cv_x).apply(lambda x:np.log10(x))
            else:
                train_x_product = elem_product_list(train_x)
                test_x_product = elem_product_list(test_x) if test_x is not None else None
                cv_x_product = elem_product_list(cv_x)
            X_all_train = pd.concat([X_all_train, train_x_product], axis = 1)
            if test_x_product is not None:
                X_all_test = pd.concat([X_all_test, test_x_product], axis = 1)
            X_all_cv = pd.concat([X_all_cv, cv_x_product], axis = 1)

        #Ratio log変換するかもTest
        if feature_setting.setting_X_ratio == "on":
            if feature_setting.setting_X_ratio_log == "on":
                train_x_ratio = elem_ratio_list(train_x).apply(lambda x:np.log10(x))
                test_x_ratio = elem_ratio_list(test_x).apply(lambda x:np.log10(x)) if test_x is not None else None
                cv_x_ratio = elem_ratio_list(cv_x).apply(lambda x:np.log10(x))
            else:
                train_x_ratio = elem_ratio_list(train_x)
                test_x_ratio = elem_ratio_list(test_x) if test_x is not None else None
                cv_x_ratio = elem_ratio_list(cv_x)
            X_all_train = pd.concat([X_all_train, train_x_ratio], axis = 1)
            if test_x_ratio is not None:
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
                s_scaler_test = pd.DataFrame(sc.transform(test_x_log), index = test_x.index) if test_x_log is not None else None
                s_scaler_cv = pd.DataFrame(sc.transform(cv_x_log), index = cv_x.index)
            else:
                sc.fit(train_x)
                s_scaler_train = pd.DataFrame(sc.transform(train_x), index = train_x.index)
                s_scaler_test = pd.DataFrame(sc.transform(test_x), index = test_x.index) if test_x is not None else None
                s_scaler_cv = pd.DataFrame(sc.transform(cv_x), index = cv_x.index)

            #PCA
            if feature_setting.setting_PCA == "on":
                #Fit & Predict
                pca.fit(s_scaler_train)
                pca_train = pd.DataFrame(pca.fit_transform(s_scaler_train), index = train_x.index, columns=PCA_index)
                pca_test = pd.DataFrame(pca.transform(s_scaler_test), index = test_x.index, columns=PCA_index) if s_scaler_test is not None else None
                pca_cv = pd.DataFrame(pca.transform(s_scaler_cv), index = cv_x.index, columns=PCA_index)
                #Concat
                X_all_train = pd.concat([X_all_train, pca_train], axis = 1)
                if pca_test is not None:
                    X_all_test = pd.concat([X_all_test, pca_test], axis = 1)
                X_all_cv = pd.concat([X_all_cv, pca_cv], axis = 1)
            #ICA
            if feature_setting.setting_ICA == "on":
                ICA.fit(s_scaler_train)
                ICA_train = pd.DataFrame(ICA.fit_transform(s_scaler_train), index = train_x.index, columns=ICA_index)
                ICA_test = pd.DataFrame(ICA.transform(s_scaler_test), index = test_x.index, columns=ICA_index) if s_scaler_test is not None else None
                ICA_cv = pd.DataFrame(ICA.transform(s_scaler_cv), index = cv_x.index, columns=ICA_index)
                #Concat
                X_all_train = pd.concat([X_all_train, ICA_train], axis = 1)
                if ICA_test is not None:
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
def _ngboost_rmse_monitor(distribution, targets, sample_weight=None):
    """Evaluate the NGBoost location prediction with the primary log10 RMSE."""
    locations = np.asarray(distribution.params["loc"], dtype=float)
    targets = np.asarray(targets, dtype=float)
    squared_errors = (locations - targets) ** 2
    if sample_weight is None:
        return np.sqrt(np.mean(squared_errors))
    return np.sqrt(np.average(squared_errors, weights=sample_weight))


def _restore_ngboost_best_iteration(model):
    """Remove post-best boosting stages so ordinary saved-model prediction is safe."""
    actual_iterations = len(model.base_models)
    best_index = getattr(model, "best_val_loss_itr", None)
    best_iteration = actual_iterations if best_index is None else int(best_index) + 1
    best_iteration = max(1, min(best_iteration, actual_iterations))

    for attribute in ("base_models", "scalings", "col_idxs"):
        values = getattr(model, attribute)
        setattr(model, attribute, values[:best_iteration])
    model.n_estimators = best_iteration
    return best_iteration, actual_iterations


def model_construction(
    feature_setting,
    params,
    train_x_valid,
    train_y_valid,
    cv_x,
    cv_y,
    optimization_setting=None,
):
    optimization_setting = optimization_setting or Model_Optimization_Setting()
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
        ###　学習曲線を記録

        # コールバックを定義
        callbacks_list = [
            lgb.callback.record_evaluation(evals_result),  # 評価結果を格納するため
            lgb.early_stopping(
                stopping_rounds=optimization_setting.lightgbm_early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            ),
        ]

        # verbose_eval の機能を実現 modified ver 250603 for LightGBM 4.0以上に対応
        # verbose_eval=True の場合、各イテレーションで出力:
        callbacks_list.append(lgb.callback.log_evaluation(period=50)) # 50回おきにイテレーションを出力

        ########### model construct & train　modified ver 250603 for LightGBM 4.0以上に対応
        if params == 0:
            print("param-0")
            params = {
                # 基本パラメータ
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'seed': 42,
                'verbosity': -1,
            }
            model = lgb.train(params,
                            train_set,
                            num_boost_round=optimization_setting.lightgbm_max_boost_rounds,
                            valid_sets=[train_set, val_set],
                            callbacks=callbacks_list,
                            valid_names=['train', 'eval'])
        else:
            print("param-1")
            # study.best_paramsにはOptunaがsuggestした値しか入らないため、
            # 探索中に使用した固定パラメータを最終再学習でも必ず補完する。
            lightgbm_params = {
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": "rmse",
                "seed": 42,
                "bagging_freq": 1,
                "verbosity": -1,
            }
            lightgbm_params.update(params)
            model = lgb.train(lightgbm_params,
                            train_set,
                            num_boost_round=optimization_setting.lightgbm_max_boost_rounds,
                            valid_sets=[train_set, val_set],
                            callbacks=callbacks_list,
                            valid_names=['train', 'eval'])
        ########### model construct & train
        model.prm_training_metadata = {
            "best_iteration": int(model.best_iteration),
            "stopping_iteration": len(evals_result["eval"]["rmse"]),
            "max_iteration": optimization_setting.lightgbm_max_boost_rounds,
            "stopping_metric": "rmse_log10",
        }

    elif Model_algorithm == "NGBoost":
        if params == 0:
            Base_model = DecisionTreeRegressor(criterion='squared_error')
            params = {
                "Base":Base_model,
                "random_state":0,
                "verbose":False,
                "Dist":ngboost.distns.Normal,
                "n_estimators":optimization_setting.ngboost_max_estimators,
            }
            model = ngboost.NGBRegressor(**params)
            model.fit(
                train_x_valid,
                train_y_valid,
                X_val=cv_x,
                Y_val=cv_y,
                train_loss_monitor=_ngboost_rmse_monitor,
                val_loss_monitor=_ngboost_rmse_monitor,
                early_stopping_rounds=optimization_setting.ngboost_early_stopping_rounds,
            )
        else:
            min_samples_leaf = params['min_samples_leaf']
            Base_model = DecisionTreeRegressor(criterion='squared_error',\
                max_depth = params['max_depth'],
                min_samples_leaf = min_samples_leaf,
                min_samples_split = params.get(
                    'min_samples_split',
                    max(2, 2 * min_samples_leaf),
                ),
                )
            params = {
                "Base":Base_model,
                "random_state":0,
                "verbose":False,
                "Dist":ngboost.distns.Normal,
                "minibatch_frac": params['minibatch_frac'],
                "col_sample": params.get('col_sample', 1.0),
                "learning_rate": params.get('learning_rate', 0.01),
                "n_estimators":optimization_setting.ngboost_max_estimators,
            }
            model = ngboost.NGBRegressor(**params)
            model.fit(
                train_x_valid,
                train_y_valid,
                X_val=cv_x,
                Y_val=cv_y,
                train_loss_monitor=_ngboost_rmse_monitor,
                val_loss_monitor=_ngboost_rmse_monitor,
                early_stopping_rounds=optimization_setting.ngboost_early_stopping_rounds,
            )
        metric_name = next(iter(model.evals_result["train"]))
        evals_result = {
            "train": {"rmse": model.evals_result["train"][metric_name]},
            "eval": {"rmse": model.evals_result["val"][metric_name]},
        }
        best_iteration, actual_iterations = _restore_ngboost_best_iteration(model)
        model.prm_training_metadata = {
            "best_iteration": best_iteration,
            "stopping_iteration": actual_iterations,
            "max_iteration": optimization_setting.ngboost_max_estimators,
            "stopping_metric": "rmse_log10",
        }
    #####################モデルごとに推定をして行く
    return model, evals_result



def model_predict(feature_setting, model, X_data):
    #Name read
    Model_algorithm = feature_setting.ML_algorithm_name
    #####################モデルごとに推定をして行く
    if Model_algorithm == "LightGBM":
        best_iteration = getattr(model, "best_iteration", None)
        Y_pred =model.predict(X_data, num_iteration=best_iteration) # predict Y
        Y_pred_dist=0 # sigmaは出力できないので０を入れておく

    elif Model_algorithm == "NGBoost":
        Y =model.pred_dist(X_data)# predict Y
        Y_pred = Y.params['loc']# predict Y
        Y_pred_dist = Y.params['scale']# sigma

    return Y_pred, Y_pred_dist


def calculate_ensemble_prediction_and_rmse(test_predictions, test_y):
    """Average fold predictions and score that same ensemble in log10 space."""
    pred_test = np.mean(np.asarray(test_predictions, dtype=float), axis=0)
    y_test = np.asarray(test_y, dtype=float)
    test_score = np.sqrt(np.mean((pred_test - y_test) ** 2))
    return pred_test, test_score


def calculate_oof_rmse(oof_predictions, train_y):
    """Score one out-of-fold prediction per training sample in log10 space."""
    predictions = np.asarray(oof_predictions, dtype=float)
    targets = np.asarray(train_y, dtype=float)

    if predictions.shape != targets.shape:
        raise ValueError("oof_predictions and train_y must have the same shape")

    errors = predictions - targets
    if not np.all(np.isfinite(errors)):
        return 1e10

    errors = np.clip(errors, -1e10, 1e10)
    return np.sqrt(np.mean(errors ** 2))


def predict_cv(
    params,
    feature_setting,
    Fold_num,
    train_x,
    train_y,
    test_x=None,
    test_y=None,
    optimization_setting=None,
):

    if (test_x is None) != (test_y is None):
        raise ValueError("test_x and test_y must be provided together")
    evaluate_holdout = test_x is not None
    optimization_setting = optimization_setting or Model_Optimization_Setting()

    ############ model append list
    models = []
    sc_s = []
    pca_s = []
    ICA_s = []
    ############ model append list

    # score append
    train_score_append = []

    # score dist append
    test_score_dist_append = []
    train_score_dist_append = []
    cv_score_dist_append = []

    #predict append
    test_predict_append = []
    train_predict_append = []
    oof_predictions = np.full(len(train_y), np.nan, dtype=float)

    #predicted dist
    test_predict_dist_append = []
    train_predict_dist_append = []
    cv_predict_dist_append = []

    # learning curve append
    evals_result_append = {} #dictionary type
    training_metadata = []
    num_models = 0

    #cross_varidationの設定
    kf = KFold(n_splits = Fold_num, shuffle = True, random_state=5)

    #cross_validation
    for train_index, cv_index in kf.split(train_x):

        num_models = num_models + 1

        #Cross_validationによる分離
        train_x_valid, cv_x = train_x.iloc[train_index], train_x.iloc[cv_index]
        train_y_valid, cv_y = train_y.iloc[train_index], train_y.iloc[cv_index]
        test_x_valid = test_x if evaluate_holdout else None
        #Cross_validationによる分離

        ###########################################特徴量の設定
        train_x_valid, test_x_valid, cv_x, sc, pca, ICA = feature_making(feature_setting, train_x_valid, test_x_valid, cv_x)
        ###########################################特徴量の設定

        ###########################################モデルの作成
        model, evals_result = model_construction(
            feature_setting,
            params,
            train_x_valid,
            train_y_valid,
            cv_x,
            cv_y,
            optimization_setting=optimization_setting,
        )
        ###########################################モデルの作成

        ########### model append
        models.append(model)
        sc_s.append(sc)
        pca_s.append(pca)
        ICA_s.append(ICA)
        evals_result_append[num_models] = evals_result #dictionary type
        model_metadata = dict(getattr(model, "prm_training_metadata", {}))
        model_metadata["fold"] = num_models
        model_metadata["algorithm"] = feature_setting.ML_algorithm_name
        training_metadata.append(model_metadata)
        ########### model append

        #predict
        predicted_train_y, predicted_dist_train_y = model_predict(feature_setting, model, train_x_valid) ## model.predict(example)
        if evaluate_holdout:
            predicted_test_y, predicted_dist_test_y = model_predict(feature_setting, model, test_x_valid)
        predicted_cv_y, predicted_dist_cv_y = model_predict(feature_setting, model, cv_x)

        #predict append
        train_predict_append.append(predicted_train_y)
        if evaluate_holdout:
            test_predict_append.append(predicted_test_y)
        # KFoldの位置indexを使い、各trainサンプルのOOF予測を一度だけ格納する。
        # DataFrame/Seriesのラベルindexが重複していても復元順には影響しない。
        oof_predictions[cv_index] = np.asarray(predicted_cv_y, dtype=float)

        #predict dist append
        train_predict_dist_append.append(predicted_dist_train_y)
        if evaluate_holdout:
            test_predict_dist_append.append(predicted_dist_test_y)
        cv_predict_dist_append.append(predicted_dist_cv_y)

        #score myself
        ######################################################################
        # スコアを計算
        # 実装が間違っていた: 修正, 以前のものはMAEの式になっていた。
        # 以下の実装
        # train_score = np.average(np.sqrt((predicted_train_y - train_y_valid)**2))
        # test_score = np.average(np.sqrt((predicted_test_y - test_y_valid)**2))
        # cv_score = np.average(np.sqrt((predicted_cv_y - cv_y)**2))

        ############################################################
        # RMSE (Safe implementation)
        ############################################################
        p_tr = np.asarray(predicted_train_y, dtype=float)
        y_tr = train_y_valid.to_numpy(dtype=float)
        e_tr = p_tr - y_tr
        train_score = np.sqrt(np.mean(e_tr ** 2))

        ######################################################################

        #スコアを加えていく
        train_score_append.append(train_score)

        #Dist スコアを加えていく ## https://www.statology.org/averaging-standard-deviations/
        train_score_dist_append.append(np.sqrt(np.average((predicted_dist_train_y)**2)))
        if evaluate_holdout:
            test_score_dist_append.append(np.sqrt(np.average((predicted_dist_test_y)**2)))
        cv_score_dist_append.append(np.sqrt(np.average((predicted_dist_cv_y)**2)))

    pred_test = None
    test_score = None
    pred_test_dist = None
    test_score_dist = None
    if evaluate_holdout:
        # fold予測の平均を最終予測とし、同じ予測値に対してTest RMSEを計算する
        pred_test, test_score = calculate_ensemble_prediction_and_rmse(
            test_predict_append,
            test_y,
        )
        test_predict_dist_append = pd.DataFrame(test_predict_dist_append)**2 # ２乗
        test_predict_dist_append = test_predict_dist_append.mean()
        pred_test_dist = test_predict_dist_append**(1/2)
        test_score_dist = np.mean(test_score_dist_append)

    # CV scoreは、全trainサンプルのOOF予測に対して一度だけ計算するlog10 RMSE
    cv_score = calculate_oof_rmse(oof_predictions, train_y)

    # Test scoreは、保存・利用される最終アンサンブル予測のlog10 RMSE
    scores = [np.mean(train_score_append), test_score, cv_score]
    scores_dist = [np.mean(train_score_dist_append), test_score_dist, np.mean(cv_score_dist_append)]
    evals_result_append["_training_metadata"] = training_metadata

    feature_name = train_x_valid.columns

    return models, sc_s, pca_s, ICA_s, scores, scores_dist, pred_test, pred_test_dist, feature_name, evals_result_append
    ################################################# Model construction


################################################################################################ Parameter tuning

def _fractional_integer_bounds(sample_count, fractions, minimum, maximum):
    lower = max(minimum, int(np.ceil(sample_count * fractions[0])))
    upper = max(lower, int(np.ceil(sample_count * fractions[1])))
    upper = min(upper, maximum)
    lower = min(lower, upper)
    return lower, upper


def suggest_model_params(trial, model_algorithm, fold_train_size, optimization_setting=None):
    """Create valid, regularized Optuna parameters from replaceable settings."""
    optimization_setting = optimization_setting or Model_Optimization_Setting()
    fold_train_size = max(2, int(fold_train_size))

    if model_algorithm == "LightGBM":
        space = optimization_setting.lightgbm_search_space
        max_depth = trial.suggest_int("max_depth", *space["max_depth"])
        leaf_upper = max(1, min(space["num_leaves"][1], 2 ** max_depth))
        leaf_lower = min(max(1, space["num_leaves"][0]), leaf_upper)
        min_data_lower, min_data_upper = _fractional_integer_bounds(
            fold_train_size,
            space["min_data_in_leaf_fraction"],
            minimum=space["min_data_in_leaf_minimum"],
            maximum=max(1, fold_train_size // 2),
        )
        return {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "seed": 42,
            "verbosity": -1,
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", leaf_lower, leaf_upper),
            "learning_rate": trial.suggest_float(
                "learning_rate", *space["learning_rate"], log=True
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", min_data_lower, min_data_upper
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", *space["feature_fraction"]
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", *space["bagging_fraction"]
            ),
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": trial.suggest_float("lambda_l2", *space["lambda_l2"], log=True),
        }

    if model_algorithm == "NGBoost":
        space = optimization_setting.ngboost_search_space
        leaf_lower, leaf_upper = _fractional_integer_bounds(
            fold_train_size,
            space["min_samples_leaf_fraction"],
            minimum=space["min_samples_leaf_minimum"],
            maximum=max(1, fold_train_size // 2),
        )
        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf", leaf_lower, leaf_upper
        )
        return {
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min(fold_train_size, max(2, 2 * min_samples_leaf)),
            "learning_rate": trial.suggest_float(
                "learning_rate", *space["learning_rate"], log=True
            ),
            "minibatch_frac": trial.suggest_float(
                "minibatch_frac", *space["minibatch_frac"]
            ),
            "col_sample": trial.suggest_float("col_sample", *space["col_sample"]),
        }

    raise ValueError("Unsupported model algorithm: " + str(model_algorithm))


########### if文入れると遅くなるので、改良する必要あり→そこまで影響ないかも？
def objective_define(feature_setting, Fold_num, train_x, train_y, optimization_setting=None):
    optimization_setting = optimization_setting or Model_Optimization_Setting()
    fold_train_size = max(2, int(len(train_x) * (Fold_num - 1) / Fold_num))

    def objective(trial):
        ### parameter selection for model
        Model_algorithm = feature_setting.ML_algorithm_name
        params = suggest_model_params(
            trial,
            Model_algorithm,
            fold_train_size,
            optimization_setting,
        )

        #try:
        models, sc_s, pca_s, ICA_s, scores, scores_dist, pred_test, pred_test_dist, feature_name, evals_result_append\
        = predict_cv(
            params,
            feature_setting,
            Fold_num,
            train_x,
            train_y,
            optimization_setting=optimization_setting,
        )
        # holdoutは使用せず、trialごとのtrain/CV scoreのみを保存
        trial.set_user_attr("train_rmse_log10", float(scores[0]))
        cv_mean_score = scores[2]
        #except:
            #たまにICA convert ERRORがでるので、逃げの処理
        #    cv_mean_score=10*10

        return cv_mean_score

    return objective


def compile_optuna_score_history(study):
    """Compile train/CV log10 RMSE without exposing holdout scores."""
    return pd.DataFrame(
        {
            "trial": [trial.number for trial in study.trials],
            "train_rmse_log10": [
                trial.user_attrs.get("train_rmse_log10", np.nan)
                for trial in study.trials
            ],
            "cv_rmse_log10": [
                trial.value if trial.value is not None else np.nan
                for trial in study.trials
            ],
        }
    ).set_index("trial")


def print_score(train_mean_score, test_mean_score, cv_mean_score):
    #print_score
    print("トレーニング・平均スコア　：　" + str(train_mean_score))
    print("テスト・平均スコア　：　" + str(test_mean_score))
    print("CV・平均スコア　：　" + str(cv_mean_score))

def print_score_std(train_mean_score, test_mean_score, cv_mean_score):
    #print_score
    print("トレーニング・平均スコア　std (NGBoost)：　" + str(train_mean_score))
    print("テスト・平均スコア　std (NGBoost)：　" + str(test_mean_score))
    print("CV・平均スコア　std (NGBoost)：　" + str(cv_mean_score))


def calculate_generalization_gaps(scores):
    """Separate model-complexity and holdout-distribution gaps."""
    train_rmse, holdout_rmse, oof_rmse = [float(value) for value in scores]
    return {
        "Train_to_OOF_Gap_log10": oof_rmse - train_rmse,
        "OOF_to_Holdout_Gap_log10": holdout_rmse - oof_rmse,
    }


def select_model_phase(default_scores, optuna_scores):
    """Select by OOF RMSE only; never use holdout performance for model choice."""
    default_oof = float(default_scores[2])
    optuna_oof = float(optuna_scores[2])
    if np.isfinite(optuna_oof) and (
        not np.isfinite(default_oof) or optuna_oof < default_oof
    ):
        return "Optuna"
    return "Default"


def compile_early_stopping_summary(default_evals, optuna_evals):
    """Compile fold-level stopping metadata for both training phases."""
    rows = []
    for phase, evals in (("Default", default_evals), ("Optuna", optuna_evals)):
        for metadata in evals.get("_training_metadata", []):
            row = {"phase": phase}
            row.update(metadata)
            rows.append(row)
    columns = [
        "phase",
        "algorithm",
        "fold",
        "best_iteration",
        "stopping_iteration",
        "max_iteration",
        "stopping_metric",
    ]
    return pd.DataFrame(rows, columns=columns)


def compile_learning_curve_results(evals_result_append):
    """Compile train/eval histories even when folds stop at different iterations."""
    learning_curve_series = {}
    fold_numbers = sorted(key for key in evals_result_append if isinstance(key, int))
    for fold_number in fold_numbers:
        fold_result = evals_result_append[fold_number]
        for dataset_name in ("train", "eval"):
            if dataset_name not in fold_result:
                continue
            metric_values = next(iter(fold_result[dataset_name].values()))
            column_name = f"learning_curve_{fold_number}_{dataset_name}"
            learning_curve_series[column_name] = pd.Series(
                metric_values,
                dtype=float,
            )
    # 列を逐次追加すると最初のfoldのindexへ揃えられ、長いfoldの末尾が
    # 切り捨てられるため、全Seriesを集めてから最長indexで一括生成する。
    return pd.DataFrame(learning_curve_series)


def save_learning_curve_figures(
    evals_result_append,
    path_figure_all,
    phase_name,
    keep_legacy_names=False,
):
    """Plot each fold history and mark the restored best iteration."""
    metadata_by_fold = {
        item.get("fold"): item
        for item in evals_result_append.get("_training_metadata", [])
    }
    fold_numbers = sorted(key for key in evals_result_append if isinstance(key, int))
    for fold_number in fold_numbers:
        fold_result = evals_result_append[fold_number]
        fig, axis = plt.subplots()
        for dataset_name in ("train", "eval"):
            if dataset_name not in fold_result:
                continue
            metric_values = next(iter(fold_result[dataset_name].values()))
            iterations = np.arange(1, len(metric_values) + 1)
            axis.plot(iterations, metric_values, label=dataset_name)

        metadata = metadata_by_fold.get(fold_number, {})
        best_iteration = metadata.get("best_iteration")
        if best_iteration is not None:
            axis.axvline(
                best_iteration,
                color="black",
                linestyle="--",
                label=f"best iteration = {best_iteration}",
            )
        axis.set_xlabel("Boosting iteration")
        axis.set_ylabel(metadata.get("stopping_metric", "validation metric"))
        axis.set_title(f"{phase_name} fold {fold_number}")
        axis.legend()
        fig.savefig(
            path_figure_all + f"/learning_curve_{phase_name.lower()}_{fold_number}.pdf",
            bbox_inches="tight",
        )
        if keep_legacy_names:
            fig.savefig(
                path_figure_all + f"/learning_curve_{fold_number}.pdf",
                bbox_inches="tight",
            )
        plt.close(fig)

################################################################################################ Parameter tuning

################################################################################################ Model construction Main + Pickle

#################################################モデルの定義
def predict_model(elem, X_use, y, path_all_share, path_figure_all, feature_setting, Model_Training_Setting):
    #Scoreのデータを全て入れる
    Score_all = pd.Series([], dtype=pd.StringDtype())
    Score_all.name = elem

    print(f"X_use: {X_use.columns}")

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
    optimization_setting = getattr(
        Model_Training_Setting,
        "optimization_setting",
        Model_Optimization_Setting(),
    )
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
    = predict_cv(
        params,
        feature_setting,
        Fold_num,
        train_x,
        train_y,
        test_x,
        test_y,
        optimization_setting,
    )
    default_scores = list(scores)
    default_scores_dist = list(scores_dist)
    default_models = models
    default_sc_s = sc_s
    default_pca_s = pca_s
    default_ICA_s = ICA_s
    default_pred_test = pred_test
    default_pred_test_dist = pred_test_dist
    default_feature_name = feature_name
    default_evals_result_append = evals_result_append
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
    print_score_std(scores_dist[0], scores_dist[1], scores_dist[2])

    ##################################### Print score
    ##################################### Initial model construct: CrossVaridationによるアンサンブルモデル作成，Predict，Score

    ########################################################################### Optuna optimize
    ############################optunaによるベイズ最適化
    study = optuna.create_study() #load_if_exists=False 再度学習する場合，設定を変えている場合がほとんどだから。
    study.optimize(
        objective_define(
            feature_setting,
            Fold_num,
            train_x,
            train_y,
            optimization_setting,
        ),
        n_trials=n_trials,
    )

    #最適化の過程を可視化
    study_optuna_score = compile_optuna_score_history(study)
    study_optuna_score.plot(y=["train_rmse_log10", "cv_rmse_log10"])
    plt.savefig(path_all_share + "/Score_training_optimize.pdf", bbox_inches='tight')
    plt.savefig(path_all_share + "/Optuna_training_scores.pdf", bbox_inches='tight')
    plt.close()
    plt.show()

    # trialを旧ファイル名と修正済みファイル名の両方で保存
    optuna_trials = study.trials_dataframe()
    optuna_trials.to_excel(path_all_share + "/Optune_trials.xlsx")
    optuna_trials.to_excel(path_all_share + "/Optuna_trials.xlsx")
    study_optuna_score.to_excel(path_all_share + "/Optune_trials_ALL_score.xlsx")
    study_optuna_score.to_excel(path_all_share + "/Optuna_trials_all_scores.xlsx")
    ###best_paramをread
    best_params = study.best_params
    ############################optunaによるベイズ最適化

    #########################optunaによるベイズ最適化されたparamで計算
    models, sc_s, pca_s, ICA_s, scores, scores_dist, pred_test, pred_test_dist, feature_name, evals_result_append\
    = predict_cv(
        best_params,
        feature_setting,
        Fold_num,
        train_x,
        train_y,
        test_x,
        test_y,
        optimization_setting,
    )
    optuna_scores = list(scores)
    optuna_scores_dist = list(scores_dist)
    optuna_evals_result_append = evals_result_append
    #########################optunaによるベイズ最適化されたparamで計算
    ########################################################################### Optuna optimize

    ##################################### Print score
    #print_score
    print("########### Optuna_tuned")
    #トレーニングの平均スコアを格納
    Score_all["Optuna_Train_mean"] = optuna_scores[0]
    Score_all["Optuna_Train_Dist_mean"] = optuna_scores_dist[0]
    Score_all["Optuna_Test_mean"] = optuna_scores[1]
    Score_all["Optuna_Test_Dist_mean"] = optuna_scores_dist[1]
    Score_all["Optuna_CV_mean"] = optuna_scores[2]
    Score_all["Optuna_CV_Dist_mean"] = optuna_scores_dist[2]
    print_score(optuna_scores[0], optuna_scores[1], optuna_scores[2])
    print_score_std(optuna_scores_dist[0], optuna_scores_dist[1], optuna_scores_dist[2])
    ##################################### Print score

    # holdoutを参照せず、OOF RMSEが小さい方だけを最終保存・予測に採用する
    selected_model_phase = select_model_phase(default_scores, optuna_scores)
    if selected_model_phase == "Default":
        models = default_models
        sc_s = default_sc_s
        pca_s = default_pca_s
        ICA_s = default_ICA_s
        scores = default_scores
        scores_dist = default_scores_dist
        pred_test = default_pred_test
        pred_test_dist = default_pred_test_dist
        feature_name = default_feature_name
        evals_result_append = default_evals_result_append
    else:
        scores = optuna_scores
        scores_dist = optuna_scores_dist
        evals_result_append = optuna_evals_result_append

    Score_all["Selected_Model"] = selected_model_phase
    Score_all["Selected_Train_mean"] = scores[0]
    Score_all["Selected_Test_mean"] = scores[1]
    Score_all["Selected_CV_mean"] = scores[2]
    print("Selected model by OOF RMSE: " + selected_model_phase)
    #########score 保存
    Score_all.to_excel(path_all_share + "/Score_all.xlsx")

    # 意味とスケールを明示した新しい評価ファイル
    model_evaluation_scores = pd.Series(
        {
            "Default_Train_RMSE_log10": float(Score_all["Default_Train_mean"]),
            "Default_CV_RMSE_log10": float(Score_all["Default_CV_mean"]),
            "Default_OOF_RMSE_log10": float(Score_all["Default_CV_mean"]),
            "Default_Holdout_Test_RMSE_log10": float(Score_all["Default_Test_mean"]),
            "Optuna_Train_RMSE_log10": float(Score_all["Optuna_Train_mean"]),
            "Optuna_Tuning_CV_RMSE_log10": float(Score_all["Optuna_CV_mean"]),
            "Optuna_OOF_RMSE_log10": float(Score_all["Optuna_CV_mean"]),
            "Optuna_Holdout_Test_RMSE_log10": float(Score_all["Optuna_Test_mean"]),
            "Selected_Model_Is_Optuna": float(selected_model_phase == "Optuna"),
            "Selected_Train_RMSE_log10": float(scores[0]),
            "Selected_OOF_RMSE_log10": float(scores[2]),
            "Selected_Holdout_Test_RMSE_log10": float(scores[1]),
        },
        name=elem,
        dtype=float,
    )
    default_test_rmse = float(Score_all["Default_Test_mean"])
    optuna_test_rmse = float(Score_all["Optuna_Test_mean"])
    model_evaluation_scores["Optuna_Holdout_Test_RMSE_Improvement_log10"] = default_test_rmse - optuna_test_rmse
    if default_test_rmse != 0:
        model_evaluation_scores["Optuna_Holdout_Test_RMSE_Improvement_percent"] = 100.0 * (default_test_rmse - optuna_test_rmse) / default_test_rmse
    else:
        model_evaluation_scores["Optuna_Holdout_Test_RMSE_Improvement_percent"] = np.nan
    default_gaps = calculate_generalization_gaps(default_scores)
    optuna_gaps = calculate_generalization_gaps(optuna_scores)
    selected_gaps = calculate_generalization_gaps(scores)
    for gap_name, gap_value in default_gaps.items():
        model_evaluation_scores["Default_" + gap_name] = gap_value
    for gap_name, gap_value in optuna_gaps.items():
        model_evaluation_scores["Optuna_" + gap_name] = gap_value
    for gap_name, gap_value in selected_gaps.items():
        model_evaluation_scores["Selected_" + gap_name] = gap_value
    if Model_algorithm == "NGBoost":
        model_evaluation_scores["Default_Train_Predicted_SD_RMS_log10"] = float(Score_all["Default_Train_Dist_mean"])
        model_evaluation_scores["Default_CV_Predicted_SD_RMS_log10"] = float(Score_all["Default_CV_Dist_mean"])
        model_evaluation_scores["Optuna_Train_Predicted_SD_RMS_log10"] = float(Score_all["Optuna_Train_Dist_mean"])
        model_evaluation_scores["Optuna_Tuning_CV_Predicted_SD_RMS_log10"] = float(Score_all["Optuna_CV_Dist_mean"])
        model_evaluation_scores["Optuna_Holdout_Predicted_SD_RMS_log10"] = float(Score_all["Optuna_Test_Dist_mean"])
    model_evaluation_scores.to_excel(path_all_share + "/Model_evaluation_scores_log10.xlsx")
    early_stopping_summary = compile_early_stopping_summary(
        default_evals_result_append,
        optuna_evals_result_append,
    )
    early_stopping_summary.to_excel(
        path_all_share + "/Early_stopping_summary.xlsx",
        index=False,
    )
    print("Added evaluation output files:")
    for output_file_name in ADDED_EVALUATION_OUTPUT_FILES:
        print("- " + path_all_share + "/" + output_file_name)
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
    default_evals_result_compile = compile_learning_curve_results(
        default_evals_result_append
    )
    optuna_evals_result_compile = compile_learning_curve_results(
        optuna_evals_result_append
    )
    selected_evals_result_compile = compile_learning_curve_results(
        evals_result_append
    )
    save_learning_curve_figures(
        default_evals_result_append,
        path_figure_all,
        "Default",
    )
    save_learning_curve_figures(
        optuna_evals_result_append,
        path_figure_all,
        "Optuna",
    )
    save_learning_curve_figures(
        evals_result_append,
        path_figure_all,
        "Selected",
        keep_legacy_names=True,
    )

    for best_model in model_all.model:
        if Model_algorithm == 'LightGBM':
            try:
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
            except:
                pass
        elif Model_algorithm == 'NGBoost':
            pass
    # save importance
    importance_compile_gain.to_excel(path_all_share + "/feature_importance_gain.xlsx")
    importance_compile_split.to_excel(path_all_share + "/feature_importance_split.xlsx")
    # 既存名は最終選択モデルの履歴として維持し、Default/Optunaも別名で保存する
    selected_evals_result_compile.to_excel(path_all_share + "/learning_curve_result_compile.xlsx")
    default_evals_result_compile.to_excel(
        path_all_share + "/learning_curve_result_compile_default.xlsx"
    )
    optuna_evals_result_compile.to_excel(
        path_all_share + "/learning_curve_result_compile_optuna.xlsx"
    )
    ################################################################パラメータ・特徴量

    ########################################関数の定義
    #Scoreのデータを全て入れる #上で定義済み
    #test_誤差データ
    test_error_all = pd.DataFrame()
    #test_y raw
    if Model_algorithm == 'NGBoost':
        pred_test_dist.index = test_y.index # indexを指定してやる
    #######################テストデータの誤差計算
    #誤差の計算 -> % 表示で誤差を示す
    try:
        """
        pred_test, test_y は log10 空間の値とみなし、元スケールでの「予測/正解」の比を % 表示で計算
        score_error_freq_i = 100 × 10^(pred_i - true_i) = 100 × (10^pred_i / 10^true_i)
        （一致なら 100%、2倍なら 200%、半分なら 50%）
        Overflow Errorの問題を解決するため、npでの処理に変更 tag 2.1.1, 20260203
        """
        score_error_freq = 100.0 * np.power(10.0, np.asarray(pred_test, float) - np.asarray(test_y, float))
        test_error_all[elem] = pd.Series(score_error_freq, index = test_y.index)
        test_error_all.to_excel(path_all_share + "/test_error_all.xlsx")
    except OverflowError:
        try: 
            st.error(f"⚠️ {elem}で計算オーバーフローが発生しました。")
        except:
            print(f"⚠️ {elem}で計算オーバーフローが発生しました。")
    except Exception as e:
        try: 
            st.error(f"⚠️ {elem}でエラーが発生しました。")
        except:
            print(f"⚠️ {elem}でエラーが発生しました。")
    #誤差を格納していく

    ####################### testデータ
    # log scale -> liner scaleに変換
    y_raw_s = pd.Series(
        np.power(10.0, test_y.to_numpy(dtype=float)),
        index=test_y.index,
        name=test_y.name
    )
    test_data_all = pd.concat([y_raw_s, test_x], axis=1)
    test_data_all["RAW"] = y_raw_s

    # log scale -> liner scaleに変換
    pred_raw = np.power(10.0, np.asarray(pred_test, dtype=float))
    assert len(pred_raw) == len(test_y), f"len mismatch: pred={len(pred_raw)} != y={len(test_y)}"

    # “オーバーフロー相当” (inf) を検知して NaN にする
    mask = ~np.isfinite(pred_raw)
    if mask.any():
        print(f"⚠️ {elem}で計算オーバーフローが発生しました。(predict: {mask.sum()}件) (L1028)")
        pred_raw[mask] = np.nan
    # test data allにpredict dataを保存
    test_data_all["predict"] = pd.Series(pred_raw, index=test_y.index)

    # test data allに推定誤差を保存
    if Model_algorithm == 'NGBoost':
        test_data_all["predict_Dist"] = pred_test_dist
    # test data allを保存
    test_data_all.to_excel(path_all_share + "/test_data_all.xlsx")

    #######################################################################
    # 簡易的な散布図を作成する
    #######################################################################
    try:
        test_data_all.plot.scatter(x='RAW', y='predict', figsize=(3, 3), alpha=0.05)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(path_all_share + "/test_data_scatter.pdf", bbox_inches='tight')
        plt.close()
        plt.show()
    except:
        print("test_data_allに問題があります")

    if Model_algorithm == 'NGBoost': ### error bar付きの散布図
        try:
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
        except:
            print("test_data_allに問題があります")

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
        # Data_info.to_excel(path_all_share + "/Data_Info.xlsx")
        #######################################################データinfoの保存


        #######################################################使うデータの整理
        #目的yと入力データXを設定
        # data___ = Protolith_data[use_element].dropna() # 
        # 1. 処理前の行数を保持
        initial_count = len(Protolith_data)
        # 2. 置換と削除を実行 正の無限大(inf), 負の無限大(-inf) をまとめて NaN に置換して削除
        data___ = Protolith_data[use_element].replace([0, np.inf, -np.inf], np.nan).dropna()
        # 3. 処理後の行数を取得
        final_count = len(data___)
        # 4. 差分を表示
        dropped_count = initial_count - final_count
        Data_info["Nan/Inf dropped_count"] = dropped_count
        
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

        #######################################################データinfoの保存
        Data_info.to_excel(path_all_share + "/Data_Info.xlsx")
        #######################################################データinfoの保存

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
