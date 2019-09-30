# Imports
import numpy as np
import pandas as pd
import os
import sys
import gc
import pickle
import cv2
import gzip
from numba import jit
from scipy.stats import rankdata
import impyute as impy
import torch
import random
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from itertools import *
from sklearn.preprocessing import LabelEncoder


def seed_everything(seed=23):
    '''
    For reproducible results.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def save_in_file(arr, file_name):
    '''
    Takes an object and saves it in file_name
    '''
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    '''
    Loads the object from file_name
    Complements the save_in_file function
    '''
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    '''
    Faster save in file function
    Coupled with load_from_file_file_fast
    '''
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    '''
    Faster load from file function
    Coupled with save_in_file_fast
    '''
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def read_single_image(path):
    img = np.array(Image.open(path))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 2:
        img = img[:, :, :1]

    if img.shape[2] == 1:
        img = np.concatenate((img, img, img), axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    img2 = img2[:, :, ::-1]
    return img2


# inspired from https://www.kaggle.com/c/instant-gratification/discussion/97047#562826
def auc(actual, predicted, approx=True):
    '''
    approx=True : approx AUC score by not handling ties correctly. Reutrns close to AUC score almost always. 9x faster than Sklearn.
    approx=False : True AUC score, 3x faster than Sklearn.
    '''
    if approx:
        r = np.argsort(predicted)
    else:
        r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(r[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', auc(y_true, y_pred, True), True


def impute_mean(arr):
    '''
    Substitute missing values with the mean of that column.
    '''
    return impy.imputation.cs.mean(arr)


def impute_median(arr):
    '''
    Substitute missing values with the median of that column(middle).
    '''
    return impy.imputation.cs.median(arr)


def impute_mode(arr):
    '''
    Substitute missing values with the mode of that column(most frequent).
    '''
    return impy.imputation.cs.mode(arr)


def impute_fast_knn(arr, k=3):
    '''
    :params: arr - np.ndarray - 2D data to impute
    :params: k - int, optional - Number of neighbours used in the KNN query
    '''
    return impy.imputation.cs.fast_knn(arr, k=k)


def impute_random(arr):
    '''
    Fill missing values in with a randomly selected value from the same column.
    :params: arr - np.ndarray - 2D data to impute
    '''
    return impy.imputation.cs.random(data)


def plot_missing_values_matrix(df):
    '''
    Matrix - Number of rows is number of samples and number of columns is number of features in df.
    The sparkline at right summarizes the general shape of the data completeness and points out the maximum and minimum rows.
    Missing values are white and non missing values are black in plot.
    :params: df - pd.DataFrame
    '''
    msno.matrix(df)
    plt.show()


def plot_missing_values_bar(df):
    '''
    :params: df - pd.DataFrame
    '''
    msno.bar(df)
    plt.show()


def reduce_mem_usage(df, verbose=True):
    '''
    :params: df - pandas.Dataframe
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# taken from https://www.kaggle.com/artgor/artgor-utils
def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                           verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(
                **params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)
                                ], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist,
                              early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(
                X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(
                X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1,)

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](
                y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
        np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(
                cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(
                by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance

    return result_dict


# taken from https://www.kaggle.com/artgor/artgor-utils
def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns == None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                            'catboost_metric_name': 'AUC',
                            'sklearn_scoring_function': metrics.roc_auc_score},
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros((len(X), len(set(y.values))))

    # averaged predictions on train data
    prediction = np.zeros((len(X_test), oof.shape[1]))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(
                **params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)
                                ], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(
                X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(
                X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(
                X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid
        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](
            y_valid, y_pred_valid[:, 1]))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
        np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(
                cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(
                by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance

    return result_dict


def count_distinct_values(df, colname):
    '''
    :params: df - pd.DataFrame
    :params: colname - str
    '''
    df1 = pd.DataFrame(df[colname].value_counts(dropna=False))
    df1.index.names = [colname]
    df1.columns = ['Count']
    df1.sort_index(inplace=True)
    return df1


def label_encode_text_column(df, field, df_test=None, fillna="-999"):
    '''
    :params: df - pd.DataFrame
    :params: field - column to encode
    :params: df_test - pd.DataFrame, optional
    '''
    encoded_df = df[field].fillna(fillna)
    if(df_test is not None):
        encoded_df = encoded_df.append(df_test[field].fillna(fillna))
    label_encoder = LabelEncoder()
    encoder = label_encoder.fit(encoded_df.values)

    if(df_test is not None):
        encoded_t = encoder.transform(df_test[field].fillna(fillna).values)
        df_test[field+"_encoded"] = encoded_t
    encoded_df = encoder.transform(df[field].fillna(fillna).values)
    df[field+"_encoded"] = encoded_df
    return encoder
