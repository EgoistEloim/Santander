import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import os
import sys
import time
import datetime
from tqdm import tqdm
import lightgbm as lgb
import operator
import xgboost as xgb
#import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#from imblearn.under_sampling import RandomUnderSampler
import warnings
#from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp
from sklearn import manifold
from IPython import embed
warnings.filterwarnings("ignore")







def train_model(X, y, X_test, featurename=None, params=None, n_folds=3, model_type='lgb', plot_feature_importance=False, model=None):
    folds = KFold(n_splits=n_folds, shuffle=True)
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    #feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('\033[40;32m Fold {} started at {} \033[0m'.format(fold_n, time.ctime()))
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        if model_type == 'lgb':
            train_data = lgb.Dataset(data=X_train, label=y_train)
            valid_data = lgb.Dataset(data=X_valid, label=y_valid)
            model = lgb.train(params, train_data, num_boost_round=20000,
                              valid_sets=[train_data, valid_data], verbose_eval=1000, early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=featurename)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=featurename)
            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=1000, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=featurename),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=featurename), ntree_limit=model.best_ntree_limit)

        if model_type == 'rcv':
            model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_absolute_error', cv=3)
            model.fit(X_train, y_train)
            print(model.alpha_)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = mean_absolute_error(y_valid, y_pred_valid)
            print('\033[40;32m Fold {}. MAE: {}. \033[0m'.format(fold_n, round(score, 4)))

            y_pred = model.predict(X_test).reshape(-1, )

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = mean_absolute_error(y_valid, y_pred_valid)
            print('\033[40;32m Fold {}. MAE: {}. \033[0m'.format(fold_n, round(score, 4)))


            y_pred = model.predict(X_test).reshape(-1, )

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000, eval_metric='auc', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred_valid, pos_label=1)
        scores.append(metrics.auc(fpr, tpr))

        prediction += y_pred

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = featurename
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            try:
                feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            except:
                feature_importance = fold_importance
        if model_type == 'xgb':
            fold_importance = model.get_fscore()
            fold_importance = sorted(fold_importance.items(), key=operator.itemgetter(1))
            feature_importance = pd.DataFrame(fold_importance, columns=['feature', 'importance'])

    prediction /= n_fold
    print('\033[40;32m CV mean score: {0:.4f}, std: {1:.4f}. \033[0m'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        '''
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 26))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
        '''
        return oof, prediction, feature_importance


    elif model_type == 'xgb':
        feature_importance['importance'] /= n_fold
        '''
        if plot_feature_importance:
            plt.figure(figsize=(16, 26))
            feature_importance.plot(kind='barh', x='feature', y='importance', legend=False, figsize=(6, 10))
            plt.title('XGB Features (avg over folds)')
            plt.xlabel('relative importance')
            plt.show()
            return oof, prediction, feature_importance
        '''
        return oof, prediction, feature_importance
    else:
        return oof, prediction