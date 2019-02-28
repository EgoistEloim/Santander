class params:
    params_dic = {
        'lgb': {'num_leaves': 10,
                'min_data_in_leaf': 42,
                'objective': 'binary',
                'max_depth': 18,
                'learning_rate': 0.01,
                'boosting': 'gbdt',
                'bagging_freq': 6,
                'bagging_fraction': 0.8,
                'feature_fraction': 0.9,
                'bagging_seed': 11,
                'reg_alpha': 2,
                'reg_lambda': 5,
                'random_state': 42,
                'metric': 'auc',
                'verbosity': -1,
                #'subsample': 0.9,
                'min_gain_to_split': 0.01077313523861969,
                'min_child_weight': 19.428902804238373,
                'num_threads': 8},
        'xgb': {},
        'rcv': {},
        'cat': {}

    }

