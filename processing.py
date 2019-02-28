import numpy as np
import pandas as pd
import time
import os
from model import train_model
from config import params
from IPython import embed

random_state = 13
np.random.seed(random_state)

print('\033[40;32m Read Data \033[0m')
train = pd.read_csv('/unsullied/sharefs/chenhang/plate/Santander/data/train.csv')
test = pd.read_csv('/unsullied/sharefs/chenhang/plate/Santander/data/test.csv')
print('\033[40;32m Read Data Finished \033[0m')

test_x = test.drop(['ID_code','var_185','var_27','var_30','var_17','var_38','var_41','var_126','var_103'],axis=1)
train_x = train.drop(['ID_code','target','var_185','var_27','var_30','var_17','var_38','var_41','var_126','var_103'],axis=1)
train_y = train['target']

X = train_x.values
y = train_y.values
x_test = test_x
feature_name = train_x.columns.tolist()
n_folds = 5
model_list = ['lgb']
predict = pd.DataFrame()
oof_res = pd.DataFrame()
print('\033[40;32m Preprocessing Finished \033[0m')
for each in model_list:
	params = params.params_dic[each]
	print('\033[40;32m Start Training {} Model \033[0m'.format(each))
	print('Params is : {}'.format(params))
	if 'lgb' in each or 'xgb' in each:
		oof, prediction, feature_importance = train_model(X=X, y=y, X_test=test_x, featurename=feature_name, model_type=each, params=params, n_folds=n_folds)
	else:
		oof, prediction = train_model(X=X, y=y, X_test=test_x, feature_name=feature_name, model_type=each, params=params, n_folds=n_folds)
	predict[each] = prediction
	oof_res[each] = oof
print('\033[40;32m Training Finished, Starting Generate Submission \033[0m')



submission = pd.DataFrame({"ID_code": test.ID_code.values})
submission["target"] = predict.T.mean()
save_path = '/unsullied/sharefs/chenhang/plate/Santander/data/toy_sub.csv'
submission.to_csv(save_path, index=False)
print('\033[40;32m Starting Upload Submission and Evaluating Result \033[0m')
command = 'kaggle competitions submit -c santander-customer-transaction-prediction -f {} -m "{}"'.format(save_path, 'CSDN baseline')
os.system(command)
print('\033[40;32m All Finished \033[0m')