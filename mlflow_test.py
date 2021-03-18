import os
import sys
import yaml
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import gc
from time import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve,recall_score,precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import lightgbm as lgb

##mlflow
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn

# #logging
# logging.basicConfig(filename='last_outputs/logger.log',level=logging.WARN)
# logging.info('test')
# logger = logging.getLogger(__name__)

with open("param.yaml") as stream:
    param_lgb = yaml.safe_load(stream)
    
# with open("folder_param.yaml") as stream:
#     folda_param = yaml.safe_load(stream)

#データ取得
input_data=pd.read_csv('./creditcard.csv')
input_data=input_data[['V4','V14','Class']]

def eval_metrics(actual, pred):
    auc = roc_auc_score(actual, pred)
    pred=pred > 0.5#threshold
    recall = recall_score(actual, pred)
    precision = precision_score(actual, pred)
    return auc, recall, precision

np.random.seed(40)
df_train,df_test=train_test_split(input_data, test_size=0.3)
#トレーニングデータ、テストデータの準備
X_train=df_train.drop('Class',axis=1)
y_train=pd.DataFrame(df_train['Class'])
X_test=df_test.drop('Class',axis=1)
y_test=pd.DataFrame(df_test['Class'])
y_train=y_train.rename(columns={'Class':'target'})
y_test=y_test.rename(columns={'Class':'target'})

def predict(X_train, y_train, X_test, outputfilename,param_lgb):
    
    #初期時間取得
    start=time()

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1,)

    preds_train = np.zeros(X_train.shape[0])
    preds_test = np.zeros(X_test.shape[0])
    
    # mlflowのexperimentを生成
    mlflow.set_experiment(param_lgb['experiment_name'])
    tracking = mlflow.tracking.MlflowClient()
    experiment = tracking.get_experiment_by_name(param_lgb['experiment_name'])
    
    with mlflow.start_run():#experiment_id=experiment.experiment_id, nested=True):
    
        folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=1001)
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[valid_idx], y_train[valid_idx]

            #LightGBM parameters
            clf = lgb.LGBMClassifier(
                objective=param_lgb['objective'],
                n_estimators=param_lgb['n_estimators'],
                num_leaves=param_lgb['num_leaves'],
                is_unbalance=param_lgb['is_unbalance'],
                min_child_weight=param_lgb['min_child_weight'],
                )

            clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)],
                eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

            preds_train[valid_idx] = clf.predict_proba(X_val, num_iteration=clf.best_iteration_)[:, 1]
            preds_test += clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

            #metric保存
            (auc, recall, precision)=eval_metrics(y_val, preds_train[valid_idx])

            mlflow.log_metric("auc", auc)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)

            #parameter保存
            mlflow.log_param("objective", param_lgb['objective'])
            mlflow.log_param("n_estimators", param_lgb['n_estimators'])
            mlflow.log_param("num_leaves", param_lgb['num_leaves'])
            mlflow.log_param("is_unbalance", param_lgb['is_unbalance'])
            mlflow.log_param("min_child_weight", param_lgb['min_child_weight'])

            #モデル保存
            mlflow.sklearn.log_model(clf, str(n_fold)+"_model")

            #重要度保存
            importance = pd.DataFrame({'importance':clf.feature_importances_,'column_name':X_test.columns})
            importance.to_csv('outputs/'+str(n_fold)+'_importance.csv')
            mlflow.log_artifacts("outputs")
            
            #tag保存
            mlflow.set_tag(key=param_lgb['tag'], value=auc)

        #result保存
        preds_test_r = preds_test.reshape(1,-1)
        prob = pd.DataFrame(preds_test_r, columns=list(X_test.index)).T
        prob = prob.rename(columns={0:'TARGET'})
        prob.index.name = 'ID'
        prob['answer']=y_test
        prob.to_csv('last_outputs/%s.csv'%outputfilename)
        mlflow.log_artifacts("last_outputs")

        print('time:{}sec'.format(time()-start))
        
    return

predict(X_train, y_train, X_test, 'lgbm_test',param_lgb)


#if __name__='__main__':
    



