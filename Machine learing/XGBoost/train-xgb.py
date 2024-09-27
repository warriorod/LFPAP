import xgboost as xgb
import sys, getopt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score,recall_score,precision_score,accuracy_score

#use all internal data to train predict model for external test
def xgb_train_model(data_set, num_r):#fulltrain

    X = np.array(data_set.values[:,:-1])
    y = np.array(data_set.values[:,-1])

    total_num = data_set.shape[:][0]
    aki_num = np.sum(y)
    print(f'{total_num},{aki_num},{total_num - aki_num}')

    dtrain = xgb.DMatrix(data=X, label=y, enable_categorical=True,feature_names=data_set.columns.tolist()[:-1])

    # 设置参数
    params = {
        'objective': 'binary:logistic',  # 二分类的逻辑回归问题
        'eval_metric': 'logloss',        # 评估指标为 logloss
        'tree_method': 'hist',       # 使用 GPU 加速的直方图方法
        'device' : "cuda",
        'eta' : 0.05,
        'gamma': 0.1,
        'max_depth': 4,
        'min_child_weight': 1,
    }


    num_boost_round = num_r
    bst = xgb.train(params, dtrain, num_boost_round,verbose_eval = 0)
    y_scores = bst.predict(dtrain)

    f1_s = -1
    for thre in range(100): 
        predictions = [1 if y > thre *0.01 else 0 for y in y_scores]
        tse = accuracy_score(y, predictions)
        if f1_s < tse:
            f1_s = tse
            threshold = thre * 0.01
    print(threshold)
    bst.save_model('xgb_model0911.json')


#internal kflod test
def xgb_kflod(data_set,num_r):
    X = np.array(data_set.values[:,:-1])
    y = np.array(data_set.values[:,-1])
    total_num = data_set.shape[:][0]
    aki_num = np.sum(y)
    print(f'{total_num},{aki_num},{total_num - aki_num}')

    num_boost_round = num_r
    params = {
        'objective': 'binary:logistic',  # 二分类的逻辑回归问题
        'eval_metric': 'logloss',        # 评估指标为 logloss
        'tree_method': 'hist',       # 使用 GPU 加速的直方图方法
        'device' : "cuda",
        'eta' : 0.05,
        'gamma': 0.1,
        'max_depth': 4,
        'min_child_weight': 1
    }

    kf = StratifiedKFold(n_splits=5,shuffle=True, random_state=92)

    f1_scores = []
    pr_scores = []
    recall_scores = []
    acc_scores = []

    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        valid_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, valid_idx = next(valid_sss.split(np.zeros(len(train_index)), y_train))
        dtrain = xgb.DMatrix(X_train[train_idx], label=y_train[train_idx],enable_categorical=True)
        dvalid = xgb.DMatrix(X_train[valid_idx], label=y_train[valid_idx],enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test,enable_categorical=True)

        bst = xgb.train(params, dtrain, num_boost_round, evals=[(dvalid, 'valid')], early_stopping_rounds=3,verbose_eval = 0)

        
        y_scores = bst.predict(dtest)
        y_valid_scores = bst.predict(dvalid)
        y_valid = y_train[valid_idx]
        f1_s = -1
        threshold = 0
        for thre in range(100):
            predictions =  [1 if y > thre *0.01 else 0 for y in y_valid_scores]
            tse = accuracy_score(y_valid, predictions)
            if f1_s < tse:
                f1_s = tse
                threshold = thre * 0.01
        predictions =  [1 if y > threshold else 0 for y in y_scores]
        f1 = f1_score(y_test, predictions)

        pre_s = precision_score(y_test,predictions)
        recall_s = recall_score(y_test,predictions)
        acc_s = accuracy_score(y_test,predictions)
        f1_scores.append(f1)
        pr_scores.append(pre_s)
        recall_scores.append(recall_s)
        acc_scores.append(acc_s)
    average_f1 = np.mean(f1_scores)
    average_pr = np.mean(pr_scores)
    average_recall = np.mean(recall_scores)
    average_acc = np.mean(acc_scores)
    print(f'Average F1 Score,Average precision,Average recall,Average accuracy')
    print(f'{average_f1:.4f},{average_pr:.4f},{average_recall:.4f},{average_acc:.4f}')
    print(f'{average_pr*average_recall*2/(average_pr+average_recall):.4f}')
    print('\n')
    ttp = average_recall * aki_num
    tfp = (ttp / average_pr) - ttp
    return ttp, tfp, total_num - aki_num - tfp , aki_num -ttp

    
if __name__ == "__main__":
    arg = sys.argv[1:]
    try:
       opts, args = getopt.getopt(arg, "hc:", ["csv_file ="])
    except getopt.GetoptError:
       print('python train.py -c <csv_file>')
       sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -c <csv_file>')
            sys.exit()
        elif opt in ("-c", "--csv_file"):
            csv_file = arg

    data_set = pd.read_csv(csv_file)
    xgb_kflod(data_set,20)
    xgb_train_model(data_set,20)



   
