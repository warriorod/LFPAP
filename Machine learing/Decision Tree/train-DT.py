import sys, getopt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
import pickle

#use all internal data to train predict model for external test
def DT_train_model(data_set):

    X = np.array(data_set.values[:,:-1])
    y = np.array(data_set.values[:,-1])

    total_num = data_set.shape[:][0]
    aki_num = np.sum(y)
    print(f'{total_num},{aki_num},{total_num - aki_num}')
    model = DecisionTreeClassifier(criterion = 'log_loss', max_depth=4, class_weight = 'balanced',min_samples_split= 50, random_state=12)
    model.fit(X,y)
    y_scores = model.predict_proba(X)
    y_scores = y_scores[:,1]

    f1_s = -1
    for thre in range(100): 
        predictions = [1 if y > thre *0.01 else 0 for y in y_scores]
        tse = accuracy_score(y, predictions)
        if f1_s < tse:
            f1_s = tse
            threshold = thre * 0.01
    print(threshold)
    with open('./decision_tree_model.pkl', 'wb') as file:
        pickle.dump(model, file)

#internal kflod test
def DT_kflod(data_set):
    X = np.array(data_set.values[:,:-1])
    y = np.array(data_set.values[:,-1])
    total_num = data_set.shape[:][0]
    aki_num = np.sum(y)
    print(f'{total_num},{aki_num},{total_num - aki_num}')


    kf = StratifiedKFold(n_splits=5,shuffle=True, random_state=96)
    f1_scores = []
    pr_scores = []
    recall_scores = []
    acc_scores = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        valid_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, valid_idx = next(valid_sss.split(np.zeros(len(train_index)), y_train))
        X_valid = X_train[valid_idx]
        y_valid = y_train[valid_idx]
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        X_test, y_test = X_test, y_test


        model = DecisionTreeClassifier(criterion = 'log_loss', max_depth=10, min_samples_split= 50, random_state=12)
        model.fit(X_train,y_train)
        y_valid_scores = model.predict_proba(X_valid)
        y_valid_scores = y_valid_scores[:,1]
        y_scores = model.predict(X_test)

        f1_s = -1
        threshold = 0
        for thre in range(100):
            predictions =  [1 if y > thre *0.01 else 0 for y in y_valid_scores]
            tse = f1_score(y_valid, predictions)
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
    print(f'{average_f1:.4f},{average_pr:.4f},{average_recall:.4f},,{average_acc:.4f}')
    print('\n')




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
    DT_kflod(data_set)
    DT_train_model(data_set)


   