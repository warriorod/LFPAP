
import sys, getopt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
import pickle

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

    X = np.array(data_set.values[:,:-1])
    y = np.array(data_set.values[:,-1])

    with open('./rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    threshold = 0.51


    y_scores = model.predict_proba(X)
    y_scores = y_scores[:,1]
    


    


    predictions =  [1 if y > threshold else 0 for y in y_scores]
# 计算 ROC 曲线值
    pre_s = precision_score(y,predictions)
    recall_s = recall_score(y,predictions)
    f1_s = f1_score(y, predictions)
    print("threshold: %f" %(threshold))
    print("f1 score: %f" %(f1_s))
    print("precision: %f" %(pre_s))
    print("recall: %f" %(recall_s))

    acc_s = accuracy_score(y,predictions)
    print("accuracy: %f" %(acc_s))
