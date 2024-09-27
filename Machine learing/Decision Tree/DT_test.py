import xgboost as xgb
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

    # get test patients' data
    data_set = pd.read_csv(csv_file)
    total_aki = np.sum(np.array(data_set.values[:,-1]))
    total_patient = data_set.shape[:][0]


    X = np.array(data_set.values[:,:-1])
    y = np.array(data_set.values[:,-1])

    # load predict model 
    with open('./decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
    threshold = 0.48


    y_scores = model.predict(X)
    

    


  
    
    #cal performance
    predictions =  [1 if y > threshold else 0 for y in y_scores]
    pre_s = precision_score(y,predictions)
    recall_s = recall_score(y,predictions)
    f1_s = f1_score(y, predictions)
    acc_s = accuracy_score(y,predictions)

    print("threshold: %f" %(threshold))
    print("f1 score: %f" %(f1_s))
    print("precision: %f" %(pre_s))
    print("recall: %f" %(recall_s))
    print("accuracy: %f" %(acc_s))

