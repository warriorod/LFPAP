import sys, getopt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from sklearn.impute import SimpleImputer
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
    
    #load dataset
    data_set = pd.read_csv(csv_file)

    #impute miss feature
    data_set['Lactate Dehydrogenase'] = 195
    data_set['Direct Bilirubin'] = 6
    data_set['Uric Acid'] = 280
    data_set['Cystatin C'] = 1.2
    data_set['Lymphocyte Count'] = 1.5
    data_set['Neutrophil Count'] = 4.79
    data_set['Eosinophil Count'] = 0.125
    data_set['Red Blood Cell Count'] = 4.2

    X = np.array(data_set.values[:,:-1])
    y = np.array(data_set.values[:,-1])
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    total_num = data_set.shape[:][0]

    with open('./lgr_model.pkl', 'rb') as file:
        model = pickle.load(file)
    threshold = 0.49

    y_scores = model.predict_proba(X)[:,1]
    
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
    