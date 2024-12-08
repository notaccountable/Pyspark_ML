import os 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import csv 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import auc 
import pandas as pd
from sklearn.model_selection import train_test_split


userCSV = input("where is your database located? \n")

features = pd.read_csv(userCSV)
features.head(5)

print('features printout: \n', features )
print('The shape of our features is: ', features.shape)

# Descriptive statistics for each column
features.describe()

# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)

# Use numpy to convert to arrays
# Labels are the values we want to predict
labels = np.array(features['quality'])

# Remove the labels from the features
# axis 1 refers to the columns

features= features.drop('quality', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
print(feature_list)


# Convert to numpy array
# enabled gives an error for no attribute of iloc within arrays
#features = np.array(features)
# Using Skicit-learn to split data into training and testing sets



# pulled from ex 1
features_train_all, features_test_all, labels_train_all, labels_test_all = train_test_split(features, labels, test_size=0.1)
for dataset in [features_train_all, features_test_all, labels_train_all, labels_test_all]:
    #print (dataset)
    dataset = pd.DataFrame()
    dataset.reset_index(drop=True, inplace=True)
    

print('Training Features Shape:', features_train_all.shape)
print('Training Labels Shape:', labels_train_all.shape)
print('Testing Features Shape:', features_test_all.shape)
print('Testing Labels Shape:', labels_test_all.shape)


C = 1.0
#count = 0
#overall_average_metrics = {}
#initialized_count=0

def get_metrics(model, X_train, X_test, y_train, y_test, LSTM_flag):
    #from CrossVal import performance_metrics_calc
    #performance_metrics_calc() # subject to change location
    def performance_metrics_calc(confMatrix):
        TP = confMatrix[0][0]
        FP = confMatrix[1][0]
        TN = confMatrix[1][1]
        FN = confMatrix[0][1]

        TPR = TP / (TP+FN)
        TNR = TN / (TN+FP)
        FPR = FP / (TN+FP)
        FNR = FN / (TP+FN)
        precision = TP / (TP+FP)
        f1_measure = (2*TP) / (2*TP+FP+FN)
        acc = (TP+TN) / (TP+FP+FN+TN)
        error_rate = (FP+FN) / (TP+FP+FN+TN)
        BACC = (TPR+TNR) / 2
        TSS = TPR - FPR
        HSS = (2*(TP*TN-FP*FN)) / ((TP+FN) * (FN+TN) + (TP+FP) * (FP+TN))
        metrics = [TP, FP, TN, FN, TPR, TNR, FPR, FNR, precision, f1_measure, acc, error_rate, BACC, TSS, HSS]

        print("these are your TP: \n", TP)
        print("these are your FP: \n", FP)
        print("these are your TN: \n", TN)
        print("these are your FN: \n", FN)
        return(metrics)
    
    metrics = []
    #overall_average_metrics = {}
    #print("this is the loop you are on: ", count)
    #count += 1

    if LSTM_flag == 0:
        
        #features_train = features_train.reshape(features_train.shape[1:])
        features_train.shape
        #print(features_train)

        model.fit(features_train, labels_train)
        
        #print("#Testing purpose || 2 \n") #prev issue happening for the above that the number of samples are inconsistent| solution around was to just not use labels_ train and instead opt of other version
        predicted = model.predict(features_test)
        for x in range(len(predicted)):
            print("predicted value: ", predicted[x])
            print("actual values: ", labels_test[x])
            if predicted[x] == labels_test[x]:
                predicted[x]=1
            else:
                predicted[x]= 0

            if labels_test[x] == labels_test[x]:
                labels_test [x]=1
            else:
                labels_test [x]= 0
        #current issue -> confusion matrix is not returning accurate TP and FP readings

        matrix = confusion_matrix(labels_test, predicted, labels = [1,0])
        model_brier_score = brier_score_loss(labels_test, model.predict_proba(features_test)[:,1])
        print("This is your Brier Score: ",model_brier_score)
        metrics.extend(performance_metrics_calc(matrix))
        metrics.extend([model_brier_score, model.score(features_test, labels_test)])
        count = 0

    return metrics
        
# Initializing metrics returns
rf_metrics_list = []
rf_classifier_name = "Random Forest Classifier"
knn_metrics_list = []
knn_classifier_name = "K-nearest Neighbor Classifier"


#parameter tuning
knn_parameters = {"n_neighbors": [1,2,3,4,5,6,78,9,10,11,12,13,14,15]}
knn_model = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_model, knn_parameters, cv = 10, n_jobs=1)
knn_cv.fit(features_train_all, labels_train_all)
best_knn_params = knn_cv.best_params_["n_neighbors"]
print("Best parameters for KNN based on Gridsearch CV: \n", best_knn_params)
print("\n")

features_train = features_train_all
labels_train = labels_train_all


for dataset in [features_train, labels_train]:
    dataset = pd.DataFrame()

cv_stratified = StratifiedKFold(n_splits = 10, shuffle = True, random_state=42)
cv_stratified.get_n_splits(features_train_all)
print (cv_stratified)
C = 1.0

features_train_all = features_train_all.values
#Prompt user for which model they want to use
model_requested = input("Please enter which model you would like to use: \n 1. Random Forrest\n 2. KNN\n 3. SVM\n")


for i, (train_index, test_index) in enumerate(cv_stratified.split(features_train,labels_train),start=1):
    
    features_train, features_test = features_train_all[train_index], features_train_all[test_index]
    labels_train, labels_test = labels_train_all[train_index], labels_train_all[test_index]

    if model_requested == '1':
        rf_model = RandomForestClassifier(min_samples_split=10, n_estimators=1000)
        #s3_model = rf_model
        rf_metrics = get_metrics(rf_model, features_train, features_test, labels_train_all, labels_test, 0)
        rf_metrics_list.append(rf_metrics)
        print("These are your random forest metrics: ", rf_metrics_list)

    if model_requested == '2':
        knn_model = KNeighborsClassifier(n_neighbors=best_knn_params)
        #s3_model = knn_model
        knn_metrics = get_metrics(knn_model, features_train, features_test, labels_train_all, labels_test, 0)
        knn_metrics_list.append(knn_metrics)
        print("these are your KNN Metrics: \n", knn_metrics_list)


# used to display average of metrics after calculating
def averaged_metrics(model_metrics_list, model_name):
    dict_count = 0
    index_count = 0
    overall_average_metrics = {}

    for a in range(17):
        overall_average_metrics[a] = 0

    
    for metrics_list in model_metrics_list:
        dict_count = 0
        for b in range(17):
            overall_average_metrics[dict_count] += metrics_list[dict_count]
            dict_count+=1

    for score in overall_average_metrics:
        overall_average_metrics[score] = overall_average_metrics[score]/10
        

    oam = overall_average_metrics    
    print("These are your averaged metrics of all Cross Validation folds in ", model_name, ":")
    print("Avg TP: ",oam[0],"\nAvg FP: ",oam[1],"\nAvg TN: ",oam[2],"\nAvg FN: ",oam[3],"\nAvg TPR: ",oam[4],"\nAvg TNR: ",oam[5],"\nAvg FPR: ",oam[6],"\nAvg FNR: ",oam[7],"\nAvg precision: ",oam[8],"\nAvg f1 score: ",oam[9],"\nAvg acc: ",oam[10],"\nAvg error rate: ",oam[11],"\nAvg BACC: ",oam[12],"\nAvg TSS: ",oam[13],"\nAvg HSS: ",oam[14],"\nAvg Briar score: ",oam[15],"\nAvg AUC score: ",oam[16])

if model_requested == '1':
    averaged_metrics(rf_metrics_list, rf_classifier_name)
if model_requested == '2':
    averaged_metrics(knn_metrics_list, knn_classifier_name)


#averaged_metrics(lstm_metrics_list, lstm_classifier_name)

##