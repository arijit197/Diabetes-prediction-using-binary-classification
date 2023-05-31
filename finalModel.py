import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score

Diabetes_Dataset = pd.read_csv('updatedDataset.csv')

Diabetes_Dataset['BMI'].replace(0,np.nan,inplace=True)
Diabetes_Dataset['Glucose'].replace(0,np.nan,inplace=True)
Diabetes_Dataset['BloodPressure'].replace(0,np.nan,inplace=True)
Diabetes_Dataset['SkinThickness'].replace(0,np.nan,inplace=True)
Diabetes_Dataset['Insulin'].replace(0,np.nan,inplace=True)

imputer = KNNImputer(n_neighbors=7)
Imputed_Dataset = pd.DataFrame(imputer.fit_transform(Diabetes_Dataset))
Imputed_Dataset.columns = Diabetes_Dataset.columns

def Predict_RandomForest(X_train,Y_train,X_test,Y_test):
    model = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print("Accuracy for Random Forest Classifier is : ",accuracy_score(predictions,Y_test))
    acc = accuracy_score(predictions,Y_test)
    return round(acc,4),model

X = Imputed_Dataset[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]]
Y = Imputed_Dataset["Outcome"]

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, train_size=0.7, test_size=0.3,random_state=0)

over_sampler = RandomOverSampler(sampling_strategy='minority')
X_Over,Y_Over = over_sampler.fit_resample(X,Y)
Sampled_Dataset = X_Over.merge(Y_Over,left_index=True,right_index=True)

X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X_Over, Y_Over, train_size=0.7, test_size=0.3,random_state=0)

acc,model = Predict_RandomForest(X_train3,Y_train3,X_test1,Y_test1)
