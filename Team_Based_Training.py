import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from collections import Counter
import joblib
from matplotlib import pyplot

def main():
    #Load train and test.
    train_set_data = pd.read_csv('StatsScrapper/team_training_set_balanced.csv',index_col=0)
    test_set_data = pd.read_csv('StatsScrapper/team_test_set.csv',index_col=0)
    train_set_target =  train_set_data['match_result']
    test_set_target =  test_set_data['match_result']
    train_set_data.to_csv()
    test_set_data.to_csv()
    train_set_data = train_set_data[train_set_data.columns.difference(['match_result','season','date','a_team_title','h_team_title'])]
    test_set_data = test_set_data[test_set_data.columns.difference(['match_result','season','date','a_team_title','h_team_title'])]
    #Prepare Logistic Regression Pipeline
    lr_pipeline = Pipeline([('Standarizer', StandardScaler()),
                            ('Selector', VarianceThreshold()),
                            ('Classifier', LogisticRegression())
                            ])
    lr_pipeline_params = {'Selector__threshold': [0,0.001,0.01],
                          'Classifier__C': [0.001,0.01,0.1,1,10,100,1000],
                          'Classifier__penalty': ['l2'],
                          'Classifier__max_iter': [30,40,50,60,70,80,100],
                          'Classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                          }
    #Logistic Regression Tuning and optimized results.
    grid = GridSearchCV(lr_pipeline, lr_pipeline_params, cv=5, scoring='accuracy').fit(train_set_data, train_set_target)
    print('Training set score LR optmized: ' + str(grid.score(train_set_data, train_set_target)))
    print('Test set score LR optimized: ' + str(grid.score(test_set_data, test_set_target)))
    print(grid.best_params_)
    print(grid.best_estimator_.predict(test_set_data))
    joblib.dump(grid.best_estimator_, 'model.pkl')
    model_columns = list(train_set_data.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    #Prepare Random Forest Pipeline
    rf_pipeline = Pipeline([('Selector', VarianceThreshold()),
                            ('Classifier', RandomForestClassifier())
                            ])
    rf_pipeline_params = {'Selector__threshold': [0],
                          'Classifier__n_estimators': [1000,1100,1200,1400],
                          'Classifier__max_features': ['sqrt'],
                          'Classifier__max_depth': [5,10,20],
                          'Classifier__min_samples_split': [2,4,5,6,7,10],
                          'Classifier__min_samples_leaf': [2,3],
                          'Classifier__bootstrap': [False]
                          }
    #Logistic Regression Tuning and optimized results.
    grid = GridSearchCV(rf_pipeline, rf_pipeline_params, cv=2, scoring='accuracy',n_jobs=-1).fit(train_set_data, train_set_target)
    print('Training set score RF optmized: ' + str(grid.score(train_set_data, train_set_target)))
    print('Test set score RF optimized: ' + str(grid.score(test_set_data, test_set_target)))
    print(grid.best_params_)
    #Prepare SVC Pipeline
    svc_pipeline = Pipeline([('Standarizer', StandardScaler()),
                            ('Selector', VarianceThreshold()),
                            ('Classifier', SVC())
                            ])
    svc_pipeline_params = {'Selector__threshold': [0],
                          'Classifier__kernel':['linear', 'rbf', 'poly'],
                          'Classifier__gamma':[0.1, 1, 10, 100],
                          'Classifier__C':[0.1, 1, 10, 100, 1000],
                          'Classifier__degree':[3]
                          }
    #Logistic Regression Tuning and optimized results.
    grid = GridSearchCV(svc_pipeline, svc_pipeline_params, cv=2, scoring='accuracy',n_jobs=-1).fit(train_set_data, train_set_target)
    print('Training set score SVC optmized: ' + str(grid.score(train_set_data, train_set_target)))
    print('Test set score SVC optimized: ' + str(grid.score(test_set_data, test_set_target)))
    print(grid.best_params_)

if __name__ == '__main__':
    main()
