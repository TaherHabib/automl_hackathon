
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import autosklearn.classification
import autosklearn.metrics
from tpot import TPOTClassifier

import config.settings


def read_csv(path:os.path):
    """

    :return:
    """
    df = pd.read_csv((path))
    train, test =sklearn.model_selection.train_test_split(df,test_size=0.2)

    return train,test

def scoring_function(estimator):
    predictions = estimator.predict_proba(X_test)[:, 1]
    return sklearn.metrics.roc_auc_score(y_test, predictions, multi_class="ovo")

def train_scoring_function(estimator):
    predictions = estimator.predict_proba(X_train)[:, 1]
    return sklearn.metrics.roc_auc_score(y_train, predictions,multi_class="ovo")

if __name__ == '__main__':
    root = config.settings.get_project_path()
    dataset_path = os.path.join(root,'Data')

    train_df , test_df = read_csv(os.path.join(dataset_path,'train.csv'))
    y_train = train_df['class']
    y_test = test_df['class']
    X_train = train_df.drop('class',axis=1)
    X_test = test_df.drop('class', axis =1)


    # names = [
    #     "Nearest Neighbors",
    #     # "Linear SVM",
    #     # "RBF SVM",
    #     "Gaussian Process",
    #     "Decision Tree",
    #     "Random Forest",
    #     "Neural Net",
    #     "AdaBoost",
    #     "Naive Bayes",
    #     "QDA",
    # ]
    #
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('cat', OneHotEncoder(handle_unknown='ignore'), X_train.dtypes == "object"),
    #         ('cont', 'passthrough', X_train.dtypes != "object"),
    #     ],
    #     remainder='passthrough',
    # )
    #
    # classifiers = [
    #     KNeighborsClassifier(3),
    #     # SVC(kernel="linear", C=0.025),
    #     # SVC(gamma=2, C=1),
    #     GaussianProcessClassifier(1.0 * RBF(1.0)),
    #     DecisionTreeClassifier(max_depth=5),
    #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     MLPClassifier(alpha=1, max_iter=1000),
    #     AdaBoostClassifier(),
    #     GaussianNB(),
    #     QuadraticDiscriminantAnalysis(),
    # ]
    #
    # for i, estimator_gradboost in enumerate(classifiers):
    #     pipeline_byop = Pipeline([
    #         ('preprocessor', preprocessor),
    #         ('gradboost', estimator_gradboost),
    #     ])
    #     # -------------------------
    #
    #     pipeline_byop.fit(X_train, y_train)
    #
    #     # Score the pipeline
    #     print(f'Classifier {names[i]} achievies following scores: ')
    #     print("model score: %.3f" % pipeline_byop.score(X_test, y_test))
    #     # performance_byop = train_scoring_function(pipeline_byop)
    #     # print(f"Train performance of my pipeline is {performance_byop}")
    #     #
    #     # performance_byop = scoring_function(pipeline_byop)
    #     # print(f"Test performance of my pipeline is {performance_byop}")
    #
    #     print('========================================================')

    # estimator_askl = autosklearn.classification.AutoSklearnClassifier(
    #     # ------------------------- edit code here
    #     time_left_for_this_task=30,  # in seconds
    #     seed=42,
    #     resampling_strategy='holdout',
    #     metric=autosklearn.metrics.roc_auc,
    #     n_jobs=1,
    #     # -------------------------
    # )
    # # Auto-sklearn ingests the pandas dataframe and detects column types
    # estimator_askl.fit(X_test, y_test, dataset_name='c')

    tpot_pipeline = TPOTClassifier(generations=64,
                                   population_size=256,
                                   offspring_size=512,
                                   scoring='accuracy',
                                   max_time_mins=1000,
                                   n_jobs=8,
                                   early_stop=52,
                                   log_file='./tpot_log_1.txt',
                                   verbosity=2)
    tpot_pipeline.fit(X_train, y_train)

    print("\nTPOT  Accuracy: ", accuracy_score(y_test, tpot_pipeline.predict(X_test)))

    tpot_pipeline.export('tpot_1.py')







