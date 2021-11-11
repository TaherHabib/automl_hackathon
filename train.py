
import pandas as pd

import sklearn
import os


from sklearn.metrics import accuracy_score

import config.settings
import autosklearn

def read_csv(path:os.path):
    """

    :return:
    """
    df = pd.read_csv((path))
    # train, test =sklearn.model_selection.train_test_split(df,test_size=0.2)

    return df


def scoring_function(estimator, X, Y):
    predictions = estimator.predict(X)
    return sklearn.metrics.accuracy_score(Y, predictions)


if __name__ == '__main__':
    root = config.settings.get_project_path()
    dataset_path = os.path.join(root,'Data')

    df= read_csv(os.path.join(dataset_path,'train.csv'))
    df_test = read_csv(os.path.join(dataset_path,'test.csv'))
    # y_train = train_df['class']
    # y_test = test_df['class']
    # X_train = train_df.drop('class',axis=1)
    # X_test = test_df.drop('class', axis =1)

    X = df.loc[:, df.columns != 'class']

    Y = df.loc[:, df.columns == 'class']
    test_size = 0.25
    shuffle = True

    train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(X, Y, test_size=test_size, shuffle=shuffle)

    estimator = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=300,  # in seconds
        seed=42,
        resampling_strategy='cv',
        resampling_strategy_arguments={'shuffle': True, 'folds': 5},
        metric=autosklearn.metrics.accuracy,
        n_jobs=-1
    )
    estimator.fit(train_X, train_Y, dataset_name='hackathon_data')
    print(f"Train Auto-Sklearn Classifier performance is {scoring_function(estimator, train_X, train_Y)}")
    print(f"Test Auto-Sklearn Classifier performance is {scoring_function(estimator, test_X, test_Y)}")

    print(estimator.show_models())

    test = df.loc[:, df_test.columns != 'id']
    id = df.loc[:, df_test.columns == 'id']

    test_result = estimator.predict(test)











