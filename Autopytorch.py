from autoPyTorch import AutoNetClassification


# Other imports for later usage
import pandas as pd
import numpy as np
import os as os
import openml
import json
from train import read_csv
import config
import sklearn

autonet = AutoNetClassification(config_preset="full_cs", result_logger_dir="logs/",)
root = config.settings.get_project_path()
dataset_path = os.path.join(root,'Data')
df = read_csv(os.path.join(dataset_path,'train.csv'))
df_test = read_csv(os.path.join(dataset_path,'test.csv'))

X = df.loc[:, df.columns != 'class']

Y = df.loc[:, df.columns == 'class']
test_size = 0.25
shuffle = True

train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(X, Y, test_size=test_size, shuffle=shuffle)


results_fit = autonet.fit(train_X,
            train_Y,
            validation_split=0.3,
            max_runtime=600,
            min_budget=60,
            max_budget=200,
            refit=True
            )
with open("logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)

y_pred = autonet.predict(test_X)
print("Accuracy score", sklearn.metrics.accuracy_score(test_Y, y_pred))

df_test = read_csv(os.path.join(dataset_path,'test.csv'))
test = df_test.loc[:, df_test.columns != 'id']
id_ = df_test.loc[:,df_test.columns == 'id']
test_result = autonet.predict(test)
id_['Predicted'] = list(test_result)
id_.to_csv('submission.csv',index=False)