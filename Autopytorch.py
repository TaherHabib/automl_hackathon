from autoPyTorch.api.tabular_classification import TabularClassificationTask


# Other imports for later usage
import pandas as pd
import numpy as np
import os as os
import openml
import json
from train import read_csv
import config
import sklearn

root = config.settings.get_project_path()
dataset_path = os.path.join(root,'Data')
df = read_csv(os.path.join(dataset_path,'train.csv'))
df_test = read_csv(os.path.join(dataset_path,'test.csv'))

X = df.loc[:, df.columns != 'class']

Y = df.loc[:, df.columns == 'class']
test_size = 0.25
shuffle = True

train_X, test_X, train_Y, test_Y = sklearn.model_selection.train_test_split(X, Y, test_size=test_size, shuffle=shuffle)



api = TabularClassificationTask(
    # To maintain logs of the run, you can uncomment the
    # Following lines
    temporary_directory='./tmp/autoPyTorch_example_tmp_01',
    output_directory='./tmp/autoPyTorch_example_out_01',
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
    seed=42,
)

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=train_X,
    y_train=train_Y,
    X_test=test_X.copy(),
    y_test=test_Y.copy(),
    optimize_metric='accuracy',
    total_walltime_limit=300,
    func_eval_time_limit_secs=50
)

############################################################################
# Print the final ensemble performance
# ====================================
print(api.run_history, api.trajectory)
y_pred = api.predict(test_X)
score = api.score(y_pred, test_Y)
print(score)
# Print the final ensemble built by AutoPyTorch
print(api.show_models())



df_test = read_csv(os.path.join(dataset_path,'test.csv'))
test = df_test.loc[:, df_test.columns != 'id']
id_ = df_test.loc[:,df_test.columns == 'id']
test_result = api.predict(test)
id_['Predicted'] = list(test_result)
id_.to_csv('submission.csv',index=False)