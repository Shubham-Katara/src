import pandas as pd
import numpy as np

import helper_functions

import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# importing data
# loan_data_backup = pd.read_csv(r"C:\Shubham\credit_scorecard\loan_data_2015\loan_data_2015_new.csv")
# loan_data = loan_data_backup.copy()

# exploring data
# print(loan_data.info())

# general preprocessing
# loan_data = preprocessing.general_preprocessing(loan_data)

# PD Model
## creating target column
# loan_data = preprocessing.target_var(loan_data)
# loan_data.to_csv(r'C:\Shubham\credit_scorecard\loan_data_2015\loan_data_with_target_var.csv',index=False)

## splitting data
# loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = preprocessing.splitting_data(loan_data)
# loan_data_inputs_train.to_csv(r"C:\Shubham\credit_scorecard\output_data\loan_data_inputs_train.csv",index=False)
# loan_data_inputs_test.to_csv(r"C:\Shubham\credit_scorecard\output_data\loan_data_inputs_test.csv",index=False)
# loan_data_targets_train.to_csv(r"C:\Shubham\credit_scorecard\output_data\loan_data_targets_train.csv",index=False)
# loan_data_targets_test.to_csv(r"C:\Shubham\credit_scorecard\output_data\loan_data_targets_test.csv",index=False)

# loan_data_inputs_train = pd.read_csv(r"C:\Shubham\credit_scorecard\output_data\loan_data_inputs_train.csv")
# loan_data_inputs_test = pd.read_csv(r"C:\Shubham\credit_scorecard\output_data\loan_data_inputs_test.csv")
# loan_data_targets_train = pd.read_csv(r"C:\Shubham\credit_scorecard\output_data\loan_data_targets_train.csv")
# loan_data_targets_test = pd.read_csv(r"C:\Shubham\credit_scorecard\output_data\loan_data_targets_test.csv")

## preprocessing discrete variables using binning technique
# df_inputs_prepr = loan_data_inputs_train.copy()
# df_targets_prepr = loan_data_targets_train.copy()

# df_inputs_prepr = preprocessing.preprocess_discrete_var(df_inputs_prepr,df_targets_prepr)

