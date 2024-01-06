import pandas as pd
import numpy as np

# sklearn modules
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)   
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# importing remaining classifiers
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import scipy.stats as stat

# importing created modules
import helper_functions

# Logistic Regression
def logistic_regression(inputs_train,loan_data_targets_train):

    reg = LogisticRegression()
    reg.fit(inputs_train, loan_data_targets_train)

    # creating a summary table
    feature_name = inputs_train.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    summary_table = summary_table.sort_index()

    return reg,summary_table

# Logistic Regression with p-values
class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self,X,y):
        self.model.fit(X,y)

        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        F_ij = F_ij.astype(np.float64) ## Inverse Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values

def logistic_regression_with_p_values(inputs_train,loan_data_targets_train):

    reg = LogisticRegression_with_p_values()
    reg.fit(inputs_train, loan_data_targets_train)

    feature_name = inputs_train.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    summary_table = summary_table.sort_index()

    p_values = reg.p_values
    p_values = np.append(np.nan, np.array(p_values))
    summary_table['p_values'] = p_values

    return reg,summary_table

# Decision Tree
def decision_tree(X_train, y_train):

    # Create a decision tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Hyperparameter Tuning
    dt_best_model, dt_best_params = hyper_tuning(X_train, y_train, dt_classifier, param_grid)

    return dt_best_model, dt_best_params

# Random Forest
def random_forest(X_train, y_train):

    # Create a random forest classifier
    rf_model = RandomForestClassifier(random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'max_depth': [2,3,5,10,20],
        'min_samples_leaf': [5,10,20,50,100,200],
        'n_estimators': [10,25,30,50,100,200]
    }

    # Hyperparameter Tuning
    rf_best_model, rf_best_params = hyper_tuning(X_train, y_train, rf_model, param_grid)

    return rf_best_model, rf_best_params

# Gaussian Naive Bayes
def gauss_NB(X_train, y_train):

    # Create a naive bayes classifier
    nb_classifier = GaussianNB(random_state=42)

    # Train the model
    nb_classifier.fit(X_train, y_train)

    return nb_classifier

# SVC
def svc(X_train, y_train):

    # Create an SVC classifier
    svc_classifier = SVC(random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # Hyperparameter Tuning
    svc_best_model, svc_best_params = hyper_tuning(X_train, y_train, svc_classifier, param_grid)

    return svc_best_model, svc_best_params

# Adaptive Boosting
def adaboost(X_train, y_train):

    # Create a base learner (e.g., Decision Tree)
    base_classifier, base_classifier_params = decision_tree(X_train, y_train)

    # Create an AdaBoost classifier with the base learner
    adaboost_classifier = AdaBoostClassifier(base_classifier,random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }

    # Hyperparameter Tuning
    adaboost_best_model, adaboost_best_params = hyper_tuning(X_train, y_train, adaboost_classifier, param_grid)

    return adaboost_best_model, adaboost_best_params

# Gradient Boosting
def gradient_boosting(X_train, y_train):

    # Create a Gradient Boosting classifier
    gb_classifier = GradientBoostingClassifier(random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Hyperparameter Tuning
    gradboost_best_model, gradboost_best_params = hyper_tuning(X_train, y_train, gb_classifier, param_grid)

    return gradboost_best_model, gradboost_best_params

# LightGBM
def lightGBM(X_train, y_train):

    # Create a LightGBM classifier
    lgb_classifier = LGBMClassifier(random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [1, 5, 10],
    }

    # Hyperparameter Tuning
    LightGBM_best_model, LightGBM_best_params = hyper_tuning(X_train, y_train, lgb_classifier, param_grid)

    return LightGBM_best_model, LightGBM_best_params

# Xgboost
def xgboost(X_train, y_train):

    # Create an XGBoost classifier
    xgb_classifier = XGBClassifier(random_state=42)


    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    # Hyperparameter Tuning
    xgboost_best_model, xgboost_best_params = hyper_tuning(X_train, y_train, xgb_classifier, param_grid)

    return xgboost_best_model, xgboost_best_params

# Catboost
def catboost(X_train, y_train):

    # Create a CatBoost classifier
    catboost_classifier = CatBoostClassifier(random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 4, 5],
    }

    # Hyperparameter Tuning
    catboost_best_model, catboost_best_params = hyper_tuning(X_train, y_train, catboost_classifier, param_grid)

    return catboost_best_model, catboost_best_params

# histogram-based gradient boosting algorithm
def hist_grad_boost(X_train, y_train):

    # Create a HistGradientBoostingClassifier
    hist_gb_classifier = HistGradientBoostingClassifier(random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'max_iter': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_leaf_nodes': [15, 31, 50]
    }

    # Hyperparameter Tuning
    histgbm_best_model, histgbm_best_params = hyper_tuning(X_train, y_train, hist_gb_classifier, param_grid)

    return histgbm_best_model, histgbm_best_params

# KNeighborsClassifier
def k_neighbors_classifier(X_train, y_train):

    # Create a KNeighborsClassifier
    knn_classifier = KNeighborsClassifier(random_state=42)

    # Define hyperparameters and their possible values for tuning
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Hyperparameter Tuning
    knn_best_model, knn_best_params = hyper_tuning(X_train, y_train, knn_classifier, param_grid)

    return knn_best_model, knn_best_params

# Hyperparameter Tuning
def hyper_tuning(X_train, y_train, model, param_grid):

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Use the best model to make predictions
    best_model = grid_search.best_estimator_

    return best_model, best_params

# Model Validation
def out_of_sample_validation(inputs_test,loan_data_targets_test,model):

    y_test_pred=model.predict(inputs_test)

    loan_data_targets_test_temp = loan_data_targets_test
    loan_data_targets_test_temp.reset_index(drop = True, inplace = True)

    df_actual_predicted_probs = pd.concat([loan_data_targets_test_temp, pd.DataFrame(y_test_pred)], axis = 1)
    df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']
    df_actual_predicted_probs.index = inputs_test.index

    return df_actual_predicted_probs

# Model Evaluation
def evaluate_model(df_actual_predicted_probs):

    df_actual_predicted_probs, confusion_matrix, accuracy, fpr, tpr, thresholds, AUROC = helper_functions.accuracy_n_auc(df_actual_predicted_probs)
    df_actual_predicted_probs, Gini, KS = helper_functions.gini_n_ks(df_actual_predicted_probs,AUROC)

    return df_actual_predicted_probs, confusion_matrix, accuracy, fpr, tpr, thresholds, AUROC, Gini, KS