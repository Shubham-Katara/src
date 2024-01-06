from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)   
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# this function will apply all the models using best parameters.
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    
    # this report will contain model name as key and its corresponding r2_score as value for every model.
    report={}

    for i in range(len(list(models))):
        model=list(models.values())[i]
        para=param[list(models.keys())[i]]
        '''
        GridSearchCV is a technique used for hyperparameter tuning in machine learning and is part of the scikit-learn 
        library in Python. It is designed to automate the process of systematically searching for the best combination 
        of hyperparameters for a machine learning model.
        '''
        gs=GridSearchCV(model,para,cv=3)
        gs.fit(X_train,y_train)
        # model.fit(X_train,y_train) # Train model

        model.set_params(**gs.best_params_)
        model.fit(X_train,y_train)

        # y_train_pred=model.predict(X_train)
        y_test_pred=model.predict(X_test)

        # train_model_score=r2_score(y_train,y_train_pred)
        test_model_score=r2_score(y_test,y_test_pred)

        report[list(models.keys())[i]]=test_model_score

    return report


def initiate_model_trainer(X_train,y_train,X_test,y_test):
    
    # X_train,y_train,X_test,y_test=(
    #     train_array[:,:-1], # all rows; all columns except last column.
    #     train_array[:,-1],  # all rows; last column only.
    #     test_array[:,:-1],
    #     test_array[:,-1]
    # )
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Naive Bayes":BernoulliNB(),
        "SVM Model": SVC(),  # You can choose different kernels like 'linear', 'rbf', etc.
        "XGBRegressor": XGBRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Adaboost Regressor": AdaBoostRegressor()
    } # can apply lightGBM and Neural Networks

    params={
        "Logistic Regression":{},
        "Random Forest":{
            # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
            # 'max_features':['sqrt','log2',None],
            'n_estimators':[8,16,32,64,128,256]
        },
        "Decision Tree":{
            'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
            # 'spliter':['best','random'],
            # 'max_features':['sqrt','log2']
        },
        "Naive Bayes":{
            'alpha':1.0
        },
        "SVC":{
            'C': 1.0,  # Regularization parameter (adjust as needed)
            'kernel': 'rbf',  # Radial basis function kernel
            'gamma': 'scale',  # Kernel coefficient for 'rbf' (other options: 'auto', float)
            'degree': 3,  # Degree of the polynomial kernel function ('poly' kernel)
        },
        "XGBRegressor":{
            'learning_rate':[.1,.01,.05,.001],
            'n_estimators':[8,16,32,64,128,256]
        },
        "Gradient Boosting":{
            'loss':['squared_error','huber','absolute_error','quantile'],
            'learning_rate':[.1,.01,.05,.001],
            'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            # 'criterion':['squared_error','friedman_mse'],
            # 'max_features':['sqrt','log2','auto'],
            'n_estimators':[8,16,32,64,128,256]
        },
        "CatBoosting Regressor":{
            'depth':[6,8,10],
            'learning_rate':[.01,.05,.1],
            'iterations':[30,50,100]
        },
        "K-Neighbors Regressor":{
            'n_neighbors':[5,7,9,11],
            # 'weights':['uniform','distance'],
            # 'algorithm':['ball_tree','kd_tree','brute']
        },
        "Adaboost Regressor":{
            'learning_rate':[.1,.01,.5,.001],
            # 'loss':['linear','square','exponential'],
            'n_estimators':[8,16,32,64,128,256]
        }
    }

    model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                        models=models,param=params)
    
    ## Get best model score from the dict
    best_model_score=max(sorted(model_report.values()))

    ## Get best model name from the dictionary
    best_model_name=list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
    ]
    best_model=models[best_model_name]

    if best_model_score<0.6:
        raise print("No best model found")
    else:
        print('Best Model:',best_model)

    # dumping my best model
    # save_object(
    #     file_path=self.model_trainer_config.trained_model_file_path,
    #     obj=best_model
    # )

    predicted=best_model.predict(X_test)

    r2_square=r2_score(y_test,predicted)

    return model_report,r2_square