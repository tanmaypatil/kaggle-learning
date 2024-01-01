# Import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

def randomized_search_class(X_train,y_train,X_test,y_test,params,scoring, reg,runs=16):

    # Instantiate RandomizedSearchCV as grid_classifier model
    rand_classify = RandomizedSearchCV(reg, params,n_iter=runs, scoring=scoring, 
                                  cv=10, n_jobs=-1, random_state=2)
    
    # Fit grid_classifier on X_train and y_train
    rand_classify.fit(X_train, y_train)

    # Extract best estimator
    best_model = rand_classify.best_estimator_

    # Extract best params
    best_params = rand_classify.best_params_

    # Print best params
    print("Best params:", best_params)
    
    # Compute best score
    best_score = rand_classify.best_score_

    # Print best score
    print("Training score: {:.3f}".format(best_score))

    # Predict test set labels
    y_pred = best_model.predict(X_test)
    
    # calculate accuracy , precision , recall on the best model
    report = classification_report(y_test, y_pred)
    print(f"report post randomised grid search on best model - Validation data :\n {report}")
    return (best_model,y_pred)