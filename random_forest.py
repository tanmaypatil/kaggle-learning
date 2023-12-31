# Import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def randomized_search_class(params, runs=16, reg=RandomForestClassifier(random_state=2, n_jobs=-1)):

    # Instantiate RandomizedSearchCV as grid_reg
    rand_reg = RandomizedSearchCV(reg, params, n_iter=runs, scoring='neg_mean_squared_error', 
                                  cv=10, n_jobs=-1, random_state=2)
    
    # Fit grid_reg on X_train and y_train
    rand_reg.fit(X_train, y_train)

    # Extract best estimator
    best_model = rand_reg.best_estimator_

    # Extract best params
    best_params = rand_reg.best_params_

    # Print best params
    print("Best params:", best_params)
    
    # Compute best score
    best_score = np.sqrt(-rand_reg.best_score_)

    # Print best score
    print("Training score: {:.3f}".format(best_score))

    # Predict test set labels
    y_pred = best_model.predict(X_test)
    
    # Import mean_squared_error from sklearn.metrics as MSE 
    from sklearn.metrics import mean_squared_error as MSE

    # Compute rmse_test
    rmse_test = MSE(y_test, y_pred)**0.5

    # Print rmse_test
    print('Test set score: {:.3f}'.format(rmse_test))