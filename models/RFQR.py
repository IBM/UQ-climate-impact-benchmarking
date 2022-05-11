import six
import sys
sys.modules['sklearn.externals.six'] = six
from skgarden import RandomForestQuantileRegressor
from joblib import dump, load
import pandas as pd
import numpy as np

def train_rfqr(X_train, y_train, X_test, random_state, min_samples_split, n_estimators, n_jobs, verbose=True, save_model=True, load_model=None, quantiles=[0.1587, 0.50, 0.8413], labels=['R0 lower', 'R0 mean', 'R0 upper']):

    X_train = X_train.reshape(X_train.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)

    if load_model == None:
        print('Training model ...')
        rfqr = RandomForestQuantileRegressor(random_state=random_state, 
                                             min_samples_split=min_samples_split, 
                                             n_estimators=n_estimators, 
                                             n_jobs=n_jobs, 
                                             verbose=verbose)

        rfqr.set_params(max_features=X_train.shape[1] // 3)
        rfqr.fit(X_train, y_train)

        if save_model:
            dump(rfqr, 'trained_models/RFQR_train2017-2020_predict2021_feat4.joblib') 
    else:
        print('Loading model ...')
        rfqr = load('trained_models/'+load_model)
        
    y_pred = []
    for quantile, label in zip(quantiles, labels):
        print('Predicting quantile:', label)
        y_pred.append(rfqr.predict(X_test, quantile=quantile*100))
        print('End of predicting quantile:', label)
        
    y_pred = pd.DataFrame(np.array(y_pred_rfqr).T, columns=labels)
    
    return(y_pred)