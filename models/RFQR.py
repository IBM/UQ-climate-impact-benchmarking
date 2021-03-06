import six
import sys
sys.modules['sklearn.externals.six'] = six
from skgarden import RandomForestQuantileRegressor
from joblib import dump, load
import pandas as pd
import numpy as np

def train_rfqr(X_train, y_train, X_test, random_state, min_samples_split, n_estimators, n_jobs, target_scaler, split, sequence_length, n_features, verbose=True, save_model=True, load_model=None, quantiles=[0.1587, 0.50, 0.8413], labels=['R0 lower', 'R0 mean', 'R0 upper']):

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
            dump(rfqr,
                 'trained_models/RFQR'
                 +'_'+split
                 +'_seq'+str(sequence_length)
                 +'_feat'+str(n_features)
                 +'_rs'+str(random_state)
                 +'_minss'+str(min_samples_split)
                 +'_estims'+str(n_estimators)
                 +'_jobs'+str(n_jobs)
                 +'.joblib')
    else:
        print('Loading model ...')
        rfqr = load('trained_models/'+load_model)
        
    y_pred = []
    for quantile, label in zip(quantiles, labels):
        print('Predicting quantile:', label)
        if target_scaler == None:
            y_pred.append(rfqr.predict(X_test, quantile=quantile*100))
        else:
            y_pred.append(target_scaler.inverse_transform(rfqr.predict(X_test, quantile=quantile*100)))
        
    y_pred = pd.DataFrame(np.array(y_pred).T, columns=labels)
    
    return(y_pred)