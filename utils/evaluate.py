import numpy as np


def calc_rmse(df, true_variable, predicted_variable):
    return( np.sqrt(((df[true_variable] - df[predicted_variable]) **2 ).mean()) )


def calc_var(df, variable, q):
    for name, group in df.groupby('datetime'):
        var = group[variable].quantile(q)
        cvar = group[group[variable] > var][variable].mean()
        df.loc[df['index'].isin(group['index']), 'var'+variable] = var 
        df.loc[df['index'].isin(group['index']), 'cvar'+variable] = cvar 
    return df