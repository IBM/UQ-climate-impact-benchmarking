import numpy as np


def calc_rmse(df, true_variable, predicted_variable):
    return( np.sqrt(((df[true_variable] - df[predicted_variable]) **2 ).mean()) )


def calc_zscore(df, true_variable, predicted_variable, predicted_variance):
    return( ((df[predicted_variable]-df[true_variable])/(df[predicted_variance]-df[predicted_variable]).abs()).replace([np.inf, -np.inf], np.nan).mean(skipna=True) )


def hit_rate(df, true, upper, lower):
    return sum(sum([(df[true]<df[upper])&(df[true]>df[lower])]))/len(df)


def calc_var(df, variable, q):
    for name, group in df.groupby('datetime'):
        var = group[variable].quantile(q)
        cvar = group[group[variable] > var][variable].mean()
        df.loc[df['index'].isin(group['index']), 'var'+variable] = var 
        df.loc[df['index'].isin(group['index']), 'cvar'+variable] = cvar 
    return df