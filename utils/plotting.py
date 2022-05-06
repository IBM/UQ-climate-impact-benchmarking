import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DayLocator, HourLocator, DateFormatter, drange



def plot_true_vs_predict(df, x, y, label, title, figname):
    fig = px.scatter(df,
                     color_discrete_sequence = (['crimson']),
                     x = x,
                     labels=label,
                     y= y,
                     title="R0 vs Time",
                     width=400, height=400,
                     trendline = 'ols',
                     trendline_color_override='blue')
    results = px.get_trendline_results(fig).px_fit_results.iloc[0]
    r2 = results.rsquared
    mb = results.params
    fig.add_annotation(x=0.2, y= 1.05, text =f'<b>y={round(mb[1],2)}x+{round(mb[0],2)}</b>', showarrow=False)
    fig.add_annotation(x=0., y= 0.97, text =f'<b>r<sup>2</sup>={r2:.2f}</b>', showarrow=False)
    fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>",title_text= "<b> True R0</b>", ticks="outside")
    fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>",title_text= "<b> Predicted R0 </b>", ticks="outside", range = [-0.1,1.1])
    fig.for_each_trace(lambda t: t.update(name = '<b>' + t.name +'</b>'))
    fig.update_layout(
        title="",
        xaxis_title="<b> True R<sub>0</sub></b>",
        yaxis_title = "<b> Predicted R<sub>0</sub> </b>",
        font=dict(
            family="Verdana",
            size=12,
            color="black"
        )
    )
    fig.write_image(figname, engine="kaleido")

    return fig

def plot_true_in_interval(df, sortby, figname):

    df_visu = df.sort_values(by=sortby).copy()
    
    fig = plt.figure(figsize=(5,4))
    plt.plot(range(len(df)), df_visu['R0'], "C0o", markersize=0.5, label="True values")
    #plt.plot(range(len(y_test)), df_visu_std['R0 mean'], "bo", markersize=0.5)

    plt.fill_between(range(len(df)),
        df_visu['R0 sigminus'], df_visu['R0 sigplus'], alpha=0.5, color="C3",
        label="Quantiles 15.87 - 84.13")

    plt.legend()
    plt.xlabel('Test samples sorted by $True\ ' + sortby +'$')
    plt.ylabel('R0')
    fig.savefig(figname, bbox_inches='tight')
    
    return fig


def plot_single_members(df, iforecasts, figname):
    
    


    fig, axes = plt.subplots(len(iforecasts), 1, figsize=(7,7), sharex=True, sharey=True)
    for ax, iforecast in zip(axes, iforecasts):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        df_visu = df[df['forecast']==iforecast].copy()
        ax.fill_between(df_visu['date'],
                         df_visu['R0 sigminus'], df_visu['R0 sigplus'], alpha=0.5, color="C3",
                         label="Quantiles 15.87 - 84.13")

        ax.plot(df_visu['date'], df_visu['R0'], label="Ensemble member "+str(iforecast))
        ax.set_ylabel('R0')
        ax.legend()
    
    fig.autofmt_xdate()
    ax.set_xlabel('Date')
    fig.savefig(figname, bbox_inches='tight')
    
    return fig
    
    
def aggregate_ensembles(df, figname):
    df_assembled = df[['date', 'R0 mean', 'R0 sigminus', 'R0 sigplus']].copy()
    df_assembled = df_assembled.groupby('date').agg({'R0 mean':'mean', 'R0 sigminus':'mean', 'R0 sigplus':'mean'})

    total = 0
    for idate in df['date'].unique():
        idf = df[df['date']==idate].copy()
        idf['upper'] = df_assembled[df_assembled.index==idate]['R0 sigplus'].values[0]
        idf['lower'] = df_assembled[df_assembled.index==idate]['R0 sigminus'].values[0]
        total += sum(sum([(idf['R0']<idf['upper'])&(idf['R0']>idf['lower'])]))
    print(total/len(df))
    print(total)
    print(len(df))
    
    fig, ax = plt.subplots(figsize=(5,4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.scatter(df['date'], df['R0'], c='C0', marker='o', s=0.5, label="True values")
    ax.fill_between(df_assembled.index,
        df_assembled['R0 sigplus'], df_assembled['R0 sigminus'], alpha=0.5, color="C3",
        label="Quantiles 15.87 - 84.13")
    fig.autofmt_xdate()

    plt.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('R0')
    fig.savefig(figname, bbox_inches='tight')
    
    
def aggregate_ensembles_samples(df, figname):
    
    df_agg_melt = df.drop(
        ['index', 'forecast', 'season', 'R0'], axis=1
    ).melt(
        id_vars='date',value_name="Value"
    ).copy()
    print(df_agg_melt)
    df_assembled = df_agg_melt.groupby('date').quantile()
    df_assembled['lower'] = df_agg_melt.groupby('date').quantile(0.1587)
    df_assembled['upper'] = df_agg_melt.groupby('date').quantile(0.8413)

    total = 0
    for idate in df['date'].unique():
        idf = df[df['date']==idate].copy()
        idf['upper'] = df_assembled[df_assembled.index==idate]['upper'].values[0]
        idf['lower'] = df_assembled[df_assembled.index==idate]['lower'].values[0]
        total += sum(sum([(idf['R0']<idf['upper'])&(idf['R0']>idf['lower'])]))
    print(total/len(df))

    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.scatter(df['date'], df['R0'], c='C0', marker='o', s=0.5, label="True values")
    ax.fill_between(df_assembled.index,
        df_assembled['lower'], df_assembled['upper'], alpha=0.5, color="C3",
        label="Quantiles 15.87 - 84.13")
    fig.autofmt_xdate()

    plt.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('R0')
    fig.savefig(figname, bbox_inches='tight')

    
def hit_rate(df, true, upper, lower):
    return sum(sum([(df[true]<df[upper])&(df[true]>df[lower])]))/len(df)