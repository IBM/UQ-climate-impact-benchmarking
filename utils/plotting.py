import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DayLocator, HourLocator, DateFormatter, drange


def plot_ensembles(df, variable, title, figname=None):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    for iforecast in df['forecast'].unique():
        df_plot = df[df['forecast']==iforecast].sort_values(by=['datetime'])
        ax.plot(df_plot['datetime'], df_plot[variable], alpha = 0.1, c='C0')

    fig.autofmt_xdate()
    ax.grid(axis='x', ls=":", color='0.5')
    ax.set_ylabel(variable)
    ax.set_xlabel("Date")
    ax.set_title(title)
    if figname:
        fig.savefig('figures/'+figname, bbox_inches='tight')
        

def plot_single_members(df, iforecasts, figname=None, quantiles_lu=[0.1587, 0.8413], labels_lu=['R0 lower', 'R0 upper']):

    fig, axes = plt.subplots(len(iforecasts), 1, figsize=(7,7), sharex=True, 
                             sharey=True)
    for ax, iforecast in zip(axes, iforecasts):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        df_visu = df[df['forecast']==iforecast].copy()
        ax.fill_between(df_visu['datetime'],
                        df_visu[labels_lu[0]], df_visu[labels_lu[1]], 
                        alpha=0.5, color="C3", label="Quantiles: "+
                        str(quantiles_lu[0])+' - '+str(quantiles_lu[1]))

        ax.plot(df_visu['datetime'], df_visu['R0'], label="Ensemble member"
                +str(iforecast))
        ax.set_ylabel('R0')
        ax.legend()
    
    fig.autofmt_xdate()
    ax.set_xlabel('Date')
    if figname:
        fig.savefig(figname, bbox_inches='tight')
    
    return fig
        
        
def plot_true_in_interval(df, sortby, figname=None, quantiles_lu=[0.1587, 0.8413], labels_lu=['R0 lower', 'R0 upper']):

    df_visu = df.sort_values(by=sortby).copy()
    
    fig = plt.figure(figsize=(5,4))
    plt.plot(range(len(df)), df_visu['R0'], "C0o", markersize=0.5, label="True values")
    #plt.plot(range(len(y_test)), df_visu_std['R0 mean'], "bo", markersize=0.5)

    plt.fill_between(range(len(df)),
        df_visu[labels_lu[0]], df_visu[labels_lu[1]], alpha=0.5, color="C3",
        label="Quantiles: "+str(quantiles_lu[0])+' - '+str(quantiles_lu[1]))

    plt.legend()
    plt.xlabel('Test samples sorted by $True\ ' + sortby +'$')
    plt.ylabel('R0')
    if figname:
        fig.savefig(figname, bbox_inches='tight')
    
    return fig        
        
    
def plot_cvar(df, figname=None):
    
    dfi = df[df['forecast']==1].copy()
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.scatter(df['datetime'], df['R0'], c='C0', marker='o', s=0.5, 
               label="True values")
    ax.plot(dfi['datetime'], dfi['cvarR0'], 'C0', label='true CVaR')

    ax.fill_between(dfi['datetime'], dfi['cvarR0 lower'], dfi['cvarR0 upper'], 
                alpha=0.5, color="C3", label="Predicted CVaR")

    fig.autofmt_xdate()
    plt.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('R0')

    if figname:
        fig.savefig(figname, bbox_inches='tight')    
    
    return fig 
    