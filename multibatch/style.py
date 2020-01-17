import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler

### MATPLOTLIB SETUP ###

#default_color = '#aee4c8'
default_color = '#dddddd'
default_font = {'family' : 'sans-serif',
              'weight' : 'normal',
               'size'   : 14}

def style_pd():
    ''' default styling for pd '''
    pd.set_option('display.max_colwidth', -1)

def style_plt():
    ''' default styling for plt '''
    # general
    plt.rc('font', **default_font)
    plt.rc('lines', linewidth=4)

    # figure
    plt.rc('figure', titlesize='small', figsize=[4, 2])
    plt.rc('image', aspect=1.75)

    # general graphs
    plt.rc('axes', edgecolor='black', facecolor="none", axisbelow=True, grid=False, \
        labelpad=4.0, linewidth=1.0, labelcolor='black', labelsize='medium', labelweight="normal", \
            prop_cycle=cycler('color', [default_color]))
    plt.rc('axes.spines', top=False, right=False)
    plt.rc('xtick.major', width=1.5, size=8.0)
    plt.rc('xtick', direction="inout", labelsize='medium', color='black')
    plt.rc('ytick.major', width=1.5, size=5.0)
    plt.rc('ytick', direction="out", labelsize='small', color='black')
    plt.rc('grid', color='w', linestyle='solid', lw=1.5)

    # boxplots
    plt.rc('boxplot', showcaps=True, showmeans=False, showbox=True, whiskers=True, patchartist=True)
    plt.rc('boxplot.whiskerprops', color='black', linewidth=1.5)
    plt.rc('boxplot.boxprops', color='black', linewidth=1.3)
    plt.rc('patch', facecolor=default_color)
    plt.rc('boxplot.medianprops', color="black", linewidth=5.0)
    plt.rc('boxplot.flierprops', markersize=3.0)
    plt.rc('errorbar', capsize=15.0)