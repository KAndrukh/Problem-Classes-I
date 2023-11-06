import gc
import os
import warnings
from os import path
from cycler import V

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

sample_size = 28
calculations_dir = os.path.join('out', 'calculations', f'n{sample_size}')

os.makedirs(calculations_dir, exist_ok=True)

metrics = {
    'acc_equality_diff.bin': 'Accuracy equality',
    'equal_opp_diff.bin': 'Equal opportunity',
    'pred_equality_diff.bin': 'Predictive equality',
    'stat_parity.bin': 'Statistical parity',
    'neg_pred_parity_diff.bin': 'Negative predictive parity',
    'pos_pred_parity_diff.bin': 'Positive predictive parity'
}

sample_size = 28
ratios = [1./28, 1./4, 1./2, 3./4, 27./28] if sample_size == 56 else [1./14, 1./4, 1./2, 3./4, 13./14]
ratios_labels = ['1/28', '1/4', '1/2', '3/4', '27/28'] if sample_size == 56 else ['1/14', '1/4', '1/2', '3/4', '13/14']

with open(path.join(calculations_dir, 'gr.bin'), 'rb') as f:
    gr = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['gr'])

with open(path.join(calculations_dir, 'ir.bin'), 'rb') as f:
    ir = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['ir'])


def plot_creation(m_name, df, grs, irs, ratios_labels, bins_n):
    ir_labels = ratios_labels[::-1]
    gr_labels = ratios_labels
    
    mosaic = [
        [f'a{i}{g}{x}'
         for g in range(len(grs))
         for x in ['', 'n']]
         for i in range(len(irs))
    ]

    fig, axs = plt.subplot_mosaic(mosaic,
                                  width_ratios=[50, 1]*len(grs),
                                  sharex=False, sharey=True,
                                  layout='constrained',
                                  figsize=(20, 10),
                                  gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.suptitle(f'{m_name}')

    for i, ir_val in enumerate(irs):
        for g, gr_val in enumerate(grs):

            # separate nans and numbers
            df_tmp = df.loc[(df.ir == ir_val) & (df.gr == gr_val)]
            total = df_tmp.shape[0]

            df_not_nan = df_tmp.loc[np.logical_not(np.isnan(df_tmp[m_name]))]
            nan_prob = df_tmp.loc[np.isnan(df_tmp[m_name])].shape[0] / total if total > 0 else 0

            # prepare data for plotting
            binned, edges = np.histogram(df_not_nan[m_name], bins=bins_n)
            binned = binned / total

            # plot not nans
            axs[f'a{i}{g}'].hist(edges[:-1], edges, weights=binned, fc='black', ec='black')
            axs[f'a{i}{g}'].spines[['top', 'right']].set_visible(False)

            # plot nans - without drawing the full axis frame
            axs[f'a{i}{g}n'].bar(0, nan_prob, fc='red', ec='red', width=0.1, lw=0)
            axs[f'a{i}{g}n'].spines[['top', 'left']].set_visible(False)

            # styling
            if g == 0:
                axs[f'a{i}{g}'].set_ylabel(f'IR = {ir_labels[i]}')
            if i == 0:
                axs[f'a{i}{g}'].set_title(f'GR = {gr_labels[g]}')
            if i == len(irs) - 1:   # last row
                axs[f'a{i}{g}n'].set_xticks([0], ['Undef.'])
            else:
                axs[f'a{i}{g}'].set_xticklabels([])
                axs[f'a{i}{g}n'].set_xticks([0], [''])

    return fig
              
def heatmap_creation(df_comp, m_name, m_metr_name, bins_n):
    plots_dir_comp = os.path.join(plots_dir_comp, 'Corellations')
    os.makedirs(plots_dir_comp, exist_ok=True)
    
    #calculate correlations using Spearman method
    df_corr = df_comp.corr('spearman')
    
    #set titles and save the heatmap
    fig_heatmap = sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f")
    fig_heatmap.set_title(m_metr_name + ' To ' + m_name)
    fig_heatmap.set_xticklabels(['gr', 'ir', m_metr_name[:3] + 'To' + m_name[:3]], rotation=0)
    fig_heatmap.set_yticklabels(['gr', 'ir', m_metr_name[:3] + 'To' + m_name[:3]])
    fig_heatmap.figure.savefig(path.join(plots_dir_comp, f'Correlation_{m_metr_name[:3] + 'To' + m_name[:3]}_b{bins_n}_histogram_titled.svg'), dpi=300)

    plt.close(fig_heatmap.figure)      
    
    del df_corr
    gc.collect()

def calc_comparisons(metric_info, grs, irs):
    m_file, m_name = metric_info

    with open(path.join(calculations_dir, m_file), 'rb') as f:
        df_orig = pd.concat([gr, ir, pd.DataFrame(np.fromfile(f), columns=[m_name])], axis=1)

    # filter to get only results for selected ratios
    df_orig = df_orig.loc[df_orig.ir.isin(irs) & df_orig.gr.isin(grs)]

    #go through each metric
    for metric in metrics.items():
        m_metr_file, m_metr_name = metric
        
        #ignore the comparisons between the same dataset
        if m_metr_name == m_name:
            continue
        
        #create a dataframe for the current compared metric
        with open(path.join(calculations_dir, m_metr_file), 'rb') as f_comp:
            df_metric = pd.concat([gr, ir, pd.DataFrame(np.fromfile(f_comp), columns=[m_metr_name])], axis=1)
            
        df_metric = df_metric.loc[df_metric.ir.isin(irs) & df_metric.gr.isin(grs)]
        
        #create a dataframe with comparison of outputs of two metrics
        df_comp = pd.concat([gr, ir, pd.DataFrame(df_metric[m_metr_name] / df_orig[m_name] , columns=[m_metr_name + ' to ' + m_name])], axis=1)
        df_comp = df_comp.loc[df_comp.ir.isin(irs) & df_comp.gr.isin(grs)]
        #filter all infinite values
        df_comp = df_comp.replace(np.inf, np.nan)
        df_comp = df_comp.replace(-np.inf, np.nan)
      
        plots_dir_comp = os.path.join('out', 'plots_comp', f'n{sample_size}', 'histograms', m_name)
        calculations_dir_comp = os.path.join('out', 'calculations_comp', f'n{sample_size}', m_name)
                
        os.makedirs(plots_dir_comp, exist_ok=True)
        os.makedirs(calculations_dir_comp, exist_ok=True)
        
        #save the comparison dataframe in .bin file
        with open(path.join(calculations_dir_comp, m_metr_name + ".bin"), "wb+") as f:
            df_comp.to_numpy().tofile(f)
            
        BINS = 109
        
        #create and save a histogram for the comparison dataframe
        if not df_comp.empty:
            fig = plot_creation(df_comp.columns[2], df_comp, grs, irs, ratios_labels, BINS)
            fig.savefig(path.join(plots_dir_comp, f'{m_metr_name}_b{BINS}_histogram_titled.svg'), dpi=300)
            plt.close(fig)
        
        #save a heatmap of correlations in the comparison dataframe
        heatmap_creation(df_comp, m_name, m_metr_name, BINS)
        
        del df_comp            
        del df_metric
        gc.collect()
    
    del df_orig
    gc.collect()    

grs = np.float16(ratios)
irs = np.float16(ratios[::-1])

for metric_info in metrics.items():
    calc_comparisons(metric_info, grs, irs)