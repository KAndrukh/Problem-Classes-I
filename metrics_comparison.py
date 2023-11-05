import gc
import os
import warnings
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sample_size = 28
plots_dir = os.path.join('out', 'plots', f'n{sample_size}', 'histograms')
calculations_dir = os.path.join('out', 'calculations', f'n{sample_size}')

os.makedirs(plots_dir, exist_ok=True)
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
                

def calc_comparisons(metric_info, grs, irs):
    m_file, m_name = metric_info

    with open(path.join(calculations_dir, m_file), 'rb') as f:
        df_orig = pd.concat([gr, ir, pd.DataFrame(np.fromfile(f), columns=[m_name])], axis=1)

    # filter to get only results for selected ratios
    df_orig = df_orig.loc[df_orig.ir.isin(irs) & df_orig.gr.isin(grs)]

    for metric in metrics.items():
        m_metr_file, m_metr_name = metric
        
        if m_metr_name == m_name:
            continue
        
        with open(path.join(calculations_dir, m_metr_file), 'rb') as f_comp:
            df_metric = pd.concat([gr, ir, pd.DataFrame(np.fromfile(f_comp), columns=[m_metr_name])], axis=1)
            
        df_metric = df_metric.loc[df_metric.ir.isin(irs) & df_metric.gr.isin(grs)]
        df_comp = pd.concat([gr, ir, pd.DataFrame(df_metric[m_metr_name] / df_orig[m_name] , columns=[m_metr_name + ' to ' + m_name])], axis=1)
        
        df_comp = df_comp.loc[df_comp.ir.isin(irs) & df_comp.gr.isin(grs)]
        df_comp = df_comp.replace(np.inf, np.nan)
        df_comp = df_comp.replace(-np.inf, np.nan)
      
        plots_dir_comp = os.path.join('out', 'plots_comp', f'n{sample_size}', 'histograms', m_name)
        calculations_dir_comp = os.path.join('out', 'calculations_comp', f'n{sample_size}', m_name)
                
        os.makedirs(plots_dir_comp, exist_ok=True)
        os.makedirs(calculations_dir_comp, exist_ok=True)
        
        with open(path.join(calculations_dir_comp, m_metr_name + ".bin"), "wb+") as f:
            df_comp.to_numpy().tofile(f)
            
        BINS = 109
        
        if not df_comp.empty:
            fig = plot_creation(df_comp.columns[2], df_comp, grs, irs, ratios_labels, BINS)
            fig.savefig(path.join(plots_dir_comp, f'{m_metr_name}_b{BINS}_histogram_titled.svg'), dpi=300)
            plt.close(fig)
        else:
            print(m_name, m_metr_name)
        


        
        
        del df_comp            
        del df_metric
        gc.collect()
    
    del df_orig
    gc.collect()    

grs = np.float16(ratios)
irs = np.float16(ratios[::-1])

for metric_info in metrics.items():
    calc_comparisons(metric_info, grs, irs)