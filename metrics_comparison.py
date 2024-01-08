import gc
import os
from random import sample
import warnings
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

sample_size = 24
calculations_dir = os.path.join('out', 'calculations', f'n{sample_size}')

os.makedirs(calculations_dir, exist_ok=True)

metrics = {
    'acc_equality_diff.bin': 'Accuracy equality',
    'equal_opp_diff.bin': 'Equal opportunity',
    'pred_equality_diff.bin': 'Predictive equality',
    'stat_parity.bin': 'Statistical parity',
    'neg_pred_parity_diff.bin': 'Negative predictive parity',
    'pos_pred_parity_diff.bin': 'Positive predictive parity',
    'pred_acc_equality_diff.bin': 'Predictive accuracy equality',
    'neg_pred_equality_diff.bin': 'Negative predictive equality',
    'pos_pred_equality_diff.bin': 'Positive predictive equality',
}

ratios = [1./(sample_size/2), 1./4, 1./2, 3./4, (sample_size/2- 1)/(sample_size/2)]
ratios_labels = [f'1/{int(sample_size/2)}', '1/4', '1/2', '3/4', f'{int(sample_size/2) - 1}/{int(sample_size/2)}'] 

with open(path.join(calculations_dir, 'gr.bin'), 'rb') as f:
    gr = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['gr'])

with open(path.join(calculations_dir, 'ir.bin'), 'rb') as f:
    ir = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['ir'])
              
with open(path.join(calculations_dir, 'sr.bin'), 'rb') as f:
    sr = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['sr'])
    
def heatmap_creation(df_comp: pd.DataFrame, plots_dir_comp: str, method: str):
    os.makedirs(plots_dir_comp, exist_ok=True)
    
    #calculate correlations using Spearman method
    df_corr = df_comp.corr(method)
    matrix = np.triu(df_corr)
    columns = df_corr.columns
    
    #set titles and save the heatmap
    fig_heatmap = sns.heatmap(df_corr, annot=True, vmin=-1, vmax=1, mask=matrix, cmap='coolwarm', fmt=".2f")
    fig_heatmap.set_title(f'Correlations of fairness measures({method})')
    fig_heatmap.set_xticklabels(columns, rotation=0)
    fig_heatmap.set_yticklabels(columns, rotation=0)
    fig_heatmap.figure.savefig(path.join(plots_dir_comp, f'Correlations_general_{method}.svg'), dpi=300)

    plt.close(fig_heatmap.figure)
    
    # df_corr.to_csv(path.join(plots_dir_comp, f'Correlations_general_{method}.csv'))
    
    del df_corr
    gc.collect()
    
def heatmap_with_distributions(df_comp: pd.DataFrame, irs:np.float16, grs:np.float16, plots_dir_comp:str, method: str):
    ir_labels = ratios_labels
    gr_labels = ratios_labels

    rows = len(ir_labels)
    cols = len(gr_labels)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex='col', sharey='row', gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

    #merging values of GR and IR to an existing dataframe to filter it by these ratios
    df_comp = pd.concat([gr, ir, df_comp], axis=1)

    plots_dir_comp = path.join(plots_dir_comp)
    os.makedirs(plots_dir_comp, exist_ok=True)

    for i, ir_val in enumerate(irs):
        for j, gr_val in enumerate(grs):
            #filter values by GR and IR
            filtered_data = df_comp[(df_comp['ir'] == ir_val) & (df_comp['gr'] == gr_val)]
            #we drop GR and IR to not include them in the correlation hitmap
            filtered_data = filtered_data.drop(columns=['ir', 'gr'])

            df_corr = filtered_data.corr(method=method)
            # df_corr.to_csv(path.join(plots_dir_comp,
            #                          f'Correlations_IR={ratios_labels[i].replace('/', '_')}_GR={ratios_labels[j].replace('/', '_')}.csv'))
            #masking the upper triangle
            mask = np.tri(df_corr.shape[0], k=-1).astype(bool)

            ax = axs[rows - 1 - i, j]

            sns.heatmap(df_corr, annot=True, vmin=-1, vmax=1, mask=~mask, cmap='coolwarm', fmt=".2f", ax=ax)
            
            #styling
            if i == len(ir_labels)-1: 
                ax.xaxis.set_label_position('top')
                ax.set_xlabel(f'GR={gr_labels[j]}', rotation=0, ha='center', fontsize=18)
            if j == 0:
                ax.set_ylabel(f'IR={ir_labels[i]}', rotation=90, ha='center', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(path.join(plots_dir_comp, f'Correlations_ratios_{method}.svg'), dpi=300)

    del df_corr
    del df_comp
    gc.collect()
    
def plot_histograms_with_stereotypical_ratio(df: pd.DataFrame, irs: np.float16, grs: np.float16, plots_dir_comp: str, sr: np.float16):
    plots_dir_comp = path.join(plots_dir_comp, "SR of meassures")
    os.makedirs(plots_dir_comp, exist_ok=True)

    ir_labels = ratios_labels
    gr_labels = ratios_labels

    rows = len(ir_labels)
    cols = len(gr_labels)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex='col', sharey='row', gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.text(0.5, 0.95, f'{df.columns[0]}', ha='center', va='center', fontsize=30)
    
    #merging values of GR, IR and SR positive and negative  to an existing dataframe to filter it by these ratios
    df = pd.concat([df, gr, ir, sr], axis=1)

    for i, ir_val in enumerate(irs):
        for j, gr_val in enumerate(grs):
            #filter values by GR and IR
            filtered_data = df[(df['ir'] == ir_val) & (df['gr'] == gr_val)]
    
            #we drop GR, IR and SR positive to not include them in the correlation 
            # filtered_data = filtered_data.drop(columns=['ir', 'gr'])
                        
            ax = axs[rows - 1 - i, j]
            
            #generating histplot for defined GR and IR
            sns.histplot(filtered_data, x=filtered_data[filtered_data.columns[0]], y='sr', cmap="coolwarm", ax=ax)
            
            del filtered_data
            gc.collect()
            
            
    #styling
    for ax, gr_label in zip(axs[-1], gr_labels):
        ax.set_xlabel(f'GR={gr_label}', fontsize=30)
    for ax, ir_label in zip(axs[:, 0], ir_labels[::-1]):
        ax.set_ylabel(f'IR={ir_label}', fontsize=30)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(path.join(plots_dir_comp, f'{df.columns[0]}.png'), dpi=300)
    plt.savefig(path.join(plots_dir_comp, f'{df.columns[0]}.svg'), dpi=300)

    del df
    gc.collect()
        

def calc_comparisons(grs, irs):    
    df = pd.DataFrame()
    
    #go through each fairness measure
    for metric in metrics.items():
        m_metr_file, m_metr_name = metric

        #adding a fairness meassure as a separate column to the dataframe
        with open(path.join(calculations_dir, m_metr_file), 'rb') as f_comp:
            words = m_metr_name.title().split()
            
            name = ''.join(word[:3] for word in words)
            df[name] = np.fromfile(f_comp)            

    plots_dir_comp = os.path.join('out', 'plots_comp', f'n{sample_size}')
    
    #creates a heatmap of general correlations in a dataframe
    heatmap_creation(df, plots_dir_comp, 'pearson')
    heatmap_creation(df, plots_dir_comp, 'spearman')

    #creates a heatmap of correlations in a dataframe with different GR and IR
    heatmap_with_distributions(df, irs, grs, plots_dir_comp, 'pearson')
    heatmap_with_distributions(df, irs, grs, plots_dir_comp, 'spearman')
    del df
    gc.collect()    

grs = np.float16(ratios)
irs = np.float16(ratios)

calc_comparisons(grs, irs)