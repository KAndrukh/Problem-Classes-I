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

plots_dir = os.path.join('out', 'sr_plots', f'n{sample_size}')
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

ratios = [1./(sample_size/2), 1./4, 1./2, 3./4, (sample_size/2 - 1)/(sample_size/2)]
ratios_labels = [f'1/{int(sample_size/2)}', '1/4', '1/2', '3/4', f'{(int(sample_size/2)) - 1}/{int(sample_size/2)}'] 

with open(path.join(calculations_dir, 'gr.bin'), 'rb') as f:
    gr = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['gr'])

with open(path.join(calculations_dir, 'ir.bin'), 'rb') as f:
    ir = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['ir'])
              
with open(path.join(calculations_dir, 'sr.bin'), 'rb') as f:
    sr = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['sr'])
    
def plot_histograms_with_stereotypical_ratio(df: pd.DataFrame, irs: np.float16, grs: np.float16, plots_dir: str):
    plots_dir = path.join(plots_dir, "SR of meassures")
    os.makedirs(plots_dir, exist_ok=True)

    ir_labels = ratios_labels
    gr_labels = ratios_labels

    rows = len(ir_labels)
    cols = len(gr_labels)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex='col', sharey='row', gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.text(0.5, 0.95, f'{df.columns[0]}', ha='center', va='center', fontsize=24)
    
    #merging values of GR, IR and SR to an existing dataframe to filter it by these ratios
    df = pd.concat([df, gr, ir, sr], axis=1)

    for i, ir_val in enumerate(irs):
        for j, gr_val in enumerate(grs):
            #filter values by GR and IR
            filtered_data = df[(df['ir'] == ir_val) & (df['gr'] == gr_val)]
    
            #we drop GR, IR and SR positive to not include them in the correlation 
            filtered_data = filtered_data.drop(columns=['ir', 'gr'])
                        
            ax = axs[rows - 1 - i, j]
            
            #generating histplot for defined GR and IR
            if j == grs[-1]:
                sns.histplot(filtered_data, x=filtered_data[filtered_data.columns[0]], cbar=True, ax=ax)
            else:
                sns.histplot(filtered_data, x=filtered_data[filtered_data.columns[0]], ax=ax)  
                
            
            del filtered_data
            gc.collect()
              
    #styling
    for ax, gr_label in zip(axs[-1], gr_labels):
        ax.set_xlabel(f'GR={gr_label}', fontsize=30)
    for ax, ir_label in zip(axs[:, 0], ir_labels[::-1]):
        ax.set_ylabel(f'IR={ir_label}', fontsize=30)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(path.join(plots_dir, f'{df.columns[0]}.png'), dpi=300)
    plt.savefig(path.join(plots_dir, f'{df.columns[0]}.svg'), dpi=300)

    del df
    gc.collect()

def plot_histograms_1d(df: pd.DataFrame, ratio: str, plots_dir: str, irs: np.float16, grs: np.float16, srs: np.float16):
    plots_dir = path.join(plots_dir, f"{ratio.upper()} vs SR")
    os.makedirs(plots_dir, exist_ok=True)

    if ratio == 'ir':
        rs = ir
        rss = irs
    else:
        rs = gr
        rss = grs

    rs_labels = ratios_labels
    sr_labels = ratios_labels

    rows = len(rs_labels)
    cols = len(sr_labels)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex='col', sharey='row', gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.text(0.5, 0.95, f'{df.columns[0]}', ha='center', va='center', fontsize=24)
    
    #merging values of ratio and SR to an existing dataframe to filter it by these ratios
    df = pd.concat([df, rs, sr], axis=1)

    for i, rs_val in enumerate(rss):
        for j, sr_val in enumerate(srs):
            #filter values by ratio and SR
            filtered_data = df[(df[ratio] == rs_val) & (df['sr'] - sr_val < 0.02)]
    
            #we drop ratio and SR to not include them in the correlation 
            filtered_data = filtered_data.drop(columns=[ratio, 'sr'])
                        
            # print(rs_val, sr_val, filtered_data)

            ax = axs[rows - 1 - i, j]
            
            #generating histplot for defined ratio and SR
            sns.histplot(filtered_data, x=filtered_data[filtered_data.columns[0]], ax=ax)
                
            
            del filtered_data
            gc.collect()
              
    #styling
    for ax, sr_label in zip(axs[-1], sr_labels):
        ax.set_xlabel(f'SR={sr_label}', fontsize=30)
    for ax, rs_label in zip(axs[:, 0], rs_labels[::-1]):
        ax.set_ylabel(f'{ratio.upper()}={rs_label}', fontsize=30)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(path.join(plots_dir, f'{df.columns[0]}.png'), dpi=300)
    plt.savefig(path.join(plots_dir, f'{df.columns[0]}.svg'), dpi=300)

    del df
    gc.collect()
        

def calc_comparisons(grs, irs, srs):    
    for metric in metrics.items():
        m_metr_file, m_metr_name = metric
        df = pd.DataFrame()
        #adding a fairness meassure as a separate column to the dataframe
        with open(path.join(calculations_dir, m_metr_file), 'rb') as f_comp:
            df[m_metr_name] = np.fromfile(f_comp)    
            
        # plot_histograms_with_stereotypical_ratio(df, irs, grs, plots_dir)
        plot_histograms_1d(df, 'ir', plots_dir, irs, grs, srs)
        plot_histograms_1d(df, 'gr', plots_dir, irs, grs, srs)

        del df  
        gc.collect()    

grs = np.float16(ratios)
irs = np.float16(ratios)
srs = np.float16(ratios)

calc_comparisons(grs, irs, srs)