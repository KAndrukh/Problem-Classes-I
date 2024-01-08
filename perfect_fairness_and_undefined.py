#!/usr/bin/env python
# coding: utf-8

# # Probability of Perfect Fairness and undefined values
# 
# calculations for different metrics, group ratios and imbalance ratios

# In[ ]:


import os
import warnings
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from pyparsing import srange

warnings.filterwarnings('ignore')
plt.style.use('default')


sample_size = 24
# setting epsilon to another (small positive) value allows to calculate the probability of being epsilon-close to perfect fairness
epsilon = 0

calculations_dir = path.join('out', 'calculations', f'n{sample_size}')
os.makedirs(calculations_dir, exist_ok=True)
dataset_path = path.join('out', f'Set(08,{sample_size}).bin')


## Calculate values for visualizations
diff_metrics = {  # { file: metric name }
    'pos_pred_parity_diff.bin': 'Positive predictive parity difference',
    'acc_equality_diff.bin': 'Accuracy equality difference',
    'stat_parity.bin': 'Statistical parity difference',
    'equal_opp_diff.bin': 'Equal opportunity difference',
    'neg_pred_parity_diff.bin': 'Negative predictive parity difference',
    'pred_equality_diff.bin': 'Predictive equality difference',
    'pred_acc_equality_diff.bin': 'Predictive accuracy equality difference',
    'pos_pred_equality_diff.bin': 'Positive predictive equality difference',
    'neg_pred_equality_diff.bin': 'Negative predictive equality difference',
}


def calculate_ppf_diff(df, metrics, ratio_type, epsilon=0):
    pf_probs, nan_probs = {}, {}

    if epsilon == 0:
        compute_diff_prob = lambda df: np.sum(df['diff'] == 0) / len(df)
    else:
        compute_diff_prob = lambda df: np.sum(np.abs(df['diff']) < epsilon) / len(df)

    if ratio_type == 'sr':
        num_bins = 50
        
        hist, bin_edges = np.histogram(df['sr'], bins=num_bins, range=(df['sr'].min(), df['sr'].max()))
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        # Use np.clip to ensure indices are within bounds
        indices = np.clip(np.digitize(df['sr'], bin_edges) - 1, 0, len(bin_centers) - 1)
        df['sr'] = bin_centers[indices]

    for metric_file, metric_name in metrics.items():
        with open(path.join(calculations_dir, metric_file), 'rb') as f:
            diff_metric = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['diff'])
        df = pd.concat([df, diff_metric], axis=1)

        pf_bygroup = list()
        nans_bygroup = list()

        for gn, group in df.groupby(ratio_type):
            if group['diff'].isna().all():
                pf_bygroup.append([gn, np.nan])
            else:
                pf_bygroup.append([gn, compute_diff_prob(group)])
            nans_bygroup.append([gn, group['diff'].isna().sum() / group.shape[0]])

        pf_bygroup = pd.DataFrame(pf_bygroup, columns=[ratio_type, 'diff'])
        pf_probs[metric_name] = pf_bygroup['diff']

        nans_bygroup = pd.DataFrame(nans_bygroup, columns=[ratio_type, 'diff'])
        nan_probs[metric_name] = nans_bygroup['diff']

        # the dataframe (first col) can be reused for the next metric
        df.drop('diff', axis=1, inplace=True)

    pf_probs[ratio_type] = pf_bygroup[ratio_type]
    pf_df = pd.DataFrame(pf_probs).reset_index()
    pf_df.to_csv(path.join(calculations_dir, f'perfect_fairness_{ratio_type}_eps{epsilon}.csv'), index=False)

    nan_probs[ratio_type] = nans_bygroup[ratio_type]
    nan_df = pd.DataFrame(nan_probs).reset_index()
    nan_df.to_csv(path.join(calculations_dir, f'nans_{ratio_type}.csv'), index=False)


for ratio in ['ir', 'gr', 'sr']:
    with open(path.join(calculations_dir, f'{ratio}.bin'), 'rb') as f:
        df = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=[ratio])
    calculate_ppf_diff(df, diff_metrics, ratio, epsilon)
del df


# # Plotting
# 
# reading data from the csv files created above
calculations_dir = path.join('out', 'calculations', f'n{sample_size}')
plots_dir = path.join('out', 'plots', f'n{sample_size}', 'perfect_fairness')
os.makedirs(plots_dir, exist_ok=True)

epsilons = [0,]
ratio_types = ['sr', 'gr', 'ir']


dfs = {(ratio_type, epsilon): pd.read_csv(path.join(calculations_dir, f'perfect_fairness_{ratio_type}_eps{epsilon}.csv'))
       for ratio_type in ratio_types
       for epsilon in epsilons}

# colour scheme inspired by https://personal.sron.nl/~pault/
diff_metrics_styles = {
    'Accuracy equality difference': {'color': '#6699CC', 'marker': '*'},
    'Statistical parity difference': {'color': '#994455', 'marker': '.'},

    'Equal opportunity difference': {'color': '#004488', 'marker': 'v'},
    'Predictive equality difference': {'color': '#997700', 'marker': 'x'},

    'Negative predictive parity difference': {'color': '#EECC66', 'marker': '+'},
    'Positive predictive parity difference': {'color': '#EE99AA', 'marker': 'o'},
    
    'Predictive accuracy equality difference': {'color': '#8833AA', 'marker': '^'},
    'Positive predictive equality difference': {'color': '#009911', 'marker': 's'},
    'Negative predictive equality difference': {'color': '#FF2E2E', 'marker': 'd'},
}

x_description = {
    'gr': 'Protected group ratio (GR)',
    'ir': 'Imbalance ratio (IR)',
    'sr': 'Stereotypical ratio (SR)'
}

def melt_df(df, base_metric):
    temp = df.pop(base_metric)
    length = len(df.columns)
    df = pd.melt(df)
    df[base_metric] = list(temp) * length
    return df

def plot_mlp(df, base_metric, color_mapping, title='Proportion of perfect fairness', y_max=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for col in color_mapping.keys():
        ax.plot(df[base_metric], df[col], label=col.replace('difference', ''), alpha=.5, **color_mapping[col])

    if y_max is not None:
        ax.set_ylim(0, y_max)

    ax.set_xlabel(x_description[base_metric])
    ax.set_ylabel('Probability of perfect fairness')
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    return fig

for ratio_type in ratio_types:
    for eps in epsilons:
        fig = plot_mlp(dfs[(ratio_type, eps)], ratio_type, diff_metrics_styles,
                       title='')
        fig.savefig(path.join(plots_dir, f'ppf_{ratio_type}_eps{eps}.svg'),
                    dpi=300)


# # Probability of NaN - plotting
def nan_probability(df, base_metric, color_mapping, title='Probability of NaN', y_max=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for col in color_mapping.keys():
        ax.plot(df[base_metric], df[col], label=col.replace('difference', ''), alpha=.5, **color_mapping[col])

    if y_max is not None:
        ax.set_ylim(0, y_max)

    ax.set_xlabel(x_description[base_metric])
    ax.set_ylabel('Probability of undefined metric value')
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.legend()
    return fig

nan_dfs = {ratio_type: pd.read_csv(path.join(calculations_dir, f'nans_{ratio_type}.csv'))
           for ratio_type in ratio_types}

for ratio_type in ratio_types:
    fig = nan_probability(nan_dfs[ratio_type], ratio_type, diff_metrics_styles,
                          title='')
    fig.savefig(path.join(plots_dir, f'nan_{ratio_type}_line.svg'), dpi=300)

for ratio_type in ratio_types:
    fig = nan_probability(nan_dfs[ratio_type], ratio_type, diff_metrics_styles,
                          title=f'Probability of NaN for given value of {ratio_type.upper()}', y_max=.02)
    fig.savefig(path.join(plots_dir, f'nan_{ratio_type}_zoom_line.svg'), dpi=300)

