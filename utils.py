import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from seaborn import heatmap
from cmocean import cm as cmo


def PlotPCARes(pca_machine, threshold=0.85, alpha=1,
               num_pca_disp=None, ax=None):
    """Plot PCA results."""
    if ax is None:
        _, ax = pl.subplots(figsize=(12, 10))
    cum_expl_var = np.cumsum(pca_machine.explained_variance_ratio_)
    if num_pca_disp is None:
        num_pca_disp = np.argmax(cum_expl_var > 0.999) + 1

    ax.bar(range(1, num_pca_disp+1),
           pca_machine.explained_variance_ratio_[:num_pca_disp],
           align='center', color='skyblue',
           label='PC explained_variance')
    ax.step(range(1, num_pca_disp+1),
            np.cumsum(pca_machine.explained_variance_ratio_[:num_pca_disp]),
            where='mid',
            label='cumulated variance')
    ax.hlines(threshold, 0, num_pca_disp+2, linestyles='--', linewidth=2,
              label='selection cutoff: %.2f' % threshold)
    ax.set_xticks(np.arange(1, num_pca_disp+1))
    ax.set_xticklabels(['PC%d' % i for i in range(1, num_pca_disp+1)],
                       rotation=45)
    ax.set_xlim((0.5, 0.5+num_pca_disp))
    ax.set_ylim((0, 1))
    ax.set_title('PCA Explained Variance')
    ax.legend(loc='center right')

    def PlotCrossCorr(pca_data_, df, ax=None, cbax=None):
        df_rrs = df.filter(regex='rrs')
        rrs_cols = df_rrs.columns.tolist()
        pc_num = 6
        df_pca = pd.DataFrame(pca_data_[:, :pc_num],
                              columns=['PC%d' % (i+1) for i in range(pc_num)],
                              index=df.index)
        dfrrs_w_pca = pd.merge(df_pca, df.filter(regex='rrs'), 'outer',
                               left_index=True,
                               right_index=True)

        corr_w_pca = dfrrs_w_pca.corr().T
        corr_w_pca.drop(df_pca.columns, axis=0, inplace=True)
        corr_w_pca.drop(rrs_cols, axis=1, inplace=True)
        if ax is None:
            _, ax = pl.subplots(figsize=(20, 5))
        heatmap(corr_w_pca, cmap=cmo.balance, annot=True, vmin=-1, vmax=1,
                ax=ax, cbar_ax=cbax)
