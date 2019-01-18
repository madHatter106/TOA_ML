import pymc3 as pm
import matplotlib.pyplot as pl

class PyMCModel:
    def __init__(self, model, X, y, model_name='None', **model_kws):
        self.model = model(X, y, **model_kws)
        self.model.name = model_name

    def fit(self, n_samples=2000, **sample_kws):
        with self.model:
            self.trace_ = pm.sample(n_samples, **sample_kws)

    def fit_ADVI(self, n_samples=2000, n_iter=100000, inference='advi', **fit_kws):
        with self.model:
            self.approx_fit = pm.fit(n=n_iter, method=inference, **fit_kws)
            self.trace_ = self.approx_fit.sample(draws=n_samples)

    def show_model(self, save=False, view=True, cleanup=True):
        model_graph = pm.model_to_graphviz(self.model)
        if save:
            model_graph.render(save, view=view, cleanup=cleanup)
        if view:
            return model_graph

    def predict(self, likelihood_name='likelihood', **ppc_kws):
        ppc_ = pm.sample_ppc(self.trace_, model=self.model,
                             **ppc_kws)[likelihood_name]
        return ppc_

    def evaluate_fit(self, show_feats):
        return pm.traceplot(self.trace_, varnames=show_feats)

    def show_forest(self, show_feats, feat_labels=None):
        g = pm.forestplot(self.trace_, varnames=show_feats,
                             ylabels=feat_labels)
        f = pl.gcf()
        try:
            ax = f.get_axes()[1]
        except IndexError:
            ax = f.get_axes()[0]
        ax.grid(axis='y')
        return g

def subset_significant_feature(trace, labels_list, alpha=0.05, vars_=None):
    if vars_ is None:
        vars_ = ['sd_beta', 'sigma', 'bias', 'w']
    dsum = pm.summary(trace, varnames=vars_, alpha=alpha)
    lbls_list = ['w[%s]' %lbl for lbl in labels_list]
    dsum.index = vars_[:-1] + lbls_list
    hpd_lo, hpd_hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    if str(hpd_lo).split('.')[1] == '0':
        hpd_lo = int(hpd_lo)
    if str(hpd_hi).split('.')[1] == '0':
        hpd_hi = int(hpd_hi)
    dsum_subset = dsum[(((dsum[f'hpd_{hpd_lo}']<0)&(dsum[f'hpd_{hpd_hi}']<0))|
                    ((dsum[f'hpd_{hpd_lo}']>0) & (dsum[f'hpd_{hpd_hi}']>0))
                   )]
    pattern1 = r'w\s*\[([a-z_\sA-Z0-9]+)\]'
    return list(dsum_subset.index.str.extract(pattern1).dropna().values.flatten())


def create_smry(trc, labels, vname=['w']):
    ''' Conv fn: create trace summary for sorted forestplot '''
    dfsm = pm.summary(trc, varnames=vname)
    dfsm.rename(index={wi: lbl for wi, lbl in zip(dfsm.index, feature_labels)},
                inplace=True)
    #dfsm.sort_values('mean', ascending=True, inplace=True)
    dfsm['ypos'] = np.linspace(1, 0, len(dfsm))
    return dfsm
