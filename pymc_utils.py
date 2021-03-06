import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as pl
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class PyMCModel:
    def __init__(self, model, X, y, model_name='None', **model_kws):
        self.model = model(X, y, **model_kws)
        self.model.name = model_name
        
    def fit(self, n_samples=2000):
        with self.model:
            self.trace_ = pm.sample(n_samples)
    
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
    
    def predict(self, X, likelihood_name='y', **ppc_kws):
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
    
    def plot_model_fits(self, y_obs, y_pred=None, title=None, ax=None, range_=None,
                       loss_metric='rmse', **kwargs):
        loss_metric_val=-1
        likelihood_var_name = kwargs.pop('likelihood_var_name', 'mu')
        if y_pred is None:
            y_pred = self.trace_.get_values(likelihood_var_name)
        y_pred_mean = np.mean(y_pred, axis=0)
        try:
            rmse = np.sqrt(mean_squared_error(y_obs, y_pred_mean))
        except ValueError:
            mask = np.isnan(y_obs)
            y_pred_mean = np.ma.array(data=y_pred_mean, mask=mask).compressed()
            y_obs = np.ma.array(data=y_obs, mask=mask).compressed()
        finally:    
            r2 = r2_score(y_obs, y_pred_mean)
            if loss_metric == 'rmse':
                loss_metric_val = np.sqrt(mean_squared_error(y_obs, y_pred_mean))
            elif loss_metric == 'mse':
                loss_metric_val = mean_squared_error(y_obs, y_pred_mean)
            elif loss_metric == 'mae':
                loss_metric_val = mean_absolute_error(y_obs, y_pred_mean)
            
        if ax is None:
            _, ax = pl.subplots(figsize=(10, 10),)
        ax.set_title(title)
        ax.set_xlabel('model output mean')
        ax.set_ylabel('observed')
        ax.scatter(y_pred_mean, y_obs, color='k', alpha=0.5,
                     label='$r^2=%.2f$, %s=%.2f' %(r2, loss_metric, loss_metric_val));
        if range_ is None:
            axmin = min(y_pred_mean.min(), y_obs.min())
            axmax = max(y_pred_mean.max(), y_obs.max())
        else:
            axmin, axmax = range_
        ax.plot([axmin, axmax], [axmin, axmax], 'k--', label='1:1')
        ax.axis('equal')
        ax.legend(loc='best')
        f = pl.gcf()
        f.tight_layout()
        return ax
    
    def plot_model_ppc_stats(self, ppc, y_obs, alpha_level1=0.05,
                             alpha_level2=0.5, ax=None):
        if ax is None:
            _, ax = pl.subplots()
        iy = np.argsort(y_obs)
        ix = np.arange(iy.size)
        ppc_mean = ppc.mean(axis=0)
        ax.scatter(ix, y_obs.values[iy], label='observed', edgecolor='k', s=50,
                   color='steelblue')
        ax.scatter(ix, ppc_mean[iy], label='prediction mean', edgecolor='k', s=50,
                   color='red')
                 
        if alpha_level2:
            lik_hpd_2 = pm.hpd(ppc, alpha=alpha_level2)
            ax.fill_between(ix, y1=lik_hpd_2[iy, 0], y2=lik_hpd_2[iy, 1], alpha=0.5,
                            color='k',
                            label=f'prediction {1-alpha_level2:.2f}%CI',)
        if alpha_level1:
            lik_hpd_1 = pm.hpd(ppc, alpha=alpha_level1)
            ax.fill_between(ix, y1=lik_hpd_1[iy, 0], y2=lik_hpd_1[iy, 1], alpha=0.5,
                            color='k', label=f'prediction {1-alpha_level1:.2f}%CI',)
        ax.legend(loc='best')
        return ax
    
    def plot_model_fits2(self, y_obs, y_pred=None, title=None, ax=None, ci=0.95):
        if y_pred is None:
            y_pred = self.trace_.get_values('mu')
        y_obs = y_obs.values
        mask = np.logical_not(np.isnan(y_obs))
        y_obs = y_obs[mask]
        y_pred_mean = np.mean(y_pred, axis=0)[mask]
        y_pred_hpd = pm.hpd(y_pred, alpha=1-ci)[mask]
        xi = np.arange(y_obs.size)
        iy = np.argsort(y_obs)
        if ax is None:
            _, ax = pl.subplots(figsize=(12, 8),)
        ax.set_title(title)
        ax.plot(xi, y_obs[iy], marker='.', ls='',
                markeredgecolor='darkblue', markersize=13,
                label='observed')
        ax.plot(xi, y_pred_mean[iy], marker='o', color='indigo',
                ls='', markeredgecolor='k', alpha=0.5, label='predicted avg.')
        ax.fill_between(xi, y_pred_hpd[iy, 0], y_pred_hpd[iy, 1],
                        color='k', alpha=0.5,
                        label=f'{ci*100}%CI on pred.' );
        ax.legend(loc='best')
        return ax

    
def hs_regression(X, y_obs, ylabel='y', tau_0=None, regularized=False):
    """See Piironen & Vehtari, 2017 (DOI: 10.1214/17-EJS1337SI)"""
    if tau_0 is None:
        M = X.shape[1]
        m0 = M/2
        N = X.shape[0]
        tau_0 = m0 / ((M - m0) * np.sqrt(N))
    if regularized:
        slab_scale = kwargs.pop('slab_scale', 3)
        slab_scale_sq = slab_scale ** 2
        slab_df = kwargs.pop('slab_df', 8)
        half_slab_df = slab_df / 2
        with pm.Model() as mhsr:
            tau = pm.HalfCauchy('tau', tau_0)
            c_sq = pm.InverseGamma('c_sq', alpha=half_slab_df,
                                   beta=half_slab_df * slab_scale_sq)
            lamb_m = pm.HalfCauchy('lambda_m', beta=1)
            lamb_m_bar = tt.sqrt(c_sq) * lamb_m / (tt.sqrt(c_sq + 
                                                           tt.pow(tau, 2) *
                                                           tt.pow(lamb_m, 2)
                                                          )
                                                  )
            w = pm.Normal('w', mu=0, sd=tau*lamb_m_bar, shape=X.shape[1])
            mu_ = pm.Deterministic('mu', tt.dot(X, w))
            sig = pm.HalfCauchy('sigma', beta=10)
            y = pm.Normal('y', mu=mu_, sd=sig, observed=y_obs.squeeze())
        return mhsr
    else:
        with pm.Model() as mhs:
            tau = pm.HalfCauchy('tau', tau_0)
            lamb_m = pm.HalfCauchy('lambda_m', beta=1)
            w = pm.Normal('w', mu=0, sd = tau*lamb_m, shape=X.shape[1])
            mu_ = pm.Deterministic('mu', tt.dot(X, w))
            sig = pm.HalfCauchy('sigma', beta=10)
            y = pm.Normal('y', mu=mu_, sd=sig, observed=y_obs.squeeze())
        return mhs

    
def lasso_regression(X, y_obs, ylabel='y'):
    num_obs, num_feats = X.eval().shape
    with pm.Model() as mlasso:
        sd_beta = pm.HalfCauchy('sd_beta', beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=0, b=sd_beta)
        w = pm.Laplace('w', mu=0, b=sd_beta, shape=num_feats)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X, w))
        y = pm.Normal('y', mu=mu_, sd=sig, observed=y_obs.squeeze())
    return mlasso


def lasso_regr_impute_y(X, y_obs, ylabel='y'):
    num_obs, num_feats = X.eval().shape
    with pm.Model() as mlass_y_na:
        sd_beta = pm.HalfCauchy('sd_beta', beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=0, b=sd_beta)
        w = pm.Laplace('w', mu=0, b=sd_beta, shape=num_feats)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X, w))
        mu_y_obs = pm.Normal('mu_y_obs', 0.5, 1)
        sigma_y_obs = pm.HalfCauchy('sigma_y_obs', 1)
        y_obs_ = pm.Normal('y_obs', mu_y_obs, sigma_y_obs, observed=y_obs.squeeze())
        y = pm.Normal('y', mu=y_obs_, sd=sig)
    return mlass_y_na


def hier_lasso_regr(X, y_obs, add_bias=True, ylabel='y'):
    num_obs, num_feats = X.shape
    with pm.Model() as mlasso:
        hyp_beta = pm.HalfCauchy('hyp_beta', beta=2.5)
        hyp_mu = pm.HalfCauchy('hyp_mu', mu=0, beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=hyp_mu, b=hyp_beta)
        w = pm.Laplace('w', mu=hyp_mu, b=hyp_beta, shape=num_feats)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X, w))
        y = pm.Normal('y', mu=mu_, sd=sig, observed=y_obs.squeeze())
    return mlasso


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
