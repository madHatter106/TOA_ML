class PyMCModel:
    def __init__(self, model, X, y, model_name='None'):
        self.model = model(X, y)
        self.model.name = model_name
        
    def fit(self, n_samples=2000):
        with self.model:
            self.trace_ = pm.sample(n_samples)
    
    def fit_ADVI(self, n_samples=2000, n_iter=100000, inference='advi'):
        with self.model:
            approx_fit = pm.fit(n=n_iter, method=inference)
            self.trace_ = approx_fit.sample(draws=n_samples)
    
    def show_model(self):
        return pm.model_to_graphviz(self.model)
    
    def predict(self, X, w_index=[]):
        ws = self.trace_['w'].T
        if len(w_index)>0:
            ws = ws[w_index]
        return (X.dot(ws) + self.trace_['alpha']).T
    
    def evaluate_fit(self, show_feats):
        return pm.traceplot(self.trace_, varnames=show_feats)
    
    def show_forest(self, show_feats, feat_labels=None):
        g = pm.forestplot(self.trace_, varnames=show_feats,
                             ylabels=feat_labels)
        f = pl.gcf()
        ax = f.get_axes()[1]
        ax.grid(axis='y')
        return g
    
    def plot_model_fits(self, y_obs, y_pred=None, title=None, ax=None):
        if y_pred is None:
            y_pred = self.trace_.get_values('mu')
        y_pred_mean = np.mean(y_pred, axis=0)
        try:
            rmse = np.sqrt(mean_squared_error(y_obs, y_pred_mean))
        except ValueError:
            mask = np.isnan(y_obs)
            y_pred_mean = np.ma.array(data=y_pred_mean, mask=mask).compressed()
            y_obs = np.ma.array(data=y_obs, mask=mask).compressed()
        finally:    
            r2 = r2_score(y_obs, y_pred_mean)
            rmse = np.sqrt(mean_squared_error(y_obs, y_pred_mean))
        if ax is None:
            _, ax = pl.subplots(figsize=(10, 10),)
        ax.set_title(title)
        ax.set_xlabel('modeled')
        ax.set_ylabel('observed')
        ax.scatter(y_pred_mean, y_obs, color='k', alpha=0.5,
                     label='$log_{10}(chl)$, $r^2=%.2f$, rmse=%.2f' %(r2, rmse));
        ax.plot([-1.5, 1.5], [-1.5, 1.5], 'k--', label='1:1')
        ax.axis('equal')
        ax.legend(loc='best')
        f = pl.gcf()
        f.tight_layout()
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
        ax.plot(xi, y.values[iy], marker='.', ls='',
                markeredgecolor='darkblue', markersize=13,
                label='observed')
        ax.plot(xi, y_pred_mean[iy], marker='o', color='indigo',
                ls='', markeredgecolor='k', alpha=0.5, label='predicted avg.')
        ax.fill_between(xi, y_pred_hpd[iy, 0], y_pred_hpd[iy, 1],
                        color='k', alpha=0.5,
                        label=f'{ci*100}%CI on pred.' );
        ax.legend(loc='best')
        return ax