"""Runs ARD BNN"""

import pickle

from loguru import logger

import pandas as pd

from theano import shared
from pymc_models import PyMCModel
from pymc_models import hs_regression

from sklearn.preprocessing import PolynomialFeatures
if __name__ == "__main__":
    logger.add("linreg_wi_{time}.log")
    # load datasets
    with open('./pickleJar/AphiTrainTestSplitDataSets.pkl', 'rb') as fb:
        datadict = pickle.load(fb)
    X_s_train = datadict['x_train_s']
    y_train = datadict['y_train']
    X_s_test = datadict['x_test_s']
    y_test = datadict['y_test']

    poly_tranf = PolynomialFeatures(interaction_only=True, include_bias=False)

    X_s_train_w_int = pd.DataFrame(poly_tranf.fit_transform(X_s_train),
                                   columns=poly_tranf.get_feature_names(input_features=
                                                                        X_s_train.columns),
                                   index=X_s_train.index)
    X_s_test_w_int = pd.DataFrame(poly_tranf.fit_transform(X_s_test),
                                   columns=poly_tranf.get_feature_names(input_features=
                                                                        X_s_train.columns),
                                   index=X_s_test.index)
    bands = [411, 443, 489, 510, 555, 670]
    model_dict=dict.fromkeys(bands)

    # create theano shared variable
    X_shared = shared(X_s_train_w_int.values)

    # Fitting aphi411 model:
    # Instantiate PyMC3 model with bnn likelihood
    for band in bands:
        logger.info("processing aphi{band}", band=band)
        X_shared.set_value(X_s_train_w_int.values)
        hshoe_wi_ = PyMCModel(hs_regression,
                            X_shared, y_train['log10_aphy%d' %band], n_hidden=4)
        hshoe_wi_.model.name = 'hshoe_wi_aphy%d' %band
        hshoe_wi_.fit(n_samples=2000, cores=4, chains=4, tune=10000,
                    nuts_kwargs=dict(target_accept=0.95))
        ppc_train_ = hshoe_wi_.predict(likelihood_name='likelihood')
        X_shared.set_value(X_s_test_w_int.values)
        ppc_test_ = hshoe_wi_.predict(likelihood_name='likelihood')
        run_dict = dict(model=hshoe_wi_.model, trace=hshoe_wi_.trace_,
                        ppc_train=ppc_train_, ppc_test=ppc_test_)
        model_dict[band]=run_dict
        with open('./pickleJar/Results_190118/hshoe_wi_model_dict.pkl', 'wb') as fb:
            pickle.dump(model_dict, fb, protocol=pickle.HIGHEST_PROTOCOL)
