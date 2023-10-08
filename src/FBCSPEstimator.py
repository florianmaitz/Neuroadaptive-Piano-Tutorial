import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from mne.decoding import CSP
from scipy import signal
import mne


class FBCSP(BaseEstimator, ClassifierMixin):

    def __init__(self, n_filt=4):

        self.n_filt = n_filt

    def fit(self, X, y):
        # Last dimension of X stands for different bands as specified

        # Check that X and y have correct shape
        self._check_Xy(X[:, :, :, 0], y)

        # Store the classes seen during fit
        # self.classes_ = unique_labels(y)
        self.CSP_models = []
        n_bands = X.shape[-1]

        for idx_band in range(n_bands):
            self.CSP_models.append(CSP(n_components=self.n_filt, reg=None, log=True, norm_trace=False))
            self.CSP_models[-1].fit(X[:, :, :, idx_band], y)

        return self

    def transform(self, X):
        # Input validation
        self._check_Xy(X[:, :, :, 0])

        n_bands = X.shape[-1]
        n_epochs = X.shape[0]
        X_ = np.empty(shape=(n_epochs, 0))

        # concatenate the features of every frequency band after transforming it to X_
        for idx_band in range(n_bands):
            csp_feat = self.CSP_models[idx_band].transform(X[:, :, :, idx_band])
            X_ = np.concatenate((X_, csp_feat), axis=1)

        return X_

    def _check_Xy(self, X, y=None):
        """Check input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError('X and y must have the same length.')
        if X.ndim < 3:
            raise ValueError('X must have at least 3 dimensions.')