import numpy as np
import mne
from scipy import signal
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, cross_val_score, KFold)
from sklearn.preprocessing import StandardScaler
from pylsl import local_clock

from src.FBCSPEstimator import FBCSP


class Classifier:

    def __init__(self, eeg_info, epochs, fs, adaptive):

        self.eeg_fs = fs

        self.f_band_vars = eeg_info["f_band_vars"]
        self.n_filt = eeg_info["n_csp_filt"]

        self.mean_acc = None
        self.trained_pipe = None

        self.piano_window_preds = []    # predictions for every window of the current level
        self.all_piano_level_preds = []     # predictions for all windows of all levels
        self.piano_levels = []         # level for each round (start with 1)
        self.mwl_mean = []
        self.adaptive = adaptive
        self.high_mwl_streak = 0

        if epochs:
            self.epochs = epochs
            self.f_band_epochs = self.create_filter_bands(self.epochs.get_data())
            self.labels = self.epochs.events[:, -1]

    def create_filter_bands(self, epochs):
        """
        Create the filter bands for the FBCSP according to f_band_vars.

        :param f_band_vars: [n_filter_bands + 1] Filter Band Values. Two adjacent values create one filter band.
        """

        t = local_clock()
        epoch_data = epochs[:, :, :-1]

        f_band_epochs = np.empty(shape=(epoch_data.shape[0], epoch_data.shape[1], epoch_data.shape[2],
                                             len(self.f_band_vars)-1))

        for idx in range(len(self.f_band_vars) - 1):
            # t_loop = local_clock()
            f_band_epochs[:, :, :, idx] = mne.filter.filter_data(data=epoch_data, sfreq=self.eeg_fs,
                                                l_freq=self.f_band_vars[idx], h_freq=self.f_band_vars[idx + 1],
                                                l_trans_bandwidth=0.1, h_trans_bandwidth=0.1, verbose=40,
                                                                      method='iir', iir_params=None, n_jobs=4)
            # print(local_clock()-t_loop)

        print(local_clock()-t)
        return f_band_epochs

    def cv_pipeline(self):

        t = local_clock()
        cv = KFold(10)

        pipe = Pipeline([('FBCSP', FBCSP(n_filt=self.n_filt,)),
                         ('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto"))])

        scores = cross_val_score(pipe, self.f_band_epochs, self.labels, cv=cv, scoring='accuracy', n_jobs=4)

        print(f"cv time: {local_clock()-t}")
        return np.mean(scores)

    def train_final_classifier(self):

        self.trained_pipe = Pipeline([('FBCSP', FBCSP(n_filt=self.n_filt,)),
                                      ('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto"))])
        self.trained_pipe.fit(self.f_band_epochs, self.labels)


    def predict_window(self, window):
        # predict the MWL of a window during a piano tutorial level
        # self.piano_window_preds.append(0)
        # return

        reshaped_window = window[np.newaxis, :, :]
        fb_window = self.create_filter_bands(reshaped_window)

        # 0 -> low MWL, 1 -> high MWL
        pred = self.trained_pipe.predict(fb_window)
        prob = self.trained_pipe.predict_proba(fb_window)
        print(f"Prediction appended: class: {pred} | prob: {prob}")

        self.piano_window_preds.append([int(pred[0]), list(prob[0])])

    def decide_next_level(self):
        """
        Decide if the next level is the same level repeated or the next level in the list
        :return:
        """
        if len(self.piano_window_preds) >= 1:
            self.mwl_mean.append(np.mean([preds[-1][1] for preds in self.piano_window_preds]))

            self.all_piano_level_preds.append(self.piano_window_preds.copy())
            self.piano_window_preds = []

            if self.adaptive:
                # mwl_mean < 0.5 means, that there was low MWL during the level -> next higher level
                if self.mwl_mean[-1] < 0.5:
                    self.piano_levels.append(self.piano_levels[-1] + 1)
                    self.high_mwl_streak = 0
                else:
                    # if the mental workload is high too many times in a row, still change to the next level
                    if self.high_mwl_streak >= 2:
                        self.piano_levels.append(self.piano_levels[-1] + 1)
                        self.high_mwl_streak = 0
                    else:
                        self.piano_levels.append(self.piano_levels[-1])
                        self.high_mwl_streak += 1
            else:
                self.piano_levels.append(self.piano_levels[-1] + 1)

            print(f"mwl_mean: {self.mwl_mean[-1]} | next level: {self.piano_levels[-1]}")
            return self.piano_levels[-1]
        else:
            # Start with the first Level
            self.piano_levels.append(1)
            return self.piano_levels[-1]

    def restart_piano_tut(self):
        self.piano_window_preds = []
        self.all_piano_level_preds = []
        self.piano_levels = []
        self.mwl_mean = []
