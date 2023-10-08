import numpy as np
import matplotlib
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('TKAgg')


class SaveData:

    def __init__(self, subj_nr, adaptive):
        self.subj_nr = subj_nr
        self.adaptive = adaptive
        if subj_nr < 10:
            self.subj = "sub-P00" + str(subj_nr)
        else:
            self.subj = "sub-P0" + str(subj_nr)

        self.subj_path = str(Path(__file__).parent.parent.absolute()) + '/data/' + self.subj
        Path(self.subj_path).mkdir(parents=True, exist_ok=True)
        self.data_path = self.subj_path + "/python_data"
        Path(self.data_path).mkdir(parents=True, exist_ok=True)

        self.eeg_signal_nback, self.eeg_ts_nback = None, None
        self.marker_ids_nback, self.marker_ts_nback = None, None

    def load_nback_raw(self, eeg_signal, eeg_ts, marker_ids, marker_ts):
        self.eeg_signal_nback = eeg_signal
        self.eeg_ts_nback = eeg_ts
        self.marker_ids_nback = marker_ids
        self.marker_ts_nback = marker_ts

    def save_files(self, stages, mean_acc=None, epochs=None, online_windows=None, trained_pipeline=None,
                   pred_classes_and_probs=None, level_progress=None, mwl_mean=None):

        training_params = {
            "subj": self.subj,
            "adaptive": self.adaptive,
            "mean_cv_acc": mean_acc
        }

        piano_params = None

        if stages['piano_tut']:
            piano_params = {
                "level_progress": level_progress,
                "mwl_mean": [round(mean, 4) for mean in mwl_mean],
                "pred_classes_and_probs": pred_classes_and_probs
            }

        self.save_json(training_params, piano_params)

        epoch_data = np.array(epochs.get_data())
        labels = epochs.events[:, -1]

        self.save_pickle(epoch_data, labels, trained_pipeline, online_windows)

        self.save_pickle_nback()

        print('Files correctly saved')

    def save_json(self, training_params, piano_params):
        json_object = json.dumps(training_params, indent=2)

        # Writing to sample.json
        with open(self.data_path + "/train_params.json", "w") as outfile:
            outfile.write(json_object)

        json_object = json.dumps(piano_params, indent=2)

        # Writing to sample.json
        with open(self.data_path + "/piano_params.json", "w") as outfile:
            outfile.write(json_object)

    def save_pickle(self, epoch_data, labels, trained_pipeline, online_windows):
        """
        Function to save epochs and labels of the n-back task, the trained pipeline and the time windows for the online
        classification during the piano tutorial.
        """

        with open(self.data_path + "/epoch_data.pkl", 'wb') as f:
            pickle.dump(epoch_data, f)

        with open(self.data_path + "/labels.pkl", 'wb') as f:
            pickle.dump(labels, f)

        with open(self.data_path + "/trained_pipeline.pkl", 'wb') as f:
            pickle.dump(trained_pipeline, f)

        with open(self.data_path + "/online_windows.pkl", 'wb') as f:
            pickle.dump(online_windows, f)

    def save_pickle_nback(self):
        path = self.data_path + '/nback_pickles'
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + "/eeg_signal.pkl", 'wb') as f:
            pickle.dump(self.eeg_signal_nback, f)

        with open(path + "/eeg_ts.pkl", 'wb') as f:
            pickle.dump(self.eeg_ts_nback, f)

        with open(path + "/marker_ids.pkl", 'wb') as f:
            pickle.dump(self.marker_ids_nback, f)

        with open(path + "/marker_ts.pkl", 'wb') as f:
            pickle.dump(self.marker_ts_nback, f)

    def load_pickle_nback(self):
        path = self.data_path + '/nback_sig_pickles'

        file = open(path + "/eeg_signal.pkl", 'rb')
        eeg_signal = pickle.load(file)
        file.close()

        file = open(path + "/eeg_ts.pkl", 'rb')
        eeg_ts = pickle.load(file)
        file.close()

        file = open(path + "/marker_ids.pkl", 'rb')
        marker_ids = pickle.load(file)
        file.close()

        file = open(path + "/marker_ts.pkl", 'rb')
        marker_ts = pickle.load(file)
        file.close()

        return eeg_signal, eeg_ts, marker_ids, marker_ts

    def save_pickle_for_testing(self, stages, mean_acc=None, epochs=None,
                   online_windows=None, trained_pipeline=None, pred_classes_and_probs=None, level_progress=None, mwl_mean=None):

        path = self.data_path + '/save_data_testing'
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path + "/stages.pkl", 'wb') as f:
            pickle.dump(stages, f)

        with open(path + "/mean_acc.pkl", 'wb') as f:
            pickle.dump(mean_acc, f)

        with open(path + "/epochs.pkl", 'wb') as f:
            pickle.dump(epochs, f)

        with open(path + "/online_windows.pkl", 'wb') as f:
            pickle.dump(online_windows, f)

        with open(path + "/trained_pipeline.pkl", 'wb') as f:
            pickle.dump(trained_pipeline, f)

        with open(path + "/pred_classes_and_probs.pkl", 'wb') as f:
            pickle.dump(pred_classes_and_probs, f)

        with open(path + "/level_progress.pkl", 'wb') as f:
            pickle.dump(level_progress, f)

        with open(path + "/mwl_mean.pkl", 'wb') as f:
            pickle.dump(mwl_mean, f)

    def load_pickle_for_testing(self):
        path = self.data_path + '/save_data_testing'

        file = open(path + "/stages.pkl", 'rb')
        stages = pickle.load(file)
        file.close()

        file = open(path + "/mean_acc.pkl", 'rb')
        mean_acc = pickle.load(file)
        file.close()

        file = open(path + "/epochs.pkl", 'rb')
        epochs = pickle.load(file)
        file.close()

        file = open(path + "/online_windows.pkl", 'rb')
        online_windows = pickle.load(file)
        file.close()

        file = open(path + "/trained_pipeline.pkl", 'rb')
        trained_pipeline = pickle.load(file)
        file.close()

        file = open(path + "/pred_classes_and_probs.pkl", 'rb')
        pred_classes_and_probs = pickle.load(file)
        file.close()

        file = open(path + "/level_progress.pkl", 'rb')
        level_progress = pickle.load(file)
        file.close()

        file = open(path + "/mwl_mean.pkl", 'rb')
        mwl_mean = pickle.load(file)
        file.close()

        return stages, mean_acc, epochs, online_windows, trained_pipeline, pred_classes_and_probs, level_progress, mwl_mean

    def load_pickles(self):
        file = open(self.data_path + "/online_windows.pkl", 'rb')
        online_windows = pickle.load(file)
        file.close()

        file = open(self.data_path + "/epoch_data.pkl", 'rb')
        epoch_data = pickle.load(file)
        file.close()

        file = open(self.data_path + "/trained_pipeline.pkl", 'rb')
        trained_pipeline = pickle.load(file)
        file.close()

        file = open(self.data_path + "/labels.pkl", 'rb')
        labels = pickle.load(file)
        file.close()

        return online_windows, epoch_data, trained_pipeline, labels

    def save_csp(self, fb_epochs, labels, epoch_info, fb_vars, n_feat):
        path = self.data_path + "/csp"
        Path(path).mkdir(parents=True, exist_ok=True)

        from mne.decoding.csp import CSP
        m = CSP(n_components=n_feat, reg=None, log=True, norm_trace=False)

        for idx_band in range(fb_epochs.shape[3]):
            m_fit = m.fit(fb_epochs[:, :, :, idx_band], labels)
            img = m_fit.plot_patterns(epoch_info, ch_type='eeg', units='Patterns (AU)', size=1.5, show=False)
            img.savefig(path + "/fb-" + str(fb_vars[idx_band:idx_band+2]) + ".png")
