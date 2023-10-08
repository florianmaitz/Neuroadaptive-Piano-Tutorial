import os.path as op
import pathlib
from pathlib import Path
from matplotlib import pyplot as plt

import mne
from scipy import signal
from mne.decoding import CSP
from mne_features.feature_extraction import FeatureExtractor
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score, KFold)
from sklearn.preprocessing import StandardScaler
import pywt


import numpy as np
import pyxdf
import math


class OldEEGSource:
    """
    Class implemented for the EEG preprocessing.
    Florian Maitz

    Giulia Pezzutti
    Assoc.Prof. Dr. Selina Christin Wriessnegger
    M.Sc. Luis Alberto Barradas Chacon

    Institute for Neural Engineering, @ TUGraz
    """

    def __init__(self, dict_info, path):
        self.channels_names, self.channels_types = {}, {}
        self.raw = None
        self.eeg_instants, self.eeg_signal = None, None
        self.marker_instants, self.marker_ids = None, None
        self.length = None
        self.eeg_fs = None
        self.file_info = {}
        self.events, self.event_mapping = None, None
        self.bad_channels, self.annotations = None, None
        self.epochs = None
        self.rois_numbers = {}
        self.freq_subbands = None
        self.feat_rel_power = None
        self.epoch_data = None

        self.input_info = dict_info
        self.xdf_path = path
        self.get_info_from_path()

        self.t_min = self.input_info['t_min']  # start of each epoch
        self.t_max = self.input_info['t_max']  # end of each epoch
        self.stream_names = dict_info['streams']

    def get_info_from_path(self):
        """
        Getting main information from file path regarding subject, folder and output folder according to
        LSLRecorder standard
        """

        # get name of the original file
        base = op.basename(self.xdf_path)
        file_name = op.splitext(base)[0]

        # main folder in which data is contained
        base = op.abspath(self.xdf_path)
        folder = op.dirname(base).split('data/')[0]
        folder = folder.replace('\\', '/')

        project_folder = str(pathlib.Path(__file__).parent.parent.absolute())

        # extraction of subject, session and run indexes
        if self.input_info['lsl-version'] == '1.12':
            subject = (file_name.split('subj_')[1]).split('_block')[0]
        elif self.input_info['lsl-version'] == '1.16':
            subject = (file_name.split('sub-')[1]).split('_ses')[0]
        else:
            subject = ''

        # output folder according to the standard
        output_folder = str(pathlib.Path(__file__).parent.parent.absolute()) + '/images/sub-' + subject
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.file_info = {'input_folder': folder, 'file_name': file_name, 'subject': subject,
                          'output_images_folder': output_folder, 'project_folder': project_folder}

    def load_data(self):
        dat = pyxdf.load_xdf(self.xdf_path)[0]

        for i in range(len(dat)):
            stream_name = dat[i]['info']['name'][0]
            orn_signal, orn_instants = [], []

            # gets 'BrainVision RDA Markers' stream
            if stream_name == self.stream_names['EEGMarkers']:
                orn_signal = dat[i]['time_series']
                orn_instants = dat[i]['time_stamps']

            # gets 'BrainVision RDA Data' stream
            if stream_name == self.stream_names['EEGData']:
                self.eeg_signal = dat[i]['time_series'][:, :32]  # eeg channels
                self.eeg_signal = self.eeg_signal * 1e-6
                self.eeg_instants = dat[i]['time_stamps']

                self.eeg_fs = int(float(dat[i]['info']['nominal_srate'][0]))
                effective_sfreq = float(dat[i]['info']['effective_srate'])

                self.load_channels(dat[i]['info']['desc'][0]['channels'][0]['channel'])

                # cast to arrays
                self.eeg_instants = np.array(self.eeg_instants)
                self.eeg_signal = np.asmatrix(self.eeg_signal)

            # gets LSL marker stream
            if stream_name == self.stream_names['Triggers']:
                self.marker_ids = dat[i]['time_series']
                self.marker_instants = dat[i]['time_stamps']

        # check lost-samples problem
        if len(orn_signal) != 0:
            print('\n\nATTENTION: some samples have been lost during the acquisition!!\n\n')
            self.fix_lost_samples(orn_signal, orn_instants, effective_sfreq)

        # get the length of the acquisition
        self.length = self.eeg_instants.shape[0]

        # remove samples at the beginning and at the end
        samples_to_be_removed = self.input_info['samples_remove']
        if samples_to_be_removed > 0:
            self.eeg_signal = self.eeg_signal[samples_to_be_removed:self.length - samples_to_be_removed]
            self.eeg_instants = self.eeg_instants[samples_to_be_removed:self.length - samples_to_be_removed]

        # reference all the markers instant to the eeg instants (since some samples at the beginning of the
        # recording have been removed)
        self.marker_instants -= self.eeg_instants[0]
        self.marker_instants = self.marker_instants[self.marker_instants >= 0]

        # remove signal mean
        self.eeg_signal = self.eeg_signal - np.mean(self.eeg_signal, axis=0)

        # create raw
        info = mne.create_info(list(self.channels_names.values()), self.eeg_fs, list(self.channels_types.values()))
        self.raw = mne.io.RawArray(self.eeg_signal.T, info)

        # set montage setting according to the input
        standard_montage = mne.channels.make_standard_montage(self.input_info['montage'])
        self.raw.set_montage(standard_montage)

        if len(self.bad_channels) > 0:
            self.raw.info['bads'] = self.bad_channels
            self.raw.interpolate_bads(reset_bads=True)

        rois = self.input_info['rois']

        # channel numbers associated to each roi
        for roi in rois.keys():
            self.rois_numbers[roi] = np.array([self.raw.ch_names.index(i) for i in rois[roi]])

        self.epoching()

    def data_preprocessing(self):
        """
        Preprocessing of the EEG data

        """
        print(f"Preprocessing started...")

        if self.input_info['spatial_filtering'] is not None:
            self.raw_spatial_filtering()

        if self.input_info['filtering'] is not None:
            self.raw_time_filtering()

    def feature_extraction(self, feat="Welch", classifier="LDA", split=4):
        """
        Feature extraction and classification of the preprocessed data.

        """
        epoch_data_ = self.epochs.get_data()
        self.epoch_data = epoch_data_[:, :, :-1]  # cut off the last time point to size it to 25.000 samples per condition
        labels = self.epochs.events[:, -1]

        labels = self.split_epoch(labels, split)

        self.subband_decomp()

        funcs_params = {}
        selected_funcs = []

        if self.input_info['subband_decomp'] == "Welch":
            # self.welch_feature_ex(epoch_data, min_freq=8, max_freq=12, step_size=1)
            selected_funcs = ['pow_freq_bands']
            # Each key of the funcs_params dict should be of the form: [alias_feature_function]__[optional_param]
            # (for example: higuchi_fd__kmax).
            funcs_params = {'pow_freq_bands__freq_bands': np.array([8, 9, 10, 11, 12])}     # [8, 9, 10, 11, 12]
        # elif feat is "wavelet":
            # self.wavelet_feature_ex(epoch_data)
            # selected_funcs = ['wavelet_coef_energy', 'mean', 'ptp_amp']
        else:
            print(f"{self.input_info['feat_extraction']} is not a valid feature extraction method. "
                  f"Standard used instead.")

        fe = FeatureExtractor(sfreq=self.eeg_fs, selected_funcs=selected_funcs, params=funcs_params)
        print(f"{fe.get_params()}")

        if self.input_info["classifier"] == "LDA":
            solvers = ["svd", "lsqr", "eigen"]
            pipe = Pipeline([('fe', fe),
                             ('scaler', StandardScaler()),
                             ('LDA', LinearDiscriminantAnalysis(solver=solvers[1], shrinkage="auto"))])
        elif self.input_info["classifier"] == "SVM":
            pipe = Pipeline([('fe', fe),
                             # ('scaler', StandardScaler()),
                             ('SVM', SVC(C=1.0, kernel="rbf", gamma='scale'))])
        else:
            print(f"{self.input_info['classifier']} is not a valid classifier model. Standard used instead.")

        # Splits for the KFold. If enough samples per condition, set to 10
        n_splits = min(int((self.epoch_data.shape[0]/len(self.epochs.event_id))), 10)

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y = labels

        x_train, x_test, y_train, y_test = train_test_split(self.epoch_data, y, test_size=0.2)
        accuracy = pipe.fit(x_train, y_train).score(x_test, y_test)
        y_pred = pipe.predict(x_test)
        scores = cross_val_score(pipe, self.epoch_data, y, cv=kf, scoring='accuracy')
        print('Accuracy score = %1.3f' % accuracy)
        print('cross_val_score: %1.3f' % np.mean(scores))
        print(f'cross_val_scores = {scores}')
        print(f'y_pred: {y_pred} y: {y_test}')

        return np.mean(scores)

    def subband_decomp(self, freq_bands=[8, 9, 10, 11, 12]):
        """

        :return:
        """
        subband_sig = []

        for idx in range(len(freq_bands)-1):
            # design chebyl type 2 filter
            sos = signal.cheby2(12, 20, [freq_bands[idx], freq_bands[idx+1]], 'bandpass', fs=self.eeg_fs, output='sos')
            subband_sig.append(signal.sosfilt(sos, self.epoch_data))

        return subband_sig

    def split_epoch(self, labels, split):
        """
        Split the epoch array into smaller time frames per condition, to achieve a higher sample size

        :param epoch_data: The epoch data (n_cond, n_channels, n_times)
        :param split: Number of times the array should be split (2, 4 or 8)
        :param labels: The conditions
        :return: The reshaped epoch data array (n_cond * split, n_channels, n_times / split) and labels array
        """

        if split not in [2, 4, 8]:
            split = 1

        split_array = np.array_split(self.epoch_data, split, axis=2)
        self.epoch_data = np.concatenate(split_array, axis=0)   # y[0] == y[12]

        # Reshape labels so they fit the new reshaped epoch data
        reshaped_labels = labels
        for idx in range(split-1):
            reshaped_labels = np.concatenate((reshaped_labels, labels))

        # a = reshaped_epoch_data[12]
        # b = epoch_data[0, :, 6250:12500]
        # c = a == b

        return reshaped_labels

    def epoching(self):
        """
        Divide the raw data into Epochs according to the LSL events

        """
        self.create_annotations()
        self.raw.set_annotations(self.annotations)

        self.events, self.event_mapping = mne.events_from_annotations(self.raw)

        # Automatic rejection criteria for the epochs
        reject_criteria = self.input_info['epochs_reject_criteria']

        # generation of the epochs according to the events
        # baseline correction excluded since there is no appropriate time intervall (baseline=(self.t_min, 0))
        self.epochs = mne.Epochs(self.raw, self.events, event_id=self.event_mapping, preload=True,
                                 baseline=None, reject=reject_criteria, tmin=self.t_min, tmax=self.t_max)

        event_times = [self.epochs.events[:, 0] / self.eeg_fs, self.epochs.events[:, 2]]

    def load_channels(self, dict_channels):
        """
        Upload channels name from a xdf file
        """

        # x = data[0][0]['info']['desc'][0]["channels"][0]['channel']
        # to obtain the default-dict list of the channels from the original file (data, not dat!!)

        # cycle over the info of the channels
        for idx, info in enumerate(dict_channels):

            if info['label'][0].find('dir') != -1 or info['label'][0] == 'MkIdx':
                continue

            # get channel name
            self.channels_names[idx] = info['label'][0]

            # solve problem with MNE and BrainProduct incompatibility
            if self.channels_names[idx] == 'FP2':
                self.channels_names[idx] = 'Fp2'

            # get channel type
            self.channels_types[idx] = 'eog' if info['label'][0].find('EOG') != -1 else 'eeg'

        a = self.file_info['subject']

        if self.file_info['subject'] in self.input_info['bad_channels'].keys():
            self.bad_channels = self.input_info['bad_channels'][self.file_info['subject']]
        else:
            self.bad_channels = []

    def fix_lost_samples(self, orn_signal, orn_instants, effective_sample_frequency):

        print('BrainVision RDA Markers: ', orn_signal)
        print('BrainVision RDA Markers instants: ', orn_instants)
        print('\nNominal srate: ', self.eeg_fs)
        print('Effective srate: ', effective_sample_frequency)

        print('Total number of samples: ', len(self.eeg_instants))
        final_count = len(self.eeg_signal)
        for lost in orn_signal:
            final_count += int(lost[0].split(': ')[1])
        print('Number of samples with lost samples integration: ', final_count)

        total_time = len(self.eeg_instants) / effective_sample_frequency
        real_number_samples = total_time * self.eeg_fs
        print('Number of samples with real sampling frequency: ', real_number_samples)

        # print(self.eeg_instants)

        differences = np.diff(self.eeg_instants)
        differences = (differences - (1 / self.eeg_fs)) * self.eeg_fs
        # differences = np.round(differences, 4)
        print('Unique differences in instants: ', np.unique(differences))
        print('Sum of diff ', np.sum(differences))
        # plt.plot(differences)
        # plt.ylim([1, 2])
        # plt.show()

        new_marker_signal = self.marker_instants

        for idx, lost_instant in enumerate(orn_instants):
            x = np.where(self.marker_instants < lost_instant)[0][-1]

            missing_samples = int(orn_signal[idx][0].split(': ')[1])
            additional_time = missing_samples / self.eeg_fs

            new_marker_signal[(x + 1):] = np.array(new_marker_signal[(x + 1):]) + additional_time

    def create_annotations(self, full=False):
        """
        Annotations creation according to MNE definition. Annotations are extracted from markers stream data (onset,
        duration and description)
        :param full: annotations can be made of just one word or more than one. In 'full' case the whole annotation is
        considered, otherwise only the second word is kept
        :return:
        """

        # generation of the events according to the definition
        triggers = {'onsets': [], 'duration': [], 'description': []}

        # read every trigger in the stream
        for idx, marker_data in enumerate(self.marker_ids):

            # according to 'full' parameter, extract the correct annotation description
            if not full:
                condition = marker_data[0].split('/')[0]
            else:
                condition = marker_data[0]

            # annotations to be rejected
            if condition in self.input_info['bad_epoch_names']:
                continue
            else:
                condition = condition[-1]

            # extract triggers information
            triggers['onsets'].append(self.marker_instants[idx])
            triggers['duration'].append(int(0))
            triggers['description'].append(condition)

        # define MNE annotations
        self.annotations = mne.Annotations(triggers['onsets'], triggers['duration'], triggers['description'])

    def visualize_raw(self, signal=True, psd=True):
        """
        Visualization of the plots that could be generated with MNE according to a scaling property
        :param signal: boolean, if the signal plot should be generated
        :param psd: boolean, if the psd plot should be generated
        """

        viz_scaling = dict(eeg=1e-4, eog=1e-4, ecg=1e-4, bio=1e-7, misc=1e-5)

        if signal:
            mne.viz.plot_raw(self.raw, scalings=viz_scaling, duration=50, show_first_samp=True)
        if psd:
            self.raw.plot(n_channels=len(self.channels_names))
            self.raw.plot_psd(n_channels=len(self.channels_names))
            plt.close('all')

    def raw_spatial_filtering(self):
        """
        Resetting the reference in raw data according to the spatial filtering type in the input dict
        """

        mne.set_eeg_reference(self.raw, ref_channels=self.input_info['spatial_filtering'], copy=False)

    def raw_time_filtering(self):
        """
        Filter of MNE raw instance data with a band-pass filter and a notch filter
        """

        # extract the frequencies for the filtering
        l_freq = self.input_info['filtering']['low']
        h_freq = self.input_info['filtering']['high']
        n_freq = self.input_info['filtering']['notch']

        # apply band-pass filter
        if not (l_freq is None and h_freq is None):
            self.raw.filter(l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=0.1, h_trans_bandwidth=0.1, verbose=40)

        # apply notch filter
        if n_freq is not None:
            self.raw.notch_filter(freqs=n_freq, verbose=40)

    def visualize_epochs(self, signal=True, conditional_epoch=True, rois=True):
        """
        :param signal: boolean, if visualize the whole signal with triggers or not
        :param conditional_epoch: boolean, if visualize the epochs extracted from the events or the general mean epoch
        :param rois: boolean (only if conditional_epoch=True), if visualize the epochs according to the rois or not
        """

        self.visualize_raw(signal=signal, psd=False)

        rois_names = list(self.rois_numbers.keys())

        # generate the mean plots according to the condition in the annotation value
        if conditional_epoch:

            # generate the epochs plots according to the roi and save them
            if rois:
                for condition in self.event_mapping.keys():
                    images = self.epochs[condition].plot_image(combine='mean', group_by=self.rois_numbers, show=False)
                    for idx, img in enumerate(images):
                        img.savefig(
                            self.file_info['output_images_folder'] + '/' + condition + '_' + rois_names[idx] + '.png')
                        plt.close(img)

            # generate the epochs plots for each channel and save them
            else:
                for condition in self.event_mapping.keys():
                    images = self.epochs[condition].plot_image(show=False, picks=list(self.channels_names.values()))
                    for idx, img in enumerate(images):
                        img.savefig(
                            self.file_info['output_images_folder'] + '/' + condition + '_' + self.channels_names[
                                idx] + '.png')
                        plt.close(img)

        # generate the mean plot considering all the epochs conditions
        else:

            # generate the epochs plots according to the roi and save them
            if rois:
                images = self.epochs.plot_image(combine='mean', group_by=self.rois_numbers, show=False)
                for idx, img in enumerate(images):
                    img.savefig(self.file_info['output_images_folder'] + '/' + rois_names[idx] + '.png')
                    plt.close(img)

            # generate the epochs plots for each channel and save them
            else:
                images = self.epochs.plot_image(show=False, picks=list(self.channels_names.values()))
                for idx, img in enumerate(images):
                    img.savefig(self.file_info['output_images_folder'] + '/' + self.channels_names[idx] + '.png')
                    plt.close(img)

        plt.close('all')

    def visualize_data(self, vis_raw=False, vis_epochs=True):
        """
        Visualize either the raw data or epochs.

        """
        if vis_raw:
            self.visualize_raw()

        if vis_epochs:
            self.visualize_epochs()

    def acc_rt(self, m_info={}):
        """
        Calculate mean subject accuracy and response time for each condition

        :param m_info: Dictionary with all recorded data from the program
        :return:
        """

        # only involve the wanted conditions
        cond_ids = [float(i) for i in list(self.epochs.event_id.keys())]

        idx_cond = np.zeros(shape=len(m_info['n']), dtype=bool)

        for cond in cond_ids:
            for idx in range(len(idx_cond)):
                # check if cond hasn't been already checked as well as if there is a target letter entry
                a = not math.isnan(23)
                if not idx_cond[idx] and not np.isnan(m_info['x'][idx]):
                    idx_cond[idx] = bool(m_info['n'][idx] == cond)

        sub_acc = np.array(m_info['key_resp_trial.corr'])[idx_cond]
        a = np.sum(sub_acc)#/(len(sub_acc)-)
        sub_rt = m_info['key_resp_trial.rt'][idx_cond]

        return sub_acc, sub_rt




    def wavelet_feature_ex(self, epoch_data):
        """
        Feature extraction using the Discrete Wavelet Transformation by decomposing the signal step by step
        to lower subbands.

        """

        for idx_cond in range(self.epochs.events.shape[0]):
            print(f"{idx_cond}. Trial extracting features...")
            for idx_channels, info in enumerate(self.channels_names.values()):

                coeffs_lvl6 = pywt.wavedec(epoch_data[idx_cond, idx_channels], 'db4', 'smooth',
                                           level=6)  # level 6: 500 / 2^6 ~= 8 -> cA6 = 0-8Hz / cD6 = 8-16Hz
                cD6 = coeffs_lvl6[1]  # 8-16Hz
                coeffs_lvl8 = pywt.wavedec(cD6, 'db4', 'smooth', level=2)  # level 8: cA8 = 8-10Hz, cD8 = 10-12Hz
                (cA_8_9, cD_9_10) = pywt.dwt(coeffs_lvl8[0], 'db4', 'smooth')  # level 9: 1Hz steps from 8-10Hz
                (cA_10_11, cD_11_12) = pywt.dwt(coeffs_lvl8[1], 'db4', 'smooth')  # level 9: 1Hz steps from 10-12Hz

                a = len(self.channels_names)
                b = len(cA_8_9)
                if (idx_cond == 0) and (idx_channels == 0):
                    self.freq_subbands = np.ndarray(
                        shape=(self.epochs.events.shape[0], len(self.channels_names), 4, len(cA_8_9)))

                self.freq_subbands[idx_cond, idx_channels] = [cA_8_9, cD_9_10, cA_10_11, cD_11_12]

    def welch_feature_ex(self, epoch_data, min_freq=8, max_freq=12, step_size=1):
    """
    Feature extraction using the Welch method (sliding time-window) and calculation of relative
    band power (BP) of the alpha band in 1Hz steps (8-12Hz). BP is aquired through parabolas using
    the composite Simpsons' rule.

    """
    sliding_win = (2/step_size) * self.eeg_fs  # choose sliding window size according to Nyqist theorem (in samples)

    sub_bands = int((max_freq - min_freq)/step_size)
    self.feat_rel_power = np.ndarray(shape=(self.epochs.events.shape[0], len(self.channels_names), sub_bands))
    print(f"Used frequency ranges: ")

    for idx_cond in range(self.epochs.events.shape[0]):
        print(f"{idx_cond}. Trial extracting features...")

        for idx_channels, info in enumerate(self.channels_names.values()):

            for idx_freq, freq in enumerate(list(range(min_freq, max_freq, step_size))):

                frequencies, psd = signal.welch(epoch_data[idx_cond, idx_channels], self.eeg_fs, nperseg=sliding_win)

                if (idx_cond == 0) and (idx_channels == 0):
                    print(f"[{freq}-{freq + step_size}]Hz")

                idx_sub = np.logical_and(frequencies >= freq, frequencies < (freq + step_size))

                freq_res = frequencies[1]-frequencies[0]
                sub_power = simps(psd[idx_sub], dx=freq_res)
                total_power = simps(psd, dx=freq_res)
                self.feat_rel_power[idx_cond, idx_channels, idx_freq] = sub_power / total_power

    def plot_func(self):

        fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 4))
        # ax[0].plot(chirp_signal)
        # ax[1].plot(cA_8_9, label='cA_8_9')
        # ax[2].plot(cD_9_10, label='cD_9_10')
        # ax[3].plot(cA_10_11, label='cA_10_11')
        # ax[4].plot(cD_11_12, label='cD_11_12')
        # ax[0].legend(loc='upper left')
        plt.show()


    # def timer_task(self, n_stage):
    #
    #     if n_stage == "rest":
    #         self.eeg_source.f_record_eeg = False
    #
    #         # Begin with resting EEG
    #         self.t_stamps.append((datetime.now(), "rest_eeg"))
    #
    #         timer = Timer(self.t_info["t_n_resting_eeg"], self.timer_task, args=("instruction", ))
    #         timer.start()
    #         print("resting EEG time!")
    #         return
    #
    #     self.eeg_source.f_record_eeg = False
    #     if self.current_n_trial < self.t_info["n_n_conditions"]:
    #         # Set the flag false, to stop recording EEG once the timer has been called
    #
    #         if n_stage == "instruction":
    #             # Do not record EEG here!
    #             print("instruction time!")
    #             self.t_stamps.append((datetime.now(), "instruction"))
    #
    #             timer = Timer(self.t_info["t_n_instruction"], self.timer_task, args=("n_back",))
    #             timer.start()
    #             return
    #
    #         elif n_stage == "n_back":
    #             # Record EEG here!
    #             print("n-back time!")
    #             self.t_stamps.append((datetime.now(), "n_back_start"))
    #
    #             timer = Timer(self.t_info["t_n_condition"], self.timer_task, args=("instruction",))
    #             timer.start()
    #
    #             self.current_n_trial += 1
    #
    #             if self.eeg_run:
    #                 self.eeg_source.record_eeg_n_back(self.current_n_trial)
    #
    #             return
    #     else:
    #         self.busy = False
    #         return

    def set_markers_n_back(self, t_rest, t_cond, ts_n_start, n_cond):
        print("set_markers")

        # get the recorded eeg until now
        eeg = self.eeg_signal_n_back

        self.marker_ids_n_back.append("resting eeg")
        self.marker_instants_n_back.append(ts_n_start)

        for idx, cond in enumerate(n_cond):
            self.marker_ids_n_back.append("n_" + str(cond))
            if idx == 0:
                self.marker_instants_n_back.append(ts_n_start + t_rest)
            else:
                self.marker_instants_n_back.append(self.marker_instants_n_back[-1] + t_cond)

        self.f_record_eeg = False
        print("work?")