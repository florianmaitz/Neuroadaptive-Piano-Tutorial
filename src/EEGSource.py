import mne
import numpy as np
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet, local_clock

from autoreject import get_rejection_threshold
import matplotlib
matplotlib.use('TKAgg')

CHANNELS = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7',
            'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
            'F8', 'Fp2']


class EEGSource:
    """

    """

    def __init__(self, dict_info, t_info, save_data_instance, load_nback_data, com_instance, overlap, overlap_step_size):
        self.load_nback_data = load_nback_data
        self.channels_names, self.channels_types = {}, {}
        self.raw = None
        self.eeg_ts, self.eeg_signal = [], []
        self.marker_ts, self.marker_ids = [], []
        self.save_data_instance = save_data_instance
        self.com_instance = com_instance

        self.stream_name = "BrainVision RDA"    # "g.USBamp-1"  # "BrainVision RDA"

        self.epochs_nback = None

        self.length = None
        self.eeg_fs = 500
        self.file_info = {}
        self.events, self.event_mapping = None, None
        self.bad_channels, self.annotations = None, None

        self.input_info = dict_info

        self.t_n_window = t_info["t_n_window"]
        self.t_rest = t_info["t_n_resting_eeg"]
        self.t_instr = t_info["t_n_instruction"]
        self.t_n = t_info["n_n_letters"] * (t_info["t_n_cross"] + t_info["t_n_letter"] + t_info["t_n_blank"])

        # Overlapping windows
        self.overlap = overlap
        self.overlap_step_size = overlap_step_size

        self.ica = False

        # --------- LSL Source ---------
        self.eeg_ts_rec, self.eeg_signal_rec = [], []
        self.marker_ids, self.marker_ts = [], []
        self.eeg_inlet = None
        self.marker_outlet = None

        # flags:
        self.f_eeg_recording = False

        # --------- Piano ---------
        self.start_idx_level = None
        self.start_idx_window = None

        self.online_windows = []
        self.all_online_windows = []

    def set_lsl_streams(self):
        # Receive the EEG stream
        print(f"Searching for {self.stream_name} stream...")

        eeg_stream = resolve_stream("name", self.stream_name)  # self.stream_names['EEGData']
        self.eeg_inlet = StreamInlet(eeg_stream[0])
        print(f"{self.stream_name} stream is connected!")

        if self.stream_name == "BrainVision RDA":
            info = self.eeg_inlet.info()
            ch = info.desc().child("channels").child("channel")

            # also retrieve info from stream here!!! test with actual eeg stream
            self.eeg_fs = info.nominal_srate()
            self.load_channels(ch, info.channel_count())

        # Provide Marker stream
        info = StreamInfo("Marker_Stream", "Markers", 1, 0, "string", "markerstream_pianotut")
        self.marker_outlet = StreamOutlet(info)

    def load_channels(self, xml_channels, n_channels):
        """
        Get the channel names from xml

        """

        ch = xml_channels

        for idx in range(n_channels):

            label = ch.child_value("label")

            # skip directional and not needed channels
            if label.find('dir') != -1 or label == 'MkIdx':
                continue

            # get channel name
            self.channels_names[idx] = label

            # solve problem with MNE and BrainProduct incompatibility
            if self.channels_names[idx] == 'FP2':
                self.channels_names[idx] = 'Fp2'

            # get channel type
            self.channels_types[idx] = 'eog' if label.find('EOG') != -1 else 'eeg'

            ch = ch.next_sibling()

    def record_eeg(self):
        self.f_eeg_recording = True

        while self.f_eeg_recording:
            chunk, time_stamp = self.eeg_inlet.pull_chunk()

            # only if actual data is received, write it on list
            if time_stamp:
                self.eeg_signal_rec += chunk
                self.eeg_ts_rec += time_stamp

        print("EEG recording stopped...")

    def update_marker_stream(self, ts_n_start, n_cond):
        """
        Uses timings relative to the start of the resting EEG of the n-back-task, to set the LSL markers for the
        marker stream.

        :param t_rest:
        :param t_cond:
        :param ts_n_start:
        :param n_cond:
        :return:
        """
        t_cumulative = self.t_rest + self.t_instr
        idx = 0

        print("Resting EEG")
        self.marker_outlet.push_sample(["resting_EEG"], ts_n_start)

        while 1:
            ts = local_clock()
            if ts - ts_n_start >= t_cumulative:
                if idx >= len(n_cond):
                    break

                self.com_instance.send_quest(self.com_instance.quest_commands["toggle_pause_screen"])

                sample = "n_" + str(n_cond[idx])
                self.marker_outlet.push_sample([sample], ts)
                print("Pause ended. Next: " + sample)

                # Write it also on the local variable
                self.marker_ids.append(sample)
                self.marker_ts.append(ts)

                t_cumulative += self.t_n + self.t_instr
                idx += 1

    def get_n_back_epochs_preprocessed(self):
        if not self.load_nback_data:
            # Copy signal and marker onto local variable
            eeg_signal = self.eeg_signal_rec.copy()
            eeg_ts = self.eeg_ts_rec.copy()

            marker_ids = self.marker_ids.copy()
            marker_ts = self.marker_ts.copy()

            self.save_data_instance.load_nback_raw(eeg_signal, eeg_ts, marker_ids, marker_ts)
            self.save_data_instance.save_pickle_nback()
        else:
            eeg_signal, eeg_ts, marker_ids, marker_ts = self.save_data_instance.load_pickle_nback()
            self.save_data_instance.load_nback_raw(eeg_signal, eeg_ts, marker_ids, marker_ts)

        self.data_preprocessing(eeg_signal, eeg_ts, marker_ids, marker_ts)

        return self.epochs_nback, self.eeg_fs

    def data_preprocessing(self, eeg_signal, eeg_ts=None, marker_ids=None, marker_ts=None):

        # cast to arrays
        if eeg_ts:
            eeg_ts = np.array(eeg_ts)
        eeg_signal = np.asmatrix(eeg_signal)

        # use only the eeg channels and transform them to uV
        eeg_signal = eeg_signal[:, :32]  # eeg channels
        eeg_signal = eeg_signal * 1e-6

        # reference all the markers instant to the eeg instants (since some samples at the beginning of the
        # recording have been removed)
        if marker_ts:
            marker_ts -= eeg_ts[0]
            marker_ts = marker_ts[marker_ts >= 0]

        if self.stream_name == "BrainVision RDA":
            info = mne.create_info(list(self.channels_names.values()), self.eeg_fs, list(self.channels_types.values()))
        else:
            info = mne.create_info(CHANNELS, self.eeg_fs, 'eeg')

        raw = mne.io.RawArray(eeg_signal.T, info)

        # set montage setting according to the input
        standard_montage = mne.channels.make_standard_montage(self.input_info['montage'])
        raw.set_montage(standard_montage)

        raw = self.data_filtering(raw)

        if marker_ids:
            self.epoching(raw, marker_ids, marker_ts)
            return

        return raw.get_data()

    def data_filtering(self, raw):
        """
        Preprocessing of the EEG data

        """
        print(f"Filtering started...")

        t = local_clock()

        if self.input_info['filtering'] is not None:
            # extract the frequencies for the filtering
            l_freq = self.input_info['filtering']['low']
            h_freq = self.input_info['filtering']['high']
            n_freq = self.input_info['filtering']['notch']

            # apply band-pass filter
            if not (l_freq is None and h_freq is None):
                raw.filter(method='iir', iir_params=None, l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=0.1, h_trans_bandwidth=0.1, verbose=40, n_jobs=4)

            # apply notch filter
            if n_freq is not None:
                raw.notch_filter(method='iir', iir_params=None, freqs=n_freq, verbose=40, n_jobs=4)

        print(local_clock()-t)
        return raw

    def epoching(self, raw, marker_ids, marker_ts):
        """
        Divide the raw data into Epochs according to the LSL events

        """
        annotations = self.create_annotations(marker_ids, marker_ts)
        raw.set_annotations(annotations)

        events, event_mapping = mne.events_from_annotations(raw)

        # Automatic rejection criteria for the epochs
        reject_criteria = self.input_info['epochs_reject_criteria']

        if not self.ica:
            # generation of the epochs according to the events
            self.epochs_nback = mne.Epochs(raw, events, event_id=event_mapping, preload=True,
                                           baseline=None, reject=reject_criteria, tmin=0, tmax=self.t_n_window)

        else:
            # dont reject yet and correct for ica
            self.epochs_raw = mne.Epochs(raw, events, event_id=event_mapping, preload=True,
                                           baseline=None, reject=None, tmin=0, tmax=self.t_n_window)

            # reject_ica = get_rejection_threshold(self.epochs_raw)

            ica = mne.preprocessing.ICA(n_components=0.99, random_state=42)
            ica.fit(self.epochs_raw.copy(), tstep=self.t_n_window)

            #ica.plot_components()
            # ica.plot_properties(self.epochs_nback, picks=range(0, ica.n_components_))

            ica_z_thresh = 1.96
            eog_indices, eog_scores = ica.find_bads_eog(self.epochs_raw.copy(), ch_name=['Fp1', 'F8'])#, threshold=ica_z_thresh)
            ica.exclude = eog_indices

            # ica.plot_scores(eog_scores)

            self.epochs_nback = ica.apply(self.epochs_raw.copy())

            # self.epochs_raw.plot(scalings=dict(eeg=2e-4), n_epochs=2, picks=['Fp1'])
            # self.epochs_nback.plot(scalings=dict(eeg=2e-4), n_epochs=2, picks=['Fp1'])

            print(1)

    def create_annotations(self, marker_ids, marker_ts):
        """
        Annotations creation according to MNE definition. Annotations are extracted from markers stream data (onset,
        duration and description). One event gets split into multiple events according to window size to create more
        trials.

        :return:
        """

        # generation of the events according to the definition
        triggers = {'onsets': [], 'duration': [], 'description': []}

        # read every trigger in the stream
        for idx, marker_data in enumerate(marker_ids):

            if marker_data in self.input_info['nback_epochs']:
                condition = marker_data[-1]
            else:
                continue

            if not self.overlap:

                n_windows = int(self.t_n/self.t_n_window)

                for idx_w in range(n_windows):
                    # extract triggers information
                    triggers['onsets'].append(marker_ts[idx] + (idx_w * self.t_n_window))    # excluded self.t_intro
                    triggers['duration'].append(int(0))
                    triggers['description'].append(condition)

            else:
                n_windows = int((self.t_n-self.t_n_window)/self.overlap_step_size) + 1

                for idx_w in range(n_windows):
                    # extract triggers information
                    triggers['onsets'].append(marker_ts[idx] + (idx_w * self.overlap_step_size))    # excluded self.t_intro
                    triggers['duration'].append(int(0))
                    triggers['description'].append(condition)

        # define MNE annotations
        annotations = mne.Annotations(triggers['onsets'], triggers['duration'], triggers['description'])
        return annotations

    def get_piano_window_preprocessed(self, window_size, t_level):
        online_window, var = self.get_online_windows(window_size=window_size, t_level=t_level)

        if online_window is None:
            return None, None

        # online_window = self.data_preprocessing(online_window)

        online_window = np.transpose(np.asmatrix(online_window.copy())[:, :32]) * 1e-6
        # online_window = self.filter_test(online_window)
        self.online_windows.append(online_window)

        if var == 0:
            self.all_online_windows.append(self.online_windows.copy())
            self.online_windows = []

        return online_window, var

    def get_online_windows(self, window_size, t_level):

        # at the beginning of each level, it is none
        if self.start_idx_window is None:
            self.start_idx_level = len(self.eeg_ts_rec) - 1
            self.start_idx_window = self.start_idx_level

        # if enough samples according to window size are recorded, send them back to
        if len(self.eeg_ts_rec) - self.start_idx_window > window_size * self.eeg_fs:

            end_idx_window = int(self.start_idx_window + (window_size * self.eeg_fs))

            # Return the window for classification
            window = self.eeg_signal_rec[self.start_idx_window:end_idx_window].copy()

            # end idx is the new starting index of the new window
            self.start_idx_window = end_idx_window

            # if more than 60 seconds of time has been sent back, return False and signal end of level
            if self.start_idx_window + (window_size * self.eeg_fs) > self.start_idx_level + (t_level * self.eeg_fs):
                self.start_idx_level = None
                self.start_idx_window = None
                return window, 0

            # True stands for another window will come
            return window, 1

        return None, None

    # def load_online_windows(self):
    #     online_windows, epoch_data = self.save_data_instance.load_online_windows()
    #
    #     eeg_signal = np.transpose(online_windows[0][0] / 1e-6)
    #
    #     # cast to arrays
    #     eeg_signal = np.asmatrix(eeg_signal)
    #
    #     # use only the eeg channels and transform them to uV
    #     eeg_signal = eeg_signal[:, :32]  # eeg channels
    #     eeg_signal = eeg_signal * 1e-6
    #
    #     info = mne.create_info(CHANNELS, self.eeg_fs, 'eeg')
    #
    #     raw = mne.io.RawArray(eeg_signal.T, info)
    #
    #     # set montage setting according to the input
    #     standard_montage = mne.channels.make_standard_montage(self.input_info['montage'])
    #     raw.set_montage(standard_montage)
    #
    #     raw = self.data_filtering(raw)
    #
    #     raw_dat = raw.get_data()
    #
    #     print(raw_dat)
