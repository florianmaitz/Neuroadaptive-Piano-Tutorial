from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet, local_clock


class LSLSource:

    def __init__(self):
        self.eeg_ts_rec, self.eeg_signal_rec = [], []
        self.marker_ids, self.marker_ts = [], []
        self.eeg_inlet = None
        self.marker_outlet = None
        self.eeg_fs = 500
        self.start_idx_level = None
        self.start_idx_window = None

        # flags:
        self.f_eeg_recording = False

    def set_lsl_streams(self):
        # Receive the EEG stream
        stream_name = "g.USBamp-1"  # "BrainVision RDA"
        print(f"Searching for {stream_name} stream...")
        eeg_stream = resolve_stream("name", stream_name)  # self.stream_names['EEGData']
        self.eeg_inlet = StreamInlet(eeg_stream[0])
        print(f"{stream_name} stream is connected!")

        # info = self.eeg_inlet.info()
        # ch = info.desc().child("channels").child("channel")
        #
        # # also retrieve info from stream here!!! test with actual eeg stream
        # self.eeg_fs = info.nominal_srate()
        # self.load_channels(ch, info.channel_count())

        # Provide Marker stream
        info = StreamInfo("Marker_Stream", "Markers", 1, 0, "string", "markerstream_pianotut")
        self.marker_outlet = StreamOutlet(info)

    def record_eeg(self):
        print("EEG recording started...")
        self.f_eeg_recording = True

        while self.f_eeg_recording:
            chunk, time_stamp = self.eeg_inlet.pull_chunk()

            # only if actual data is received, write it on list
            if time_stamp:
                self.eeg_signal_rec += chunk
                self.eeg_ts_rec += time_stamp

        print("EEG recording stopped...")

    def get_class_windows(self, window_size=3, t_level=60):

        # at the beginning of each level, it is none
        if self.start_idx_window is None:
            self.start_idx_level = len(self.eeg_ts_rec) - 1
            self.start_idx_window = self.start_idx_level

        # if enough samples according to window size are recorded, send them back to
        if len(self.eeg_ts_rec) - self.start_idx_window > window_size * self.eeg_fs:
            end_idx_window = self.start_idx_window + (window_size*self.eeg_fs)

            # Return the window for classification
            window = self.eeg_signal_rec[self.start_idx_window:end_idx_window]

            # end idx is the new starting index of the new window
            self.start_idx_window = end_idx_window

            # if more than 60 seconds of time has been sent back, return False and signal end of level
            if self.start_idx_window + (window_size*self.eeg_fs) > self.start_idx_level + (t_level * self.eeg_fs):
                return window, False

            # True stands for another window will come
            return window, True

        return None, None

    def update_marker_stream(self, t_rest, t_cond, ts_n_start, n_cond):
        """
        Uses timings relative to the start of the resting EEG of the n-back-task, to set the LSL markers for the
        marker stream.

        :param t_rest:
        :param t_cond:
        :param ts_n_start:
        :param n_cond:
        :return:
        """
        running = True
        t_cumulative = t_rest
        idx = 0

        # Write the Resting EEG on the local marker variable
        self.marker_ids.append("Resting EEG")
        self.marker_ts.append(ts_n_start)

        while running:
            ts = local_clock()
            if ts - ts_n_start >= t_cumulative:
                sample = "n_" + str(n_cond[idx])
                self.marker_outlet.push_sample([sample])

                # Write it also on the local variable
                self.marker_ids.append(sample)
                self.marker_ts.append(ts)

                t_cumulative += t_cond
                idx += 1

                if idx >= len(n_cond):
                    break

    def get_eeg(self):
        eeg, ts = self.eeg_signal_rec, self.eeg_ts_rec
        return eeg, ts

    def get_marker_n(self):
        return self.marker_ids, self.marker_ts
