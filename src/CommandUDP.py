import numpy
import numpy as np
import mido

from src.EEGSource import EEGSource
from src.Classifier import Classifier
from src.SaveData import SaveData

import socket
import time
from pylsl import local_clock
from threading import Timer
from threading import Thread


class CommandHandler:

    def __init__(self, subj_nr, adaptive, ip, port, com_info, t_info, eeg_info, nback_data):
        # set this to True, if quest is connected
        self.udp_run = True

        # set this to True, if eeg data stream is available
        self.eeg_run = True

        # set this to True, if nback data should be loaded and the nback task should be skipped
        self.load_nback_data = False

        # set this to True, if you want to reload a measurement completely
        self.reload_measurement = False
        self.retrain_classifier = False

        # set this to True, if you want overlapping windows and specify the overlap step size
        self.overlap = True
        self.overlap_step_size = 2.1

        # set this to True, if CSP should be recorded
        self.save_csp = True

        self.save_data_instance = SaveData(subj_nr, adaptive)

        self.subj_nr = subj_nr
        self.socket = None
        self.ip = ip
        self.port = port
        self.quest_commands = com_info
        self.t_info = t_info
        self.eeg_info = eeg_info
        self.next_stage = None
        self.busy = False
        self.thread_eeg_rec = None
        self.trained_classifier = None
        self.lsl_process = None
        self.n_filt = eeg_info["n_csp_filt"]
        self.cv_acc = None
        self.delay = 0
        self.adaptive = adaptive

        # Conditions of the n-back task
        self.n_cond = [trial['n'] for trial in nback_data['levels']]

        # Write timings into variables from file
        self.t_rest = t_info["t_n_resting_eeg"]
        self.t_instr = t_info["t_n_instruction"]
        self.t_n = t_info["n_n_letters"] * (t_info["t_n_cross"] + t_info["t_n_letter"] + t_info["t_n_blank"])
        self.t_cd = t_info["t_cd"]
        self.t_level = t_info["t_piano_level"]

        self.n_lvl = t_info["n_piano_levels"]
        self.t_pause_screen = t_info["t_pause_screen"]
        self.play_active = False
        self.t_online_window = t_info['t_online_window']

        self.stages = {
            "start_lsl": 0,
            "record_eeg": 0,
            "toggle_and_swap": 0,
            "n_back_test": 0,
            "n_back": 0,
            "swap_scene": 0,
            "lcon_parenting": 0,
            "piano_test": 0,
            "piano_pre": 0,
            "piano_tut": 0,
            "piano_post": 0,
            "save_data": 0
        }

        if self.load_nback_data:
            self.stages['piano_test'] = 0
            self.stages['piano_pre'] = 0
            self.stages['piano_post'] = 0
            self.stages['n_back_test'] = 0

        self.console_commands = ["y", "status", "start_lsl", "record_eeg", "n_back_test", "n_back", "piano_test",
                                 "piano_pre", "piano_tut", "piano_post", "save_data", "lcon_parenting",
                                 "toggle_and_swap", "swap_scene", "skip"]
        self.next_lvl = 1

        if self.udp_run:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Start midi thread to continuously send the piano keys to the quest
            midi_thread = Thread(target=self.send_midi,)
            midi_thread.start()
            self.send_keys = True

        if self.eeg_run:
            self.eeg_source = EEGSource(eeg_info, t_info, self.save_data_instance, self.load_nback_data, self,
                                        self.overlap, self.overlap_step_size)

    def calc_delay(self):
        if self.udp_run:
            t = local_clock()
            t_response = []

            self.socket.sendto(str.encode(str(0)), (self.ip, self.port))

            while 1:
                bytesToSend = str.encode(str(60))
                sendtime = local_clock()
                self.socket.sendto(bytesToSend, (self.ip, self.port))
                response = self.socket.recv(1024)
                t_response.append(local_clock() - sendtime)
                print(t_response[-1])

                if local_clock() - t >= 5:
                    break

            # Calculate the average delay of the quest for receiving UDP data
            self.socket.sendto(str.encode(str(0)), (self.ip, self.port))
            self.delay = numpy.mean(t_response)/2
            print(f"Average UDP delay: {self.delay}")

    def send_midi(self):
        if self.udp_run:
            with mido.open_input() as midi_port:
                for message in midi_port:
                    if message.type == 'note_on' and message.velocity != 0:
                        if self.send_keys:
                            byte_notes = str.encode(str(message.note - 12))
                            self.socket.sendto(byte_notes, (self.ip, self.port))

    def send_quest(self, cmd):
        if self.udp_run:
            try:
                num = int(cmd)
            except ValueError:
                print("Error. Command not in Quest command list!")
                return

            if cmd == self.quest_commands["force_end"]:
                self.send_keys = False
                time.sleep(0.2)
            elif 100 <= cmd <= 111 or cmd >= 200:
                self.send_keys = True

            bytes_to_send = str.encode(str(cmd))
            self.socket.sendto(bytes_to_send, (self.ip, self.port))

            # wait for delay to continue with program
            time.sleep(self.delay)

    def read_console(self, mes):
        """
        Checks if console command is in self.console_commands and then forwards it to the command handler

        :param mes: message from the console
        :return:
        """
        if mes in self.console_commands:
            self.stage_handler(mes)
        else:
            print(f"{mes} command does not exist!")
            return

    def stage_handler(self, com):
        """
        Handle the command from the console and execute it as well as send it to the quest
        """

        # to reload a previous measurement
        if self.reload_measurement:
            self.reload_measurement_function()
            return

        if com == "y":
            # start next part of program as in get_next_command
            com = self.next_stage

        if com == "skip":
            self.stages[self.next_stage] = 1
            return

        elif com == "status":
            # show the progress during the experiment
            print(self.stages)
            return

        elif com == "start_lsl":
            # start receiving the EEG stream and start the marker stream
            self.eeg_source.set_lsl_streams()
            time.sleep(0.5)

        elif com == "record_eeg":
            # Before the n_back starts, start already recording EEG

            if not self.stages["start_lsl"]:
                return

            self.thread_eeg_rec = Thread(target=self.eeg_source.record_eeg)
            self.thread_eeg_rec.start()

            time.sleep(0.5)

            if self.eeg_source.eeg_signal_rec:
                print("EEG recording started...")
            else:
                print("No data from EEG stream retrieved..")
                self.eeg_source.f_eeg_recording = False
                return

        elif com == "toggle_and_swap":
            self.send_quest(self.quest_commands["toggle_UI"])
            self.send_quest(self.quest_commands["swap_scene"])

        elif com == "n_back_test":
            # let the participant get used to n-back and VR

            if not self.stages["start_lsl"]:
                return

            self.send_quest(self.quest_commands["load_nback_file"][0])
            if not self.play_active:
                self.send_quest(self.quest_commands["toggle_play"])
                self.play_active = True
            self.eeg_source.marker_outlet.push_sample([com])

            t1 = Timer(self.t_instr, self.trigger_pause_screen,)
            t1.start()

            t2 = Timer(2*self.t_instr + self.t_n, self.trigger_pause_screen,)
            t2.start()

            time.sleep(2*self.t_instr + 2*self.t_n)

        elif com == "n_back":
            # start the n_back paradigm and use the known timings to record the EEG

            if not self.load_nback_data:
                if not self.stages["record_eeg"]:
                    return

                self.send_quest(self.quest_commands["load_nback_file"][self.subj_nr])
                if not self.play_active:
                    self.send_quest(self.quest_commands["toggle_play"])
                    self.play_active = True

                # Safe timestamp and push into outlet
                ts_n_start = local_clock()

                # Start the Marker stream which sends the n-back condition at the beginning of each introduction
                self.eeg_source.update_marker_stream(ts_n_start, self.n_cond)

            # Once the recording is done, start the training
            self.start_training()

        elif com == "swap_scene":
            self.send_keys = True

            self.send_quest(self.quest_commands["swap_scene"])
            time.sleep(0.5)

            if self.play_active:
                self.send_quest(self.quest_commands["toggle_play"])
                self.play_active = False

        elif com == "lcon_parenting":
            self.send_keys = True

            if self.play_active:
                self.send_quest(self.quest_commands["toggle_play"])
                self.play_active = False

            self.send_quest(self.quest_commands["toggle_UI"])
            self.send_quest(self.quest_commands["toggle_lcon_parent"])

            input("Press enter once ready to parent!")

            self.send_quest(self.quest_commands["toggle_lcon_parent"])
            self.send_quest(self.quest_commands["toggle_UI"])

        elif com == "piano_test":
            # let the participant get used to the piano and VR
            print("start_piano_test")

            self.send_quest(self.quest_commands["set_song"][11])

            if not self.play_active:
                self.send_quest(self.quest_commands["toggle_play"])
                self.play_active = True

            self.eeg_source.marker_outlet.push_sample([com])

            time.sleep(self.t_level + self.t_cd)

            self.send_quest(self.quest_commands["force_end"])

        elif com == "piano_pre":
            # start the comparison song before the piano tutorial
            print("starting the comparison song")

            self.send_quest(self.quest_commands["set_song"][0])

            if not self.play_active:
                self.send_quest(self.quest_commands["toggle_play"])
                self.play_active = True

            self.eeg_source.marker_outlet.push_sample([com])

            time.sleep(self.t_level + self.t_cd)

            self.send_quest(self.quest_commands["force_end"])

        elif com == "piano_tut":
            # start the adaptive/non-adaptive piano tutorial and send to quest, which level the classifier chose
            # add more variables to be fullfilled before starting
            if self.trained_classifier is None:
                print("No classifier trained yet!")
                return

            if self.stages[com]:
                msg = input(f'Restart piano_tut? (y/n): ')
                if msg == 'y':
                    self.trained_classifier.restart_piano_tut()
                else:
                    return

            self.busy = True

            if not self.play_active:
                self.send_quest(self.quest_commands["toggle_play"])
                self.play_active = True

            self.eeg_source.marker_outlet.push_sample([com])

            self.start_classification()

            self.busy = False

        elif com == "piano_post":
            # start the comparison song after the piano tutorial
            print("starting the comparison song")

            self.send_quest(self.quest_commands["set_song"][0])

            if not self.play_active:
                self.send_quest(self.quest_commands["toggle_play"])
                self.play_active = True

            self.eeg_source.marker_outlet.push_sample([com])

            time.sleep(self.t_level + self.t_cd)

            self.send_quest(self.quest_commands["force_end"])

        elif com == "save_data":
            print("Save the data")

            # stages, mean_acc, epochs, online_windows, trained_pipeline, pred_classes_and_probs, level_progress, mwl_mean = self.save_data_instance.load_pickle_for_testing()
            #
            # self.save_data_instance.save_files(stages, mean_acc, epochs, online_windows, trained_pipeline, pred_classes_and_probs, level_progress, mwl_mean)

            if self.stages["n_back"]:
                # self.save_data_instance.save_pickle_for_testing(self.stages, self.cv_acc, self.trained_classifier.epochs,
                #                                    self.eeg_source.all_online_windows, self.trained_classifier.trained_pipe,
                #                                    self.trained_classifier.all_piano_level_preds, self.trained_classifier.piano_levels,
                #                                    self.trained_classifier.mwl_mean)

                self.save_data_instance.save_files(self.stages, self.cv_acc, self.trained_classifier.epochs,
                                                   self.eeg_source.all_online_windows, self.trained_classifier.trained_pipe,
                                                   self.trained_classifier.all_piano_level_preds, self.trained_classifier.piano_levels,
                                                   self.trained_classifier.mwl_mean)


        # Save stage progress in variable
        self.stages[com] = 1

    def get_next_stage(self):
        for (key, value) in self.stages.items():
            if value == 0:
                self.next_stage = key
                return key

    def start_training(self):
        # Do preprocessing and get the epochs
        epochs, fs = self.eeg_source.get_n_back_epochs_preprocessed()

        self.trained_classifier = Classifier(self.eeg_info, epochs, fs, self.adaptive)

        self.cv_acc = self.trained_classifier.cv_pipeline()
        self.trained_classifier.train_final_classifier()

        n_0_events = 0
        for evs in epochs.events:
            if evs[2] == 1:
                n_0_events += 1

        print(f"Total events: {len(epochs.events)}")
        print(f"n_0 events: {n_0_events}")
        print(f"n_2 events: {len(epochs.events) - n_0_events}")

        print(f"Cv accuracy for trained classifier: {self.cv_acc}")

        if self.save_csp:
            self.save_data_instance.save_csp(self.trained_classifier.f_band_epochs, self.trained_classifier.labels,
                                             epochs.info, self.trained_classifier.f_band_vars, self.n_filt)

    def start_classification(self):
        print("Start Piano Tutorial")

        # Go through the number of levels to be played
        for idx_level in range(self.n_lvl):
            # Get the new level according to the classifier
            self.next_lvl = self.trained_classifier.decide_next_level()

            print(f"Start level {self.next_lvl}")
            # Start the countdown of the upcoming level
            self.send_quest(self.quest_commands["set_song"][self.next_lvl])

            # Wait for the Cd
            time.sleep(self.t_cd)

            # Start a timer to end each level accordingly
            timer_end_song = Timer(self.t_level + 0.3, self.end_of_level, args=(idx_level,))
            timer_end_song.start()

            self.eeg_source.marker_outlet.push_sample(["start_level_" + str(self.next_lvl)])

            # Start the online classification
            self.start_online()

        self.trained_classifier.decide_next_level()

    def start_online(self):
        ts = local_clock()

        timer = Timer(self.t_level + self.t_pause_screen, empty_timer_task)
        timer.start()

        while 1:
            window_level, var = self.eeg_source.get_piano_window_preprocessed(window_size=self.t_online_window,
                                                                              t_level=self.t_level)
            if var is not None:
                self.trained_classifier.predict_window(window_level)

                if var == 0:
                    break

        if timer.is_alive():
            # wait until it is time to send the command for the next level
            timer.join()

        print(f"Time passed: {local_clock() - ts}")

    def end_of_level(self, idx_level):
        # todo: check with lucchas??
        if idx_level + 1 < self.n_lvl:
            self.send_quest(self.quest_commands["force_end"])
            time.sleep(0.5)
            self.send_quest(self.quest_commands["toggle_pause_screen"])
        else:
            self.send_quest(self.quest_commands["force_end"])

    def trigger_pause_screen(self):
        self.send_quest(self.quest_commands["toggle_pause_screen"])

    def reload_measurement_function(self):
        online_windows, epoch_data, trained_pipeline, labels = self.save_data_instance.load_pickles()
        self.eeg_source.stream_name = "abc"

        # also reload classifier with potential different parameters
        if self.retrain_classifier:
            self.load_nback_data = True
            self.start_training()
        else:
            self.trained_classifier = Classifier(self.eeg_info, None, 500, self.adaptive)
            self.trained_classifier.trained_pipe = trained_pipeline

        # start the classification with the loaded online data
        for idx, levels in enumerate(online_windows):
            self.trained_classifier.decide_next_level()
            # if idx > 0:
            #     break

            whole_level = np.empty((np.size(levels[0], 0), 0)) #np.zeros((len(levels[0][0]),30000))

            for win in levels:
                if self.t_online_window == 5:
                    self.trained_classifier.predict_window(win)
                elif self.t_online_window == 2.5:
                    self.trained_classifier.predict_window(win[:, :1250])
                    self.trained_classifier.predict_window(win[:, 1250:2500])
                elif self.t_online_window == 2:
                    whole_level = np.concatenate((whole_level, win), 1)

            if self.t_online_window == 2:
                for i in range(int(50/2)):
                    self.trained_classifier.predict_window(whole_level[:, i*2*500:(i+1)*2*500])

        self.trained_classifier.decide_next_level()
        print(self.trained_classifier.mwl_mean)


def empty_timer_task():
    return
