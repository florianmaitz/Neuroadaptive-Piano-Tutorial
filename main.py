import json
import time
from src.CommandUDP import CommandHandler
import sys
import atexit


def exit_handler():
    com_instance.eeg_source.eeg_inlet.close_stream()

atexit.register(exit_handler)

if len(sys.argv) == 2:
    ip = sys.argv[1]
    port = 8051
elif len(sys.argv) == 3:
    ip = sys.argv[1]
    port = int(sys.argv[2])
else:
    # ----> change before measurement <----
    ip = "192.168.0.114"
    port = 8051

if __name__ == '__main__':
    # todo:----> change before measurement <----
    subj_nr = 1
    adaptive = True

    # Load info files
    eeg_info = json.load(open('data/EEG_info.json'))
    command_info = json.load(open('data/Command_info.json'))
    t_info = json.load(open('data/Timing_info.json'))
    nback_data = json.load(open("data/n-back/" + str(subj_nr) + ".json"))

    # Create Command instance
    com_instance = CommandHandler(subj_nr=subj_nr, ip=ip, port=port, com_info=command_info, t_info=t_info,
                                  eeg_info=eeg_info, nback_data=nback_data, adaptive=adaptive)

    time.sleep(0.5)
    # com_instance.calc_delay()

    while 1:
        if not com_instance.busy:
            next_stage = com_instance.get_next_stage()
            msg = input(f'Stage ->{next_stage}<- is next. Confirm (y) or write command: ')
            com_instance.read_console(msg)
