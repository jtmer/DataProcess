# time_utils.py
import time
import logging

class TimeRecorder:
    def __init__(self, event_name):
        self.last_start_time = None
        self.duration = 0
        self.event_name = event_name

    def time_start(self):
        self.last_start_time = time.time()

    def time_end(self):
        d = time.time() - self.last_start_time
        self.duration += d
        logging.info(f"{self.event_name} last duration: {d}, cur total duration: {self.duration}")

    def get_total_duration(self):
        logging.info(f"Total {self.event_name} time: {self.duration}")
        return self.duration

def time_start():
    return time.time()

def log_time_delta(t, event_name):
    d = time.time() - t
    logging.info(f"{event_name} time: {d}")
