from time import time

class TimeLogger:

    def __init__(self, start=False):
        self.total = None
        self.iteration = None

        self.start_time = None
        self.current_time = None

        if start:
            self.start_time = time()
            self.current_time = self.start_time

    # Print iterations progress
    def printProgressBar(self, iteration, total, sec, elapsed, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        return '{} |{}| {}% {} ({:.2f} sec / {:.2f} elapsed)'.format(prefix, bar, percent, suffix, sec, elapsed)

    def initialize(self, total):
        self.total = total
        self.start_time = time()
        self.current_time = time()

    def progress(self, iteration):
        t0 = time()
        delta_time = t0 - self.current_time
        elapsed_time = (self.current_time - self.start_time)
        self.current_time = t0
        return self.printProgressBar(iteration, self.total, delta_time, elapsed_time)

    def delta(self):
        t0 = time()
        delta_time = t0 - self.current_time
        elapsed_time = (self.current_time - self.start_time)
        self.current_time = t0
        return "{:.2f} sec / {:.2f} elapsed".format(delta_time, elapsed_time)
