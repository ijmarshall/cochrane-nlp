# 
# progress bar
#

import sys
import time



class ProgressBar():
    """
    Implments a simple progress bar to monitor progress of loops

    initiate with number_to_reach = the total number of loops expected
    call the tap function with every cycle of the loop, it will auto update
    as needed

    """

    def __init__(self, number_to_reach, timer=False):

        self.number_to_reach = number_to_reach
        self.update_interval = number_to_reach / 100
        if self.update_interval == 0:
            self.update_interval = 1
        self.counter = 0
        self.timer = timer
        if self.timer:
            self.start_time = time.time()
        self.update()

    def tap(self):
        self.counter += 1
        if self.counter % self.update_interval == 0 or self.counter == self.number_to_reach:
            self.update()

    def update(self):
        percentage_progress = (100 * self.counter) / (self.number_to_reach)    
        no_bars = percentage_progress / 5
        no_spaces = 20 - no_bars
        sys.stdout.write("\r[%s%s] %d%%" % ("=" * no_bars, " " * no_spaces, percentage_progress))    

        if self.timer:
            if percentage_progress == 0:
                sys.stdout.write(" - Calculating time")
            elif percentage_progress == 100:
            	sys.stdout.write(" - done!                     ")
            else:
                seconds_passed = time.time() - self.start_time
                seconds_to_go = ((seconds_passed / percentage_progress) * (100-percentage_progress))
                if seconds_to_go > 60:
                    minutes_to_go_rounded = int((seconds_to_go / 60) + 1)
                    time_text = "%d minute%s" % (minutes_to_go_rounded, (minutes_to_go_rounded > 1) * "s")
                elif seconds_to_go <= 10:
                    seconds_to_go_rounded = int(seconds_to_go + 1)
                    time_text = "%d second%s" % (seconds_to_go_rounded, (seconds_to_go_rounded > 1) * "s")
                else:
                    seconds_to_go_rounded = int((seconds_to_go + 5) / 5) * 5
                    time_text = "%d second%s" % (seconds_to_go_rounded, (seconds_to_go_rounded > 1) * "s")
                sys.stdout.write(" - Around %s remaining     " % (time_text, ))


        if percentage_progress == 100:
            sys.stdout.write("\n")
        sys.stdout.flush()



def example():

    p = ProgressBar(10000, timer=True)
    for i in range(10000):
        p.tap()
        time.sleep(0.0005)



def main():
    example()

if __name__ == '__main__':
    main()