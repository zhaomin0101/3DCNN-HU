import sys

from utils.timer import Timer

class ProgressBar():
    def __init__(self, epoch_count, one_batch_count, pattern):
        self.total_count = one_batch_count
        self.current_index = 0
        self.current_epoch = 1
        self.epoch_count = epoch_count
        self.train_timer = Timer()
        self.pattern = pattern

    def show(self,currentEpoch, *args):
        self.current_index += 1
        if self.current_index == 1 :
            self.train_timer.tic()
        self.current_epoch = currentEpoch
        perCount = int(self.total_count / 100) # 7
        perCount = 1 if perCount == 0 else perCount
        percent = int(self.current_index / perCount)

        if self.total_count % perCount == 0:
            dotcount = int(self.total_count / perCount)
        else:
            dotcount = int(self.total_count / perCount)

        s1 = "\rEpoch:%d / %d [%s%s] %d / %d "%(
            self.current_epoch,
            self.epoch_count,
            "*"*(int(percent)),
            " "*(dotcount-int(percent)),
            self.current_index,
            self.total_count
        )

        s2 = self.pattern % tuple([float("{:.5f}".format(x)) for x in args])

        s3 = "%s,%s,remain=%s" % (
            s1, s2, self.train_timer.remain(self.current_index, self.total_count))
        sys.stdout.write(s3)
        sys.stdout.flush()
        if self.current_index == self.total_count :
            self.train_timer.toc()
            s3 = "%s,%s,total=%s" % (
                s1, s2, self.train_timer.averageTostr())
            sys.stdout.write(s3)
            sys.stdout.flush()
            self.current_index = 0
            print("\r")
            # self.current_epoch += 1