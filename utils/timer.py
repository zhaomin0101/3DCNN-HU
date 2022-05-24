import time, datetime

class Timer(object):
    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.average_time = 0
        self.remain_time = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * (max_iters - iters) / iters

        return str(datetime.timedelta(seconds=int(self.remain_time)))

    def average(self):
        return str("%.3f" % self.average_time)

    def averageTostr(self):
        return str(datetime.timedelta(seconds=int(self.average_time)))
        # if type(self.average_time)==type(1):
        #     h=self.average_time/3600
        #     sUp_h=self.average_time-3600*h
        #     m=sUp_h/60
        #     sUp_m=sUp_h-60*m
        #     s=sUp_m
        #     return ":".join(map(str,(h,m,s)))
        # else:
        #     return "[InModuleError]:itv2time(iItv) invalid argument type"
