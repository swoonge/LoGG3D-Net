import time

class Timer_for_general:
    def __init__(self):
        self.start_times = []
        self.total_time = 0.0
        self.num_measurements = 0

    def tic(self):
        # 시간 측정 시작
        self.start_times.append(time.time())

    def toc(self):
        if not self.start_times:
            raise ValueError("tic() must be called before toc()")

        # 시간 계산
        start_time = self.start_times.pop()
        elapsed_time = time.time() - start_time
        self.total_time += elapsed_time
        self.num_measurements += 1
        return elapsed_time

    def average_time(self):
        if self.num_measurements == 0:
            return 0.0
        return self.total_time / self.num_measurements

    def reset(self):
        self.start_times = []
        self.total_time = 0.0
        self.num_measurements = 0