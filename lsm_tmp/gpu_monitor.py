# lsm_tmp/gpu_monitor.py 전체 내용

import time
import threading
from datetime import datetime
from pynvml import *


class GPUMonitor:
    def __init__(self, device_index=0, log_file=None, label = ""):
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(device_index)
            self.monitoring = False
            self.gpu_utils = []
            self.mem_utils = []
            self.mem_used_mb = []
            self.has_gpu = True
            self.start_time = 0
            self.log_file = log_file
            self.label = label
        except NVMLError:
            self.has_gpu = False
            print("NVIDIA GPU 또는 드라이버를 찾을 수 없습니다. GPU 모니터링을 건너뜁니다.")

    def _monitor(self):
        while self.monitoring:
            util = nvmlDeviceGetUtilizationRates(self.handle)
            self.gpu_utils.append(util.gpu)
            self.mem_utils.append(util.memory)
            
            mem_info = nvmlDeviceGetMemoryInfo(self.handle)
            self.mem_used_mb.append(mem_info.used / (1024**2))
            time.sleep(0.1)

    def start(self):
        if not self.has_gpu: return
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        if not self.has_gpu: return
        duration = time.time() - self.start_time
        self.monitoring = False
        self.monitor_thread.join()
        
        if not self.gpu_utils: 
            #nvmlShutdown() # 추가: 데이터 없어도 종료는 호출
            return

        avg_gpu_util = sum(self.gpu_utils) / len(self.gpu_utils)
        max_gpu_util = max(self.gpu_utils)
        avg_mem_used = sum(self.mem_used_mb) / len(self.mem_used_mb)
        max_mem_used = max(self.mem_used_mb)
        total_mem = nvmlDeviceGetMemoryInfo(self.handle).total / (1024**2)

        log_message = (
            f"--- {self.label} 결과---\n"
            f"  - 소요 시간: {duration:.4f} 초\n"
            f"  - 평균 GPU 사용률: {avg_gpu_util:.2f}% (최대: {max_gpu_util:.2f}%)\n"
            f"  - 평균 VRAM 사용량: {avg_mem_used:.2f} MB (최대: {max_mem_used:.2f} MB) / 총 {total_mem:.2f} MB\n"
        )
        print(log_message)

        if self.log_file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}]\n{log_message}\n")
        
        #nvmlShutdown()