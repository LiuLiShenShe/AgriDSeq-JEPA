"""Read-only physical GPU1 telemetry, stopped by its owner after the run."""
import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml as nv

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
nv.nvmlInit()
handle = nv.nvmlDeviceGetHandleByUUID('GPU-6836f890-98b2-5891-07ef-675c25199c65')
with args.output.open('a', newline='') as f:
    fields = ['timestamp_utc','unix_seconds','uuid','gpu_util_percent','memory_used_mib','power_watts','temperature_c']
    writer = csv.DictWriter(f, fieldnames=fields)
    if f.tell() == 0:
        writer.writeheader()
    while True:
        writer.writerow(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), unix_seconds=time.time(),
                             uuid=nv.nvmlDeviceGetUUID(handle), gpu_util_percent=nv.nvmlDeviceGetUtilizationRates(handle).gpu,
                             memory_used_mib=nv.nvmlDeviceGetMemoryInfo(handle).used/1024**2,
                             power_watts=nv.nvmlDeviceGetPowerUsage(handle)/1000,
                             temperature_c=nv.nvmlDeviceGetTemperature(handle,nv.NVML_TEMPERATURE_GPU)))
        f.flush()
        time.sleep(10)
