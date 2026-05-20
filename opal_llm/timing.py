import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List


def cuda_synchronize_if_needed() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


@dataclass
class TimedBatch:
    duration_sec: float
    samples: int
    warmup: bool


class GenerationTimer:
    def __init__(self, warmup_batches: int = 1, timed_batches: int = 0):
        self.warmup_batches = max(0, int(warmup_batches))
        self.timed_batches = max(0, int(timed_batches))
        self.records: List[TimedBatch] = []
        self._seen_batches = 0

    def should_stop(self) -> bool:
        if self.timed_batches <= 0:
            return False
        timed = len([record for record in self.records if not record.warmup])
        return timed >= self.timed_batches

    def measure(self, fn, samples: int):
        warmup = self._seen_batches < self.warmup_batches
        self._seen_batches += 1
        cuda_synchronize_if_needed()
        start = time.perf_counter()
        result = fn()
        cuda_synchronize_if_needed()
        duration = time.perf_counter() - start
        self.records.append(TimedBatch(duration_sec=duration, samples=int(samples), warmup=warmup))
        return result

    def summary(self) -> Dict[str, float]:
        timed = [record for record in self.records if not record.warmup]
        if not timed:
            return {
                "warmup_batches": self.warmup_batches,
                "timed_batches": 0,
                "latency_mean_sec": 0.0,
                "latency_std_sec": 0.0,
                "latency_p50_sec": 0.0,
                "latency_p95_sec": 0.0,
                "throughput_samples_per_sec": 0.0,
            }

        durations = [record.duration_sec for record in timed]
        total_samples = sum(record.samples for record in timed)
        total_time = sum(durations)
        sorted_durations = sorted(durations)

        def percentile(p: float) -> float:
            if len(sorted_durations) == 1:
                return sorted_durations[0]
            pos = (len(sorted_durations) - 1) * p
            lower = math.floor(pos)
            upper = math.ceil(pos)
            if lower == upper:
                return sorted_durations[int(pos)]
            weight = pos - lower
            return sorted_durations[lower] * (1 - weight) + sorted_durations[upper] * weight

        return {
            "warmup_batches": self.warmup_batches,
            "timed_batches": len(timed),
            "latency_mean_sec": statistics.mean(durations),
            "latency_std_sec": statistics.pstdev(durations) if len(durations) > 1 else 0.0,
            "latency_p50_sec": percentile(0.50),
            "latency_p95_sec": percentile(0.95),
            "throughput_samples_per_sec": total_samples / total_time if total_time > 0 else 0.0,
        }

    def raw_records(self) -> List[Dict[str, object]]:
        return [record.__dict__.copy() for record in self.records]

