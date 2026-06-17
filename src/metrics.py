"""
Metrics Collector — Tracks pipeline performance for the hackathon dashboard.
Captures per-step timing, GPU/memory usage, and token consumption.
"""
import time
import os
import json
from datetime import datetime


class PipelineMetrics:
    """Collects and aggregates pipeline performance metrics."""

    def __init__(self):
        self._steps = {}  # step_name -> {start, end, duration}
        self._current_step = None
        self._start_time = time.time()
        self.gpu_info = self._detect_gpu()
        self.system_info = self._get_system_info()

    def start_step(self, step_name: str):
        """Mark the start of a pipeline step."""
        self._current_step = step_name
        self._steps[step_name] = {
            "start": time.time(),
            "end": None,
            "duration_sec": 0,
        }

    def end_step(self, step_name: str = None, extra: dict = None):
        """Mark the end of a pipeline step."""
        name = step_name or self._current_step
        if name and name in self._steps:
            self._steps[name]["end"] = time.time()
            self._steps[name]["duration_sec"] = round(
                self._steps[name]["end"] - self._steps[name]["start"], 2
            )
            if extra:
                self._steps[name].update(extra)
        self._current_step = None

    def get_step_duration(self, step_name: str) -> float:
        """Get duration of a specific step in seconds."""
        step = self._steps.get(step_name, {})
        return step.get("duration_sec", 0)

    def get_total_duration(self) -> float:
        """Get total pipeline duration in seconds."""
        return round(time.time() - self._start_time, 2)

    def _detect_gpu(self) -> dict:
        """Detect GPU information if available."""
        info = {
            "available": False,
            "name": "N/A",
            "memory_total_gb": 0,
            "memory_used_gb": 0,
            "utilization_pct": 0,
        }

        # Try PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                info["available"] = True
                info["name"] = torch.cuda.get_device_name(0)
                mem = torch.cuda.mem_get_info(0)
                info["memory_total_gb"] = round(mem[1] / (1024**3), 1)
                info["memory_used_gb"] = round((mem[1] - mem[0]) / (1024**3), 1)
                info["utilization_pct"] = round(
                    info["memory_used_gb"] / info["memory_total_gb"] * 100, 1
                )
        except Exception:
            pass

        # Try ROCm SMI (AMD GPUs)
        if not info["available"]:
            try:
                import subprocess
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info["available"] = True
                    info["name"] = "AMD Instinct MI300X (ROCm)"
                    # Parse CSV output
                    for line in result.stdout.strip().split("\n")[1:]:
                        parts = line.split(",")
                        if len(parts) >= 3:
                            info["memory_total_gb"] = round(int(parts[1]) / (1024**3), 1)
                            info["memory_used_gb"] = round(int(parts[2]) / (1024**3), 1)
                            break
            except Exception:
                pass

        return info

    def _get_system_info(self) -> dict:
        """Get basic system information."""
        info = {
            "ram_total_gb": 0,
            "ram_used_gb": 0,
            "cpu_count": os.cpu_count() or 1,
        }
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["ram_total_gb"] = round(mem.total / (1024**3), 1)
            info["ram_used_gb"] = round(mem.used / (1024**3), 1)
        except ImportError:
            pass
        return info

    def snapshot_memory(self) -> dict:
        """Take a snapshot of current memory usage."""
        snap = {"timestamp": time.time()}
        try:
            import psutil
            mem = psutil.virtual_memory()
            snap["ram_used_gb"] = round(mem.used / (1024**3), 1)
            snap["ram_pct"] = mem.percent
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info(0)
                snap["gpu_used_gb"] = round((mem[1] - mem[0]) / (1024**3), 1)
        except Exception:
            pass
        return snap

    def to_dict(self, llm_stats: dict = None, doc_stats: dict = None) -> dict:
        """
        Export all metrics as a serializable dict for the Streamlit dashboard.

        Args:
            llm_stats: Output of LLMClient.get_stats()
            doc_stats: {pages, tables, sections, chunks}
        """
        pipeline_breakdown = {}
        for name, step in self._steps.items():
            pipeline_breakdown[name] = {
                "duration_sec": step.get("duration_sec", 0),
                "pct_of_total": 0,
            }

        total = self.get_total_duration()
        if total > 0:
            for name in pipeline_breakdown:
                pipeline_breakdown[name]["pct_of_total"] = round(
                    pipeline_breakdown[name]["duration_sec"] / total * 100, 1
                )

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_sec": total,
            "pipeline_breakdown": pipeline_breakdown,
            "gpu": self.gpu_info,
            "system": self.system_info,
            "memory_snapshot": self.snapshot_memory(),
        }

        if llm_stats:
            metrics["llm"] = llm_stats
        if doc_stats:
            metrics["document"] = doc_stats

        return metrics

    def save(self, filepath: str, llm_stats: dict = None, doc_stats: dict = None):
        """Save metrics to a JSON file."""
        data = self.to_dict(llm_stats, doc_stats)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  📊 Metrics saved to: {filepath}")
        return data
