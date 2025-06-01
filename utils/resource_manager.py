# utils/resource_manager.py - NEW FILE
import psutil
import threading
from typing import Optional

class ResourceManager:
    """Manage system resources and prevent overload"""

    def __init__(self, max_memory_gb: float = 8.0, max_threads: int = 8):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.max_threads = max_threads
        self._active_threads = 0
        self._lock = threading.Lock()

    def can_process_file(self, file_size: int) -> bool:
        """Check if file can be processed within memory limits"""
        available_memory = psutil.virtual_memory().available
        return file_size < min(available_memory * 0.5, self.max_memory_bytes)

    def acquire_thread(self) -> bool:
        """Try to acquire a thread slot"""
        with self._lock:
            if self._active_threads < self.max_threads:
                self._active_threads += 1
                return True
            return False

    def release_thread(self):
        """Release a thread slot"""
        with self._lock:
            if self._active_threads > 0:
                self._active_threads -= 1

    def get_optimal_workers(self, total_files: int) -> int:
        """Calculate optimal number of workers"""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Conservative approach
        max_by_cpu = min(cpu_count, 8)
        max_by_memory = max(1, int(memory_gb // 2))
        max_by_files = min(total_files, 16)

        return min(max_by_cpu, max_by_memory, max_by_files)

# Global instance
resource_manager = ResourceManager()
