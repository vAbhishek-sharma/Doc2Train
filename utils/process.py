# utils/process.py
"""
Complete process utilities for Doc2Train v2.0 Enhanced
Process management, threading utilities, and system monitoring
"""

import os
import sys
import time
import psutil
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import signal

class ProcessManager:
    """
    Advanced process manager for handling parallel processing
    """

    def __init__(self, max_workers: int = None, use_processes: bool = False):
        """
        Initialize process manager

        Args:
            max_workers: Maximum number of workers (auto-detect if None)
            use_processes: Use processes instead of threads for CPU-bound tasks
        """
        self.max_workers = max_workers or self._get_optimal_worker_count()
        self.use_processes = use_processes
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_counter = 0
        self.lock = threading.Lock()

        # Performance monitoring
        self.start_time = None
        self.total_tasks = 0
        self.completed_count = 0
        self.failed_count = 0

    def _get_optimal_worker_count(self) -> int:
        """Determine optimal number of workers based on system"""
        cpu_count = multiprocessing.cpu_count()

        try:
            # Consider available memory
            memory_gb = psutil.virtual_memory().total / (1024**3)

            # For I/O bound tasks (threading), use more workers
            if not self.use_processes:
                return min(cpu_count * 2, 16)  # Cap at 16 threads

            # For CPU bound tasks (multiprocessing), use fewer workers
            if memory_gb < 4:
                return max(1, cpu_count - 1)  # Leave 1 CPU free on low-mem systems
            else:
                return cpu_count

        except:
            return max(1, cpu_count - 1)

    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """
        Submit a task for processing

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Task ID
        """
        with self.lock:
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1

            self.active_tasks[task_id] = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs,
                'submitted_at': time.time(),
                'started_at': None,
                'completed_at': None
            }

            return task_id

    def process_tasks_parallel(self, tasks: List[Dict[str, Any]],
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process multiple tasks in parallel

        Args:
            tasks: List of task dictionaries with 'func', 'args', 'kwargs'
            progress_callback: Optional callback for progress updates

        Returns:
            Results dictionary
        """
        self.start_time = time.time()
        self.total_tasks = len(tasks)
        self.completed_count = 0
        self.failed_count = 0

        results = {
            'successful': {},
            'failed': {},
            'summary': {}
        }

        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, task in enumerate(tasks):
                task_id = f"task_{i}"
                future = executor.submit(task['func'], *task.get('args', []), **task.get('kwargs', {}))
                future_to_task[future] = task_id

            # Process completed tasks
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]

                try:
                    result = future.result()
                    results['successful'][task_id] = result
                    self.completed_count += 1

                    with self.lock:
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]['completed_at'] = time.time()
                            self.completed_tasks[task_id] = self.active_tasks.pop(task_id)

                except Exception as e:
                    results['failed'][task_id] = str(e)
                    self.failed_count += 1

                    with self.lock:
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]['failed_at'] = time.time()
                            self.active_tasks[task_id]['error'] = str(e)
                            self.failed_tasks[task_id] = self.active_tasks.pop(task_id)

                # Progress callback
                if progress_callback:
                    progress_callback(self.completed_count + self.failed_count, self.total_tasks)

        # Generate summary
        total_time = time.time() - self.start_time
        results['summary'] = {
            'total_tasks': self.total_tasks,
            'completed': self.completed_count,
            'failed': self.failed_count,
            'success_rate': self.completed_count / self.total_tasks if self.total_tasks > 0 else 0,
            'total_time': total_time,
            'avg_time_per_task': total_time / self.total_tasks if self.total_tasks > 0 else 0,
            'tasks_per_second': self.total_tasks / total_time if total_time > 0 else 0
        }

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        with self.lock:
            current_time = time.time()

            status = {
                'total_tasks': self.total_tasks,
                'completed': self.completed_count,
                'failed': self.failed_count,
                'active': len(self.active_tasks),
                'max_workers': self.max_workers,
                'use_processes': self.use_processes,
                'elapsed_time': current_time - self.start_time if self.start_time else 0
            }

            # Calculate progress percentage
            if self.total_tasks > 0:
                processed = self.completed_count + self.failed_count
                status['progress_percent'] = (processed / self.total_tasks) * 100
            else:
                status['progress_percent'] = 0

            # Calculate ETA
            if self.completed_count > 0 and self.start_time:
                elapsed = current_time - self.start_time
                avg_time_per_task = elapsed / (self.completed_count + self.failed_count)
                remaining_tasks = self.total_tasks - self.completed_count - self.failed_count
                status['eta_seconds'] = avg_time_per_task * remaining_tasks
            else:
                status['eta_seconds'] = None

            return status

class SystemMonitor:
    """
    System resource monitoring for performance optimization
    """

    def __init__(self, interval: float = 1.0):
        """
        Initialize system monitor

        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.stats = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.stats = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect system stats
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('.')

                # Process info
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()

                stats = {
                    'timestamp': time.time(),
                    'system': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_gb': memory.available / (1024**3),
                        'disk_free_gb': disk.free / (1024**3)
                    },
                    'process': {
                        'cpu_percent': process_cpu,
                        'memory_mb': process_memory.rss / (1024**2),
                        'threads': process.num_threads()
                    }
                }

                self.stats.append(stats)

                # Keep only last 1000 entries
                if len(self.stats) > 1000:
                    self.stats = self.stats[-1000:]

            except Exception:
                # Ignore monitoring errors
                pass

            time.sleep(self.interval)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': memory.total / (1024**3)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_average_stats(self, last_n: int = 60) -> Dict[str, Any]:
        """Get average statistics over last N samples"""
        if not self.stats:
            return {}

        recent_stats = self.stats[-last_n:] if len(self.stats) > last_n else self.stats

        if not recent_stats:
            return {}

        # Calculate averages
        cpu_avg = sum(s['system']['cpu_percent'] for s in recent_stats) / len(recent_stats)
        memory_avg = sum(s['system']['memory_percent'] for s in recent_stats) / len(recent_stats)
        process_cpu_avg = sum(s['process']['cpu_percent'] for s in recent_stats) / len(recent_stats)
        process_memory_avg = sum(s['process']['memory_mb'] for s in recent_stats) / len(recent_stats)

        return {
            'cpu_percent_avg': cpu_avg,
            'memory_percent_avg': memory_avg,
            'process_cpu_percent_avg': process_cpu_avg,
            'process_memory_mb_avg': process_memory_avg,
            'sample_count': len(recent_stats),
            'time_span_seconds': recent_stats[-1]['timestamp'] - recent_stats[0]['timestamp']
        }

class TaskQueue:
    """
    Thread-safe task queue for managing work items
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize task queue

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self.queue = Queue(maxsize=maxsize)
        self.results = Queue()
        self.workers = []
        self.running = False
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()

    def add_task(self, func: Callable, *args, **kwargs):
        """Add task to queue"""
        task = {
            'id': f"task_{self.total_tasks}",
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'added_at': time.time()
        }

        self.queue.put(task)

        with self.lock:
            self.total_tasks += 1

    def start_workers(self, num_workers: int = 4):
        """Start worker threads"""
        if self.running:
            return

        self.running = True
        self.workers = []

        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)

    def stop_workers(self, timeout: float = 10.0):
        """Stop worker threads"""
        self.running = False

        # Add sentinel values to wake up workers
        for _ in self.workers:
            self.queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)

        self.workers = []

    def _worker_loop(self, worker_id: int):
        """Main worker loop"""
        while self.running:
            try:
                task = self.queue.get(timeout=1.0)

                if task is None:  # Sentinel value
                    break

                try:
                    # Execute task
                    result = task['func'](*task['args'], **task['kwargs'])

                    # Store result
                    self.results.put({
                        'id': task['id'],
                        'result': result,
                        'success': True,
                        'worker_id': worker_id,
                        'completed_at': time.time()
                    })

                    with self.lock:
                        self.completed_tasks += 1

                except Exception as e:
                    # Store error
                    self.results.put({
                        'id': task['id'],
                        'error': str(e),
                        'success': False,
                        'worker_id': worker_id,
                        'completed_at': time.time()
                    })

                    with self.lock:
                        self.failed_tasks += 1

                finally:
                    self.queue.task_done()

            except Empty:
                continue
            except Exception:
                break

    def get_results(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Get completed results"""
        results = []

        while True:
            try:
                result = self.results.get(timeout=timeout)
                results.append(result)
                self.results.task_done()
            except Empty:
                break

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get queue status"""
        with self.lock:
            return {
                'total_tasks': self.total_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'pending_tasks': self.queue.qsize(),
                'pending_results': self.results.qsize(),
                'active_workers': len([w for w in self.workers if w.is_alive()]),
                'running': self.running
            }

class ProcessLimiter:
    """
    Limit resource usage to prevent system overload
    """

    def __init__(self, max_cpu_percent: float = 80.0, max_memory_percent: float = 80.0):
        """
        Initialize process limiter

        Args:
            max_cpu_percent: Maximum CPU usage percentage
            max_memory_percent: Maximum memory usage percentage
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.monitoring = False
        self.throttle_factor = 1.0
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_resources(self):
        """Monitor system resources and adjust throttling"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_percent = psutil.virtual_memory().percent

                # Adjust throttle factor based on resource usage
                if cpu_percent > self.max_cpu_percent or memory_percent > self.max_memory_percent:
                    self.throttle_factor = max(0.1, self.throttle_factor * 0.8)
                elif cpu_percent < self.max_cpu_percent * 0.7 and memory_percent < self.max_memory_percent * 0.7:
                    self.throttle_factor = min(1.0, self.throttle_factor * 1.1)

            except Exception:
                pass

            time.sleep(1.0)

    def should_throttle(self) -> bool:
        """Check if processing should be throttled"""
        return self.throttle_factor < 1.0

    def get_throttle_delay(self) -> float:
        """Get delay time for throttling"""
        if self.throttle_factor >= 1.0:
            return 0.0

        return (1.0 - self.throttle_factor) * 2.0  # Up to 2 seconds delay

# Signal handling for graceful shutdown
class GracefulShutdown:
    """
    Handle graceful shutdown on system signals
    """

    def __init__(self):
        self.shutdown_requested = False
        self.shutdown_callbacks = []

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⚠️ Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

        # Execute shutdown callbacks
        for callback in self.shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in shutdown callback: {e}")

    def add_shutdown_callback(self, callback: Callable):
        """Add callback to execute on shutdown"""
        self.shutdown_callbacks.append(callback)

    def check_shutdown(self) -> bool:
        """Check if shutdown was requested"""
        return self.shutdown_requested

# Utility functions
def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    try:
        return {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': multiprocessing.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('.').total / (1024**3),
            'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
            'process_id': os.getpid(),
            'process_memory_mb': psutil.Process().memory_info().rss / (1024**2)
        }
    except Exception as e:
        return {'error': str(e)}

def optimize_worker_count(task_type: str = 'io_bound') -> int:
    """
    Optimize worker count based on task type and system resources

    Args:
        task_type: Type of task ('io_bound', 'cpu_bound', 'mixed')

    Returns:
        Optimal number of workers
    """
    cpu_count = multiprocessing.cpu_count()

    try:
        memory_gb = psutil.virtual_memory().total / (1024**3)

        if task_type == 'io_bound':
            # I/O bound tasks can use more workers
            return min(cpu_count * 3, 24)

        elif task_type == 'cpu_bound':
            # CPU bound tasks should not exceed CPU count
            if memory_gb < 4:
                return max(1, cpu_count - 1)
            else:
                return cpu_count

        else:  # mixed
            # Mixed workload
            return min(cpu_count * 2, 16)

    except:
        return max(1, cpu_count - 1)

def wait_for_resources(max_cpu: float = 80.0, max_memory: float = 80.0, timeout: float = 60.0) -> bool:
    """
    Wait for system resources to be available

    Args:
        max_cpu: Maximum CPU percentage to wait for
        max_memory: Maximum memory percentage to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        True if resources are available, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory_percent = psutil.virtual_memory().percent

            if cpu_percent <= max_cpu and memory_percent <= max_memory:
                return True

            # Wait a bit before checking again
            time.sleep(2.0)

        except Exception:
            break

    return False
