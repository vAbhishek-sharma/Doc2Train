# utils/progress.py
"""
Complete real-time progress tracking system for Doc2Train
Thread-safe progress updates with ETA calculation and statistics
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class ProcessingStats:
    """Container for processing statistics"""
    files_total: int = 0
    files_completed: int = 0
    files_failed: int = 0
    current_file: str = ""
    start_time: Optional[float] = None
    errors: List[Dict] = field(default_factory=list)
    total_text_chars: int = 0
    total_images: int = 0
    total_processing_time: float = 0.0

class ProgressTracker:
    """Thread-safe progress tracking with detailed statistics"""

    def __init__(self):
        self.stats = ProcessingStats()
        self.lock = threading.Lock()
        self._last_update = 0
        self._update_interval = 0.5  # Update display every 0.5 seconds

    def initialize(self, total_files: int):
        """Initialize progress tracking for a batch of files"""
        with self.lock:
            self.stats = ProcessingStats(
                files_total=total_files,
                start_time=time.time()
            )

    def start_file(self, file_name: str):
        """Mark start of file processing"""
        with self.lock:
            self.stats.current_file = file_name

    def complete_file(self, file_name: str, text_chars: int = 0, image_count: int = 0,
                     processing_time: float = 0, success: bool = True):
        """Mark file as completed with statistics"""
        with self.lock:
            if success:
                self.stats.files_completed += 1
                self.stats.total_text_chars += text_chars
                self.stats.total_images += image_count
                self.stats.total_processing_time += processing_time
            else:
                self.stats.files_failed += 1

    def add_error(self, file_name: str, error: str):
        """Add error to tracking"""
        with self.lock:
            self.stats.errors.append({
                'file': file_name,
                'error': error,
                'timestamp': time.time()
            })

    def get_stats(self) -> ProcessingStats:
        """Get current statistics (thread-safe copy)"""
        with self.lock:
            # Return a copy to avoid threading issues
            return ProcessingStats(
                files_total=self.stats.files_total,
                files_completed=self.stats.files_completed,
                files_failed=self.stats.files_failed,
                current_file=self.stats.current_file,
                start_time=self.stats.start_time,
                errors=self.stats.errors.copy(),
                total_text_chars=self.stats.total_text_chars,
                total_images=self.stats.total_images,
                total_processing_time=self.stats.total_processing_time
            )

    def get_completed_count(self) -> int:
        """Get number of completed files"""
        with self.lock:
            return self.stats.files_completed

    def get_total_count(self) -> int:
        """Get total number of files"""
        with self.lock:
            return self.stats.files_total

    def should_update_display(self) -> bool:
        """Check if display should be updated (rate limiting)"""
        current_time = time.time()
        if current_time - self._last_update >= self._update_interval:
            self._last_update = current_time
            return True
        return False

class ProgressDisplay:
    """Real-time progress display with formatting"""

    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self.last_line_length = 0

    def set_show_progress(self, show: bool):
        """Enable or disable progress display"""
        self.show_progress = show

    def print_progress(self, stats: ProcessingStats):
        """Print current progress status with ETA"""
        if not self.show_progress:
            return

        completed = stats.files_completed
        failed = stats.files_failed
        total = stats.files_total
        current = stats.current_file

        if stats.start_time:
            elapsed = time.time() - stats.start_time
            processed = completed + failed
            if processed > 0:
                avg_time_per_file = elapsed / processed
                eta_seconds = avg_time_per_file * (total - processed)
                eta_str = f"⏱️ ETA: {self._format_time(eta_seconds)}"
            else:
                eta_str = "⏱️ ETA: calculating..."
        else:
            eta_str = ""

        processed = completed + failed
        percentage = (processed / total * 100) if total > 0 else 0

        # Truncate current file name if too long
        display_current = current[:35] + "..." if len(current) > 38 else current

        # Build progress line with detailed stats
        progress_line = (
            f"\r🔄 Progress: {processed}/{total} ({percentage:.1f}%) | "
            f"✅{completed} ❌{failed} | "
            f"📄{stats.total_text_chars//1000}K chars 🖼️{stats.total_images} imgs | "
            f"Current: {display_current} | {eta_str}"
        )

        # Clear previous line if it was longer
        if len(progress_line) < self.last_line_length:
            progress_line += " " * (self.last_line_length - len(progress_line))

        print(progress_line, end="", flush=True)
        self.last_line_length = len(progress_line)

    def print_file_completion(self, file_name: str, text_chars: int, image_count: int,
                            processing_time: float, success: bool = True):
        """Print individual file completion (when not showing progress)"""
        if self.show_progress:
            return  # Don't print individual completions when showing progress

        if success:
            print(f"✅ Completed: {file_name}")
            print(f"   📄 Text: {text_chars:,} chars")
            print(f"   🖼️ Images: {image_count} extracted")
            print(f"   ⏱️ Time: {processing_time:.1f}s")
        else:
            print(f"❌ Failed: {file_name}")

    def print_error(self, file_name: str, error: str):
        """Print error message"""
        if self.show_progress:
            print()  # New line to clear progress line
        print(f"❌ Error processing {file_name}: {error}")

    def print_completion_summary(self, stats: ProcessingStats, config: Dict):
        """Print final completion summary"""
        if self.show_progress:
            print()  # New line after progress

        if stats.start_time:
            total_time = time.time() - stats.start_time
        else:
            total_time = stats.total_processing_time

        print(f"\n📊 Processing Summary:")
        print(f"   ⏱️ Total time: {self._format_time(total_time)}")
        print(f"   📄 Files processed: {stats.files_total}")
        print(f"   ✅ Successful: {stats.files_completed}")

        if stats.files_failed > 0:
            print(f"   ❌ Failed: {stats.files_failed}")

        # Performance metrics
        if stats.files_completed > 0:
            avg_time = total_time / stats.files_total if stats.files_total > 0 else 0
            print(f"   📈 Average time per file: {avg_time:.1f}s")

            threads = config.get('threads', 1)
            if threads > 1:
                sequential_estimate = avg_time * stats.files_total
                speedup = sequential_estimate / total_time if total_time > 0 else 1
                print(f"   🚀 Threading speedup: {speedup:.1f}x")

        # Content statistics
        if stats.total_text_chars > 0:
            print(f"\n📝 Content Extracted:")
            print(f"   Text: {stats.total_text_chars:,} characters")
            print(f"   Images: {stats.total_images:,}")

            if stats.files_completed > 0:
                avg_text_per_file = stats.total_text_chars / stats.files_completed
                avg_images_per_file = stats.total_images / stats.files_completed
                print(f"   Average per file: {avg_text_per_file:,.0f} chars, {avg_images_per_file:.1f} images")

        # Show recent errors if any
        if stats.errors:
            print(f"\n⚠️ Recent errors:")
            for error in stats.errors[-3:]:  # Show last 3 errors
                print(f"   ❌ {error['file']}: {error['error']}")

            if len(stats.errors) > 3:
                print(f"   ... and {len(stats.errors) - 3} more errors")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

class PerformanceMonitor:
    """Monitor and analyze processing performance"""

    def __init__(self):
        self.file_times = {}
        self.processor_stats = {}

    def record_file_processing(self, file_path: str, processor: str,
                             processing_time: float, text_chars: int, image_count: int):
        """Record processing statistics for a file"""
        self.file_times[file_path] = {
            'processor': processor,
            'time': processing_time,
            'text_chars': text_chars,
            'image_count': image_count,
            'chars_per_second': text_chars / processing_time if processing_time > 0 else 0
        }

        # Update processor statistics
        if processor not in self.processor_stats:
            self.processor_stats[processor] = {
                'files': 0,
                'total_time': 0,
                'total_chars': 0,
                'total_images': 0
            }

        stats = self.processor_stats[processor]
        stats['files'] += 1
        stats['total_time'] += processing_time
        stats['total_chars'] += text_chars
        stats['total_images'] += image_count

    def get_performance_report(self) -> Dict:
        """Generate performance analysis report"""
        if not self.file_times:
            return {}

        # Overall statistics
        total_files = len(self.file_times)
        total_time = sum(f['time'] for f in self.file_times.values())
        total_chars = sum(f['text_chars'] for f in self.file_times.values())
        total_images = sum(f['image_count'] for f in self.file_times.values())

        # Processor performance
        processor_performance = {}
        for processor, stats in self.processor_stats.items():
            avg_time = stats['total_time'] / stats['files'] if stats['files'] > 0 else 0
            avg_chars = stats['total_chars'] / stats['files'] if stats['files'] > 0 else 0
            chars_per_second = stats['total_chars'] / stats['total_time'] if stats['total_time'] > 0 else 0

            processor_performance[processor] = {
                'files_processed': stats['files'],
                'avg_time_per_file': avg_time,
                'avg_chars_per_file': avg_chars,
                'chars_per_second': chars_per_second,
                'total_images': stats['total_images']
            }

        # Identify slow files
        file_times_list = [(path, data['time']) for path, data in self.file_times.items()]
        file_times_list.sort(key=lambda x: x[1], reverse=True)
        slowest_files = file_times_list[:5]  # Top 5 slowest files

        return {
            'overall': {
                'total_files': total_files,
                'total_time': total_time,
                'total_chars': total_chars,
                'total_images': total_images,
                'avg_time_per_file': total_time / total_files if total_files > 0 else 0,
                'chars_per_second': total_chars / total_time if total_time > 0 else 0
            },
            'by_processor': processor_performance,
            'slowest_files': slowest_files
        }

# Global instances for easy access
_global_progress_tracker = ProgressTracker()
_global_progress_display = ProgressDisplay()
_global_performance_monitor = PerformanceMonitor()

def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance"""
    return _global_progress_tracker

def get_progress_display() -> ProgressDisplay:
    """Get the global progress display instance"""
    return _global_progress_display

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _global_performance_monitor

# Convenience functions for use throughout the application
def initialize_progress(total_files: int, show_progress: bool = True):
    """Initialize progress tracking for a batch"""
    _global_progress_tracker.initialize(total_files)
    _global_progress_display.set_show_progress(show_progress)

def start_file_processing(file_name: str):
    """Mark start of file processing"""
    _global_progress_tracker.start_file(file_name)

def complete_file_processing(file_name: str, text_chars: int = 0, image_count: int = 0,
                           processing_time: float = 0, success: bool = True, processor: str = ""):
    """Mark file processing as complete"""
    _global_progress_tracker.complete_file(file_name, text_chars, image_count, processing_time, success)
    _global_progress_display.print_file_completion(file_name, text_chars, image_count, processing_time, success)

    # Record performance data
    if success and processor:
        _global_performance_monitor.record_file_processing(file_name, processor, processing_time, text_chars, image_count)

def add_processing_error(file_name: str, error: str):
    """Add error to progress tracking"""
    _global_progress_tracker.add_error(file_name, error)
    _global_progress_display.print_error(file_name, error)

def update_progress_display():
    """Update progress display if enough time has passed"""
    if _global_progress_tracker.should_update_display():
        stats = _global_progress_tracker.get_stats()
        _global_progress_display.print_progress(stats)

def show_completion_summary(config: Dict):
    """Show final completion summary"""
    stats = _global_progress_tracker.get_stats()
    _global_progress_display.print_completion_summary(stats, config)

def get_processing_stats() -> ProcessingStats:
    """Get current processing statistics"""
    return _global_progress_tracker.get_stats()

def get_performance_report() -> Dict:
    """Get performance analysis report"""
    return _global_performance_monitor.get_performance_report()
