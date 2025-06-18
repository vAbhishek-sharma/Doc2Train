# core/pipeline.py
"""
Complete processing pipeline for Doc2Train v2.0 Enhanced
Orchestrates the entire document processing workflow with parallel execution
"""

import ipdb
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from doc2train.core.writers import OutputManager
from doc2train.utils.resource_manager import resource_manager
import os

from doc2train.core.registries.processor_registry import get_processor_for_file
from doc2train.core.generator import generate_training_data
from doc2train.core.llm_client import get_available_providers
from doc2train.utils.progress import (
    initialize_progress, start_file_processing, complete_file_processing,
    add_processing_error, update_progress_display, show_completion_summary
)
from doc2train.utils.validation import validate_input_and_files
from doc2train.utils.process import ProcessManager
class BaseProcessor:
    def process_files(self, file_paths: List[str], args=None) -> Dict[str, Any]:
        raise NotImplementedError

class ProcessingPipeline(BaseProcessor):
    """
    Complete processing pipeline with parallel execution and fault tolerance
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize processing pipeline"""
        self.config = config
        self.stats = {
            'files_total': 0,
            'files_successful': 0,
            'files_failed': 0,
            'consecutive_failures': 0,  # NEW: Track consecutive failures
            'total_text_chars': 0,
            'total_images': 0,
            'total_processing_time': 0.0,
            'errors': [],
            'should_stop': False,  # NEW: Auto-stop flag
            'stop_reason': ''  # NEW: Reason for stopping
        }
        self.start_time = None  # NEW: Track processing start time
        self.use_resource_limits = config.get("use_resource_limits", True)  # ADDED

        self.output_manager = OutputManager(config)
        if 'threads' not in self.config or not self.config['threads']:
            self.config['threads'] = resource_manager.get_optimal_workers(
                self.stats.get('files_total', 1)
            )

        self._setup_pipeline()

    def _should_auto_stop(self) -> Tuple[bool, str]:
        """Check if processing should auto-stop"""

        # Check consecutive failures
        max_consecutive = self.config.get('auto_stop_on_consecutive_failures', 3)
        if max_consecutive and self.stats['consecutive_failures'] >= max_consecutive:
            return True, f"Too many consecutive failures ({self.stats['consecutive_failures']})"

        # Check time limit
        time_limit_minutes = self.config.get('auto_stop_after_time')
        if time_limit_minutes and self.start_time:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= time_limit_minutes:
                return True, f"Time limit reached ({elapsed_minutes:.1f} minutes)"

        # Check file limit
        file_limit = self.config.get('auto_stop_after_files')
        if file_limit and self.stats['files_successful'] >= file_limit:
            return True, f"File limit reached ({self.stats['files_successful']} files)"

        return False, ""

    def _setup_pipeline(self):
        # Validate LLM providers for generation modes
        if self.config['mode'] in ['generate', 'full']:
            providers = get_available_providers()
            if not providers:
                print("⚠️ Warning: No LLM providers available for generation")

    def process_files(self, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multiple files with complete pipeline

        Args:
            file_paths: List of file paths to process
            args: Command line arguments

        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        self.stats['files_total'] = len(file_paths)

        # Initialize progress tracking
        initialize_progress(len(file_paths), self.config.get('show_progress', False))

        print(f"🚀 Starting processing pipeline...")
        print(f"   Mode: {self.config['mode']}")
        print(f"   Files: {len(file_paths)}")
        print(f"   Threads: {self.config.get('threads', 1)}")

        try:
            if self.config['mode'] == 'extract-only':
                results = self._process_extraction_only(file_paths, config)
            elif self.config['mode'] == 'generate':
                results = self._process_with_generation(file_paths, config)
            elif self.config['mode'] == 'full':
                results = self._process_full_pipeline(file_paths, config)
            elif self.config['mode'] == 'resume':
                results = self._process_resume(file_paths,config)
            elif self.config['mode'] == 'direct_to_llm':
                results = self._process_media_directly(file_paths,config)
            else:
                raise ValueError(f"Unknown processing mode: {self.config['mode']}")

            # Calculate final statistics
            total_time = time.time() - start_time
            self.stats['total_processing_time'] = total_time

            # Prepare final results
            final_results = {
                'success': True,
                'mode': self.config['mode'],
                'files_processed': self.stats['files_total'],
                'successful': self.stats['files_successful'],
                'failed': self.stats['files_failed'],
                'total_processing_time': total_time,
                'total_text_chars': self.stats['total_text_chars'],
                'total_images': self.stats['total_images'],
                'output_dir': self.config['output_dir'],
                'errors': self.stats['errors'],
                'config_used': self.config
            }

            # Show completion summary
            show_completion_summary(self.config)

            try:
                self.cleanup_after_processing()
            except Exception as e:
                print(f"⚠️ Cache cleanup warning: {e}")

            return final_results

        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return {
                'success': False,
                'error': str(e),
                'files_processed': self.stats['files_total'],
                'successful': self.stats['files_successful'],
                'failed': self.stats['files_failed']
            }

    def _process_extraction_only(self, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process files in extraction-only mode"""
        print("📄 Extraction-only mode: Processing documents...")

        if self.config.get('threads', 1) > 1:
            return self._process_files_parallel(file_paths, self._extract_single_file)
        else:
            return self._process_files_sequential(file_paths, self._extract_single_file)

    def _process_with_generation(self, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process files with LLM generation"""
        print("🤖 Generation mode: Extracting and generating training data...")

        if self.config.get('threads', 1) > 1:
            return self._process_files_parallel(file_paths, self._extract_and_generate_single_file)
        else:
            return self._process_files_sequential(file_paths, self._extract_and_generate_single_file)

    def _process_full_pipeline(self, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process files with full pipeline (extraction + generation + vision)"""
        print("🚀 Full pipeline: Complete processing with all features...")

        # Enable all generators and vision processing
        self.config['generators'] = ['conversations', 'embeddings', 'qa_pairs', 'summaries']
        self.config['include_vision'] = True

        if self.config.get('threads', 1) > 1:
            return self._process_files_parallel(file_paths, self._extract_and_generate_single_file)
        else:
            return self._process_files_sequential(file_paths, self._extract_and_generate_single_file)

    def _process_resume(self, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Resume processing from checkpoint"""
        print("🔄 Resume mode: Continuing from previous session...")

        # NEW: Check if we have extracted data to work with
        extracted_results = {}

        # Try to load from extraction cache or previous results
        for file_path in file_paths:
            try:
                # NEW: Add progress update for resume mode
                file_name = Path(file_path).name
                start_file_processing(file_name)

                processor = get_processor_for_file(file_path, self.config)

                # Check if we have cached extraction results
                cached_result = processor._load_from_cache(file_path)
                if cached_result:
                    extracted_results[file_path] = (cached_result['text'], cached_result['images'])
                    # NEW: Show progress for cached results
                    print(f"📌 Loaded from cache: {file_name}")
                else:
                    # If no cache, extract again
                    print(f"🔄 Re-extracting: {file_name}")
                    text, images = processor.extract_content(file_path, self.config.get('use_cache', True))
                    extracted_results[file_path] = (text, images)

                # NEW: Update progress
                text_chars = len(extracted_results[file_path][0])
                image_count = len(extracted_results[file_path][1])
                complete_file_processing(file_name, text_chars, image_count, 0.0, True, "Resume")

            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue

        # Store extracted results for output writer
        self._extracted_results = extracted_results

        # Now generate training data from extracted content
        return self._process_with_generation(file_paths)

    def _process_media_directly(self, file_paths:List[str], config: Dict[str, Any])->Dict[str,Any]:
        pass

    def _process_files_parallel(self, file_paths: List[str], process_func) -> Dict[str, Any]:
        """Process files in parallel using ThreadPoolExecutor"""
        max_workers = min(self.config.get('threads', 4), len(file_paths))
        self.start_time = time.time()  # NEW: Track start time
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_func, file_path): file_path
                for file_path in file_paths
            }

            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    self._update_stats(result)

                    # NEW: Check for auto-stop after each file
                    if self.stats['should_stop']:
                        print(f"\n⏸️  Auto-stopping: {self.stats['stop_reason']}")
                        self._save_checkpoint(file_paths, file_path)
                        # Cancel remaining futures
                        for remaining_future in future_to_file:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break

                    if result['success']:
                        self._save_file_result_immediately(file_path, result)
                        if self.config.get('save_per_file'):
                            self._save_per_file_result(file_path, result)
                    update_progress_display()

                except Exception as e:
                    self._handle_file_error(file_path, str(e))
                    # NEW: Check for auto-stop after errors too
                    if self.stats['should_stop']:
                        print(f"\n⏸️  Auto-stopping: {self.stats['stop_reason']}")
                        self._save_checkpoint(file_paths, file_path)
                        break
                    update_progress_display()

        return {'parallel_processing': True}

    def _process_files_sequential(self, file_paths: List[str], process_func) -> Dict[str, Any]:
        """Process files sequentially"""
        for file_path in file_paths:
            try:
                result = process_func(file_path)
                self._update_stats(result)

                # NEW: Save immediately after each file is processed
                if result['success']:
                    self._save_file_result_immediately(file_path, result)
                    if self.config.get('save_per_file'):
                        self._save_per_file_result(file_path, result)
                # Force progress display update
                update_progress_display()

            except Exception as e:
                self._handle_file_error(file_path, str(e))
                update_progress_display()

        return {'sequential_processing': True}

    def _extract_single_file(self, file_path: str) -> Dict[str, Any]:
        """Extract content from a single file"""
        file_name = Path(file_path).name
        start_time = time.time()
        print(f"Initiate the extraction process: {file_name}")
        if self.config.get("use_resource_limits", True):
            file_size = os.path.getsize(file_path)
            if not resource_manager.can_process_file(file_size):
                return {
                    'success': False,
                    'file_path': file_path,
                    'file_name': file_name,
                    'error': f"File too large to process safely ({file_size} bytes)"
                }
        try:
            # Get appropriate processor
            processor = get_processor_for_file(file_path, self.config)

            # Extract content
            text, images = processor.extract_content(file_path, self.config.get('use_cache', True))

            processing_time = time.time() - start_time

            result = {
                'success': True,
                'file_path': file_path,
                'file_name': file_name,
                'text': text,
                'images': images,
                'text_chars': len(text),
                'image_count': len(images),
                'processing_time': processing_time,
                'processor': processor.processor_name
            }

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'file_path': file_path,
                'file_name': file_name,
                'error': str(e),
                'processing_time': processing_time
            }

    def _extract_and_generate_single_file(self, file_path: str) -> Dict[str, Any]:
        """Extract content and generate training data for a single file"""
        # First extract content
        extract_result = self._extract_single_file(file_path)

        if not extract_result['success']:
            return extract_result

        # Then generate training data
        try:
            text = extract_result['text']
            images = extract_result['images']

            if not text.strip():
                # No text content to generate from
                extract_result['generated_data'] = {}
                return extract_result

            # Generate training data with custom prompts and async option
            generators = self.config.get('generators', ['conversations'])
            include_vision = self.config.get('include_vision', False)
            custom_prompts = self.config.get('custom_prompts', None)
            use_async = self.config.get('use_async', True)  # NEW: Async option

            generated_data = generate_training_data(
                text,
                generators=generators,
                images=images if include_vision else None,
                custom_prompts=custom_prompts,
                use_async=use_async  # NEW: Pass async option
            )

            extract_result['generated_data'] = generated_data
            extract_result['generation_successful'] = True

            return extract_result

        except Exception as e:
            extract_result['generation_error'] = str(e)
            extract_result['generation_successful'] = False
            return extract_result

    def _save_file_result_immediately(self, file_path: str, result: Dict[str, Any]):
        """NEW: Save file result immediately using OutputWriter"""
        try:
            # NEW: Use the existing OutputWriter for proper saving
            if result.get('text') and result.get('generated_data'):
                # Save both extraction and generation data
                file_results = {
                    file_path: result['generated_data']
                }
                self.output_writer.save_generated_data(file_results)
            elif result.get('text'):
                # Save only extraction data
                extraction_results = {
                    'mode': self.config['mode'],
                    'extracted_data': {file_path: (result['text'], result.get('images', []))},
                    'files_processed': 1,
                    'successful': 1,
                    'config_used': self.config
                }
                self.output_manager.save_all_results(extraction_results)

        except Exception as e:
            print(f"⚠️ Error saving immediate results for {Path(file_path).name}: {e}")

    def _update_stats(self, result: Dict[str, Any]):
        """Update processing statistics"""
        if result['success']:
            self.stats['files_successful'] += 1
            self.stats['total_text_chars'] += result.get('text_chars', 0)
            self.stats['total_images'] += result.get('image_count', 0)
            self.stats['consecutive_failures'] = 0  # NEW: Reset consecutive failures
        else:
            self.stats['files_failed'] += 1
            self.stats['consecutive_failures'] += 1  # NEW: Increment consecutive failures
            self.stats['errors'].append({
                'file': result.get('file_name', 'unknown'),
                'error': result.get('error', 'Unknown error')
            })

            # NEW: Check for quota exceeded error
            error_msg = result.get('error', '').lower()
            if ('quota' in error_msg or '429' in error_msg) and self.config.get('auto_stop_on_quota_exceeded', True):
                self.stats['should_stop'] = True
                self.stats['stop_reason'] = "API quota exceeded"

        # NEW: Check if we should auto-stop
        should_stop, reason = self._should_auto_stop()
        if should_stop:
            self.stats['should_stop'] = True
            self.stats['stop_reason'] = reason
        def _handle_file_error(self, file_path: str, error: str):
            """Handle file processing error"""
            file_name = Path(file_path).name
            self.stats['files_failed'] += 1
            self.stats['errors'].append({
                'file': file_name,
                'error': error
            })
            add_processing_error(file_name, error)

    def _save_per_file_result(self, file_path: str, result: Dict[str, Any]):
        """Save individual file result immediately (fault-tolerant)"""
        try:
            output_dir = Path(self.config['output_dir'])
            file_output_dir = output_dir / "per_file" / result['file_name']
            file_output_dir.mkdir(parents=True, exist_ok=True)

            # Save extracted text
            if result.get('text'):
                text_file = file_output_dir / "extracted_text.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])

            # Save generated training data
            if result.get('generated_data'):
                self.output_writer.save_per_file_data(file_output_dir, result['generated_data'])

            # Save metadata
            metadata = {
                'file_name': result['file_name'],
                'file_path': result['file_path'],
                'success': result['success'],
                'processing_time': result.get('processing_time', 0),
                'text_chars': result.get('text_chars', 0),
                'image_count': result.get('image_count', 0),
                'processor': result.get('processor', 'unknown'),
                'generation_successful': result.get('generation_successful', False),
                'processed_at': time.time()
            }

            metadata_file = file_output_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)

        except Exception as e:
            print(f"⚠️ Error saving per-file result for {result['file_name']}: {e}")

    def _load_resume_state(self) -> List[str]:
        """Load resume state from checkpoint file"""
        try:
            resume_file = Path(self.config['resume_from'])
            if resume_file.exists():
                import json
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                return resume_data.get('processed_files', [])
        except Exception as e:
            print(f"⚠️ Error loading resume state: {e}")

        return []

    def _save_resume_state(self, processed_files: List[str]):
        """Save resume state to checkpoint file"""
        try:
            resume_file = Path(self.config['output_dir']) / "resume_checkpoint.json"
            resume_data = {
                'processed_files': processed_files,
                'saved_at': time.time(),
                'config': self.config
            }

            with open(resume_file, 'w') as f:
                import json
                json.dump(resume_data, f, indent=2, default=str)

        except Exception as e:
            print(f"⚠️ Error saving resume state: {e}")

    def cleanup_after_processing(self):
        """Clean up resources after processing completion"""
        if self.config.get('clear_cache_after_run', False):
            try:
                result = self.perform_cache_cleanup(config=self.config, force_clear_if_needed=True)
                print("🧹 Cache cleanup completed")
            except Exception as e:
                print(f"❌ Cache cleanup error: {e}")
                import traceback
                traceback.print_exc()

    def perform_cache_cleanup(config: Optional[Dict[str, Any]] = None, force_clear_if_needed: bool = False) -> Dict[str, Any]:
        """Performs cache cleanup and optionally clears if cleanup didn't fully work."""
        from utils.cache import get_cache_stats, cleanup_cache, clear_cache

        print("🧹 Cleaning up cache...")

        max_size_gb = config.get('max_cache_size', 5.0) if config else 5.0
        max_age_days = config.get('max_cache_age', 30) if config else 30

        before_stats = get_cache_stats()
        print(f"🔍 Before cleanup: {before_stats.get('cache_entries', 0)} entries, {before_stats.get('total_size_mb', 0):.1f} MB")

        cleanup_cache(max_size_gb, max_age_days)

        after_stats = get_cache_stats()
        print(f"🔍 After cleanup: {after_stats.get('cache_entries', 0)} entries, {after_stats.get('total_size_mb', 0):.1f} MB")

        if force_clear_if_needed and after_stats.get('cache_entries', 0) > 0:
            print("🔧 Cleanup didn't work, trying force clear...")
            clear_cache()
            final_stats = get_cache_stats()
            print(f"🔍 After force clear: {final_stats.get('cache_entries', 0)} entries")

        return {
            'success': True,
            'command': 'cache_cleanup'
        }



    def _save_checkpoint(self, all_files: List[str], current_file: str):
        """Save checkpoint for resuming later"""
        try:
            processed_files = []
            remaining_files = []
            found_current = False

            for file_path in all_files:
                if file_path == current_file:
                    found_current = True
                    remaining_files.append(file_path)  # Include current file for retry
                elif found_current:
                    remaining_files.append(file_path)
                else:
                    processed_files.append(file_path)

            checkpoint_data = {
                'timestamp': time.time(),
                'stop_reason': self.stats['stop_reason'],
                'processed_files': processed_files,
                'remaining_files': remaining_files,
                'stats': self.stats,
                'config': self.config
            }

            checkpoint_file = Path(self.config['output_dir']) / 'checkpoint.json'
            with open(checkpoint_file, 'w') as f:
                import json
                json.dump(checkpoint_data, f, indent=2, default=str)

            print(f"💾 Checkpoint saved: {checkpoint_file}")
            print(f"📊 Progress: {len(processed_files)}/{len(all_files)} files completed")
            print(f"🔄 To continue: python main.py --resume-from {checkpoint_file}")

        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")

class PerformanceBenchmark(BaseProcessor):
    """Performance benchmarking for the processing pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmark_results = {}


    def process_files(self, file_paths: List[str], args=None) -> Dict[str, Any]:
        return self.run_benchmark(file_paths)

    def run_benchmark(self, file_paths: List[str]) -> Dict[str, Any]:
        """Run performance benchmark on file processing"""
        print("📊 Running performance benchmark...")

        # Test different thread counts
        thread_counts = [1, 2, 4, 8] if len(file_paths) > 4 else [1, 2]
        benchmark_results = {}

        for thread_count in thread_counts:
            if thread_count > len(file_paths):
                continue

            print(f"\n🧵 Testing with {thread_count} threads...")

            # Create config for this test
            test_config = self.config.copy()
            test_config['threads'] = thread_count
            test_config['show_progress'] = False

            # Run test
            start_time = time.time()
            pipeline = ProcessingPipeline(test_config)
            results = pipeline.process_files(file_paths[:6], None)
            end_time = time.time()

            # Record results
            benchmark_results[f'{thread_count}_threads'] = {
                'processing_time': end_time - start_time,
                'files_processed': results.get('files_processed', 0),
                'successful': results.get('successful', 0),
                'failed': results.get('failed', 0),
                'throughput': results.get('files_processed', 0) / (end_time - start_time) if end_time > start_time else 0
            }

        # Calculate speedups
        if '1_threads' in benchmark_results:
            baseline_time = benchmark_results['1_threads']['processing_time']
            for key, result in benchmark_results.items():
                if key != '1_threads':
                    speedup = baseline_time / result['processing_time'] if result['processing_time'] > 0 else 1
                    result['speedup'] = round(speedup, 2)

        # Display results
        self._display_benchmark_results(benchmark_results)
        return benchmark_results

    def _display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark results"""
        print(f"\n📈 Benchmark Results:")
        print(f"{'Threads':<8} {'Time (s)':<10} {'Files/s':<10} {'Speedup':<10}")
        print("-" * 40)

        for key, result in results.items():
            threads = key.split('_')[0]
            time_taken = result['processing_time']
            throughput = result['throughput']
            speedup = result.get('speedup', 1.0)
            print(f"{threads:<8} {time_taken:<10.2f} {throughput:<10.2f} {speedup:<10}")


class BatchProcessor(BaseProcessor):
    """Batch processor for handling large numbers of files efficiently"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get('batch_size', 10)

    def process_in_batches(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process files in batches to manage memory usage"""
        total_files = len(file_paths)
        total_results = {
            'files_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_text_chars': 0,
            'total_images': 0,
            'errors': []
        }

        # Split into batches
        batches = [file_paths[i:i + self.batch_size]
                  for i in range(0, len(file_paths), self.batch_size)]

        print(f"📦 Processing {total_files} files in {len(batches)} batches of {self.batch_size}")

        for batch_num, batch_files in enumerate(batches, 1):
            print(f"\n🔄 Processing batch {batch_num}/{len(batches)} ({len(batch_files)} files)...")

            # Process batch
            pipeline = ProcessingPipeline(self.config)
            batch_results = pipeline.process_files(batch_files, None)

            # Accumulate results
            total_results['files_processed'] += batch_results.get('files_processed', 0)
            total_results['successful'] += batch_results.get('successful', 0)
            total_results['failed'] += batch_results.get('failed', 0)
            total_results['total_text_chars'] += batch_results.get('total_text_chars', 0)
            total_results['total_images'] += batch_results.get('total_images', 0)
            total_results['errors'].extend(batch_results.get('errors', []))

            # Memory cleanup between batches
            import gc
            gc.collect()

        return total_results


# Factory function for creating appropriate processor
def create_processing_pipeline(config: Dict[str, Any]) -> ProcessingPipeline:
    """Create appropriate processing pipeline based on configuration"""
    if config.get('benchmark'):
        return PerformanceBenchmark(config)
    elif config.get('batch_size') and config.get('batch_size') < 50:
        return BatchProcessor(config)
    else:
        return ProcessingPipeline(config)
