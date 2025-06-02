# core/pipeline.py
"""
Complete processing pipeline for Doc2Train v2.0 Enhanced
Orchestrates the entire document processing workflow with parallel execution
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

from processors.base_processor import get_processor_for_file, discover_plugins
from core.generator import generate_training_data
from core.llm_client import get_available_providers
from utils.progress import (
    initialize_progress, start_file_processing, complete_file_processing,
    add_processing_error, update_progress_display, show_completion_summary
)
from utils.validation import validate_input_enhanced
from outputs.writers import OutputWriter

class ProcessingPipeline:
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
            'total_text_chars': 0,
            'total_images': 0,
            'total_processing_time': 0.0,
            'errors': []
        }
        self.output_writer = OutputWriter(config)
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Setup pipeline components"""
        # Discover plugins if plugin directory specified
        if self.config.get('plugin_dir'):
            discover_plugins(self.config['plugin_dir'])

        # Validate LLM providers for generation modes
        if self.config['mode'] in ['generate', 'full']:
            providers = get_available_providers()
            if not providers:
                print("⚠️ Warning: No LLM providers available for generation")

    def process_files(self, file_paths: List[str], args) -> Dict[str, Any]:
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

        print(f"🚀 Starting enhanced processing pipeline...")
        print(f"   Mode: {self.config['mode']}")
        print(f"   Files: {len(file_paths)}")
        print(f"   Threads: {self.config.get('threads', 1)}")

        try:
            if self.config['mode'] == 'extract-only':
                results = self._process_extraction_only(file_paths)
            elif self.config['mode'] == 'generate':
                results = self._process_with_generation(file_paths)
            elif self.config['mode'] == 'full':
                results = self._process_full_pipeline(file_paths)
            elif self.config['mode'] == 'resume':
                results = self._process_resume(file_paths)
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


            # Add this before calculating final_results
            if self.stats['files_successful'] > 0:
                try:
                    # Save extraction results if we have any
                    if hasattr(self, '_extracted_results'):
                        self.output_writer.save_extraction_results({
                            'mode': self.config['mode'],
                            'extracted_data': self._extracted_results,
                            'files_processed': self.stats['files_total'],
                            'successful': self.stats['files_successful'],
                            'config_used': self.config
                        })
                    print(f"💾 Results saved to: {self.config['output_dir']}")
                except Exception as e:
                    print(f"⚠️ Error saving results: {e}")
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

    def _process_extraction_only(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process files in extraction-only mode"""
        print("📄 Extraction-only mode: Processing documents...")

        if self.config.get('threads', 1) > 1:
            return self._process_files_parallel(file_paths, self._extract_single_file)
        else:
            return self._process_files_sequential(file_paths, self._extract_single_file)

    def _process_with_generation(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process files with LLM generation"""
        print("🤖 Generation mode: Extracting and generating training data...")

        if self.config.get('threads', 1) > 1:
            return self._process_files_parallel(file_paths, self._extract_and_generate_single_file)
        else:
            return self._process_files_sequential(file_paths, self._extract_and_generate_single_file)

    def _process_full_pipeline(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process files with full pipeline (extraction + generation + vision)"""
        print("🚀 Full pipeline: Complete processing with all features...")

        # Enable all generators and vision processing
        self.config['generators'] = ['conversations', 'embeddings', 'qa_pairs', 'summaries']
        self.config['include_vision'] = True

        if self.config.get('threads', 1) > 1:
            return self._process_files_parallel(file_paths, self._extract_and_generate_single_file)
        else:
            return self._process_files_sequential(file_paths, self._extract_and_generate_single_file)

    def _process_resume(self, file_paths: List[str]) -> Dict[str, Any]:
        """Resume processing from checkpoint"""
        print("🔄 Resume mode: Continuing from previous session...")

        # Load previous state if resume file exists
        if self.config.get('resume_from'):
            processed_files = self._load_resume_state()
            remaining_files = [f for f in file_paths if f not in processed_files]
            print(f"📄 Resuming: {len(remaining_files)} files remaining")
            file_paths = remaining_files

        # Process remaining files
        return self._process_with_generation(file_paths)

    def _process_files_parallel(self, file_paths: List[str], process_func) -> Dict[str, Any]:
        """Process files in parallel using ThreadPoolExecutor"""
        max_workers = min(self.config.get('threads', 4), len(file_paths))

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

                    # Save per-file if requested
                    if self.config.get('save_per_file') and result['success']:
                        self._save_per_file_result(file_path, result)

                    # NEW: Force progress display update
                    from utils.progress import update_progress_display
                    update_progress_display()

                except Exception as e:
                    self._handle_file_error(file_path, str(e))
                    # NEW: Update progress even on error
                    from utils.progress import update_progress_display
                    update_progress_display()

        return {'parallel_processing': True}

    def _process_files_sequential(self, file_paths: List[str], process_func) -> Dict[str, Any]:
        """Process files sequentially"""
        for file_path in file_paths:
            try:
                result = process_func(file_path)
                self._update_stats(result)

                # Save per-file if requested
                if self.config.get('save_per_file') and result['success']:
                    self._save_per_file_result(file_path, result)

                # NEW: Force progress display update
                from utils.progress import update_progress_display
                update_progress_display()

            except Exception as e:
                self._handle_file_error(file_path, str(e))
                # NEW: Update progress even on error
                from utils.progress import update_progress_display
                update_progress_display()

        return {'sequential_processing': True}

    def _extract_single_file(self, file_path: str) -> Dict[str, Any]:
        """Extract content from a single file"""
        file_name = Path(file_path).name
        start_time = time.time()

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

            if not hasattr(self, '_extracted_results'):
                self._extracted_results = {}
            if result['success']:
                self._extracted_results[file_path] = (result['text'], result['images'])
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

            # Generate training data
            generators = self.config.get('generators', ['conversations'])
            include_vision = self.config.get('include_vision', False)

            generated_data = generate_training_data(
                text,
                generators=generators,
                images=images if include_vision else None
            )

            extract_result['generated_data'] = generated_data
            extract_result['generation_successful'] = True

            return extract_result

        except Exception as e:
            extract_result['generation_error'] = str(e)
            extract_result['generation_successful'] = False
            return extract_result

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

    def _update_stats(self, result: Dict[str, Any]):
        """Update processing statistics"""
        if result['success']:
            self.stats['files_successful'] += 1
            self.stats['total_text_chars'] += result.get('text_chars', 0)
            self.stats['total_images'] += result.get('image_count', 0)
        else:
            self.stats['files_failed'] += 1
            self.stats['errors'].append({
                'file': result['file_name'],
                'error': result.get('error', 'Unknown error')
            })

    def _handle_file_error(self, file_path: str, error: str):
        """Handle file processing error"""
        file_name = Path(file_path).name
        self.stats['files_failed'] += 1
        self.stats['errors'].append({
            'file': file_name,
            'error': error
        })
        add_processing_error(file_name, error)

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
                from utils.cache import get_cache_stats, cleanup_cache

                # Show before stats
                before_stats = get_cache_stats()
                print(f"🔍 Before cleanup: {before_stats.get('cache_entries', 0)} entries, {before_stats.get('total_size_mb', 0):.1f} MB")

                # Try cleanup
                cleanup_cache()

                # Show after stats
                after_stats = get_cache_stats()
                print(f"🔍 After cleanup: {after_stats.get('cache_entries', 0)} entries, {after_stats.get('total_size_mb', 0):.1f} MB")

                # If still has entries, force clear
                if after_stats.get('cache_entries', 0) > 0:
                    print("🔧 Cleanup didn't work, trying force clear...")
                    from utils.cache import clear_cache
                    clear_cache()

                    final_stats = get_cache_stats()
                    print(f"🔍 After force clear: {final_stats.get('cache_entries', 0)} entries")

                print("🧹 Cache cleanup completed")

            except Exception as e:
                print(f"❌ Cache cleanup error: {e}")
                import traceback
                traceback.print_exc()

class PerformanceBenchmark:
    """Performance benchmarking for the processing pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmark_results = {}

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


class BatchProcessor:
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
