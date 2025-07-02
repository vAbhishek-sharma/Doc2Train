# core/pipeline.py
"""
Complete processing pipeline for Doc2Train v2.0 Enhanced
Orchestrates the entire document processing workflow with parallel execution
"""

import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from doc2train.core.registries.formatter_registry import get_formatter
from doc2train.core.registries.generator_registry import get_generator
from doc2train.core.writers import OutputManager
from doc2train.utils.resource_manager import resource_manager
import os
import ipdb

from doc2train.core.registries.processor_registry import get_processor_for_file
from doc2train.core.generator import generate_data
from doc2train.core.llm_client import get_available_providers, process_media_directly
from doc2train.utils.progress import (
    initialize_progress, start_file_processing, complete_file_processing,
    add_processing_error, update_progress_display, show_completion_summary
)
from doc2train.utils.validation import validate_input_and_files
from doc2train.utils.process import ProcessManager
from doc2train.core.registries.formatter_registry import get_formatter

class BaseProcessor:
    def process_files(self, file_paths: List[str], args=None) -> Dict[str, Any]:
        raise NotImplementedError

    def cleanup_after_processing(self):
        """Clean up resources after processing completion"""
        if self.config.get('clear_cache_after_run', False):
            try:
                # Use instance config
                ProcessingPipeline.perform_cache_cleanup(config=self.config, force_clear_if_needed=True)
                print("🧹 Cache cleanup completed")
            except Exception as e:
                print(f"❌ Cache cleanup error: {e}")
                import traceback
                traceback.print_exc()


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
            'consecutive_failures': 0,  #  Track consecutive failures
            'total_text_chars': 0,
            'total_images': 0,
            'total_processing_time': 0.0,
            'errors': [],
            'should_stop': False,  #  Auto-stop flag
            'stop_reason': ''  #  Reason for stopping
        }
        self.start_time = None  #  Track processing start time
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

    def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
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

                results = self._process_extraction_only(file_paths, self.config)
            elif self.config['mode'] == 'generate':

                results = self._process_with_generation(file_paths, self.config)
            elif self.config['mode'] == 'full':
                results = self._process_full_pipeline(file_paths,  self.config)
            elif self.config['mode'] == 'resume':
                results = self._process_resume(file_paths, self.config)
            elif self.config['mode'] == 'direct_to_llm':
                results = self._process_media_directly(file_paths, self.config)
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
        """Resume processing from checkpoint, continuing in the same mode you started."""
        print("🔄 Resume mode: Continuing from previous session...")
        mode = config.get('mode', 'extract-only')

        # 1) If we need extraction (extract-only, generate or full), reload or re-extract
        extracted_results: Dict[str, Tuple[str, List[Any]]] = {}
        if mode in ('extract-only', 'generate', 'full'):
            for file_path in file_paths:
                file_name = Path(file_path).name
                start_file_processing(file_name)
                try:
                    proc = get_processor_for_file(file_path, config)
                    cached = proc._load_from_cache(file_path)
                    if cached:
                        text, images = cached['text'], cached.get('images', [])
                        print(f"📌 Loaded from cache: {file_name}")
                    else:
                        print(f"🔄 Re-extracting: {file_name}")
                        text, images = proc.extract_content(file_path, config.get('use_cache', True))
                    extracted_results[file_path] = (text, images)
                    complete_file_processing(file_name, len(text), len(images), 0.0, True, "Resume")
                except Exception as e:
                    print(f"⚠️ Error resuming {file_name}: {e}")
            # stash it for downstream stages
            self._extracted_results = extracted_results

        # 2) Dispatch based on mode
        if mode == 'extract-only':
            # Build a summary of all extracted files and return
            summary = {
                'mode': mode,
                'files_total': len(extracted_results),
                'files_successful': 0,
                'files_failed': 0,
                'files': {}
            }
            for fp, (txt, imgs) in extracted_results.items():
                if txt is not None:
                    summary['files'][fp] = {
                        'text_chars': len(txt),
                        'image_count': len(imgs)
                    }
                    summary['files_successful'] += 1
                else:
                    summary['files_failed'] += 1
            self.output_manager.save_all_results(summary)
            return summary

        elif mode in ('generate', 'full'):
            # resume the extract+generate pipeline
            return self._process_with_generation(file_paths, config)

        elif mode == 'direct_to_llm':
            # resume the media‐direct pipeline
            return self._process_media_directly(file_paths, config)

        else:
            raise ValueError(f"Unknown resume mode: {mode!r}")

    def _process_media_directly(self, file_paths:List[str], config: Dict[str, Any])->Dict[str,Any]:
        pass

    def _process_files_parallel(self, file_paths: List[str], process_func) -> Dict[str, Any]:
        """Process files in parallel using ThreadPoolExecutor and write one combined summary."""
        max_workers = min(self.config.get('threads', 4), len(file_paths))
        self.start_time = time.time()

        # 1. Initialize combined summary
        all_results: Dict[str, Any] = {
            'mode': self.config.get('mode'),
            'files': {},
            'files_total': len(file_paths),
            'files_successful': 0,
            'files_failed': 0,
            'errors': [],
            'total_text_chars': 0,
            'total_images': 0,
        }

        # 2. Dispatch work
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_func, fp): fp
                for fp in file_paths
            }

            # 3. As each file finishes…
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    self._update_stats(result)

                    # 4. Accumulate (runs in main thread; no lock needed)
                    if result.get('success'):
                        all_results['files_successful'] += 1
                        all_results['files'][file_path] = {
                            'text_chars': result.get('text_chars', 0),
                            'image_count': result.get('image_count', 0),
                        }
                    else:
                        all_results['files_failed'] += 1
                        all_results['errors'].append({
                            'file': file_path,
                            'error': result.get('error')
                        })
                    all_results['total_text_chars'] += result.get('text_chars', 0)
                    all_results['total_images'] += result.get('image_count', 0)

                    # 5. (Optional) per-file write
                    if result.get('success') and self.config.get('save_per_file'):
                        self._save_per_file_result(file_path, result)

                    update_progress_display()

                    # 6. Auto-stop checkpointing
                    if self.stats['should_stop']:
                        print(f"\n⏸️  Auto-stopping: {self.stats['stop_reason']}")
                        self._save_checkpoint(file_paths, file_path)
                        for pending in future_to_file:
                            if not pending.done():
                                pending.cancel()
                        break

                except Exception as e:
                    self._handle_file_error(file_path, str(e))
                    update_progress_display()
                    if self.stats['should_stop']:
                        print(f"\n⏸️  Auto-stopping: {self.stats['stop_reason']}")
                        self._save_checkpoint(file_paths, file_path)
                        break
        # 7. One-time summary dump
        self.output_manager.save_all_results(all_results, 'batch_summary')
        return all_results

    def _process_files_sequential(self, file_paths: List[str], process_func) -> Dict[str, Any]:
        """Process files one by one, then write one combined summary."""
        # 1. Initialize combined summary
        all_results: Dict[str, Any] = {
            'mode': self.config.get('mode'),
            'files': {},
            'files_total': len(file_paths),
            'files_successful': 0,
            'files_failed': 0,
            'errors': [],
            'total_text_chars': 0,
            'total_images': 0,
        }

        # 2. Loop through each file
        for file_path in file_paths:
            try:
                result = process_func(file_path)
                self._update_stats(result)

                # 3. Accumulate
                if result.get('success'):
                    all_results['files_successful'] += 1
                    all_results['files'][file_path] = {
                        'text_chars': result.get('text_chars', 0),
                        'image_count': result.get('image_count', 0),
                    }
                else:
                    all_results['files_failed'] += 1
                    all_results['errors'].append({
                        'file': file_path,
                        'error': result.get('error')
                    })
                all_results['total_text_chars'] += result.get('text_chars', 0)
                all_results['total_images'] += result.get('image_count', 0)

                # 4. (Optional) per-file write
                if result.get('success'):
                    if self.config.get('save_per_file'):
                        self._save_per_file_result(file_path, result)

                update_progress_display()

            except Exception as e:
                self._handle_file_error(file_path, str(e))
                update_progress_display()

        # 5. One-time summary dump
        self.output_manager.save_all_results(all_results, 'batch_summary')
        return all_results

    def _extract_single_file(self, file_path: str) -> Dict[str, Any]:
        """Extract content from a single file"""
        file_name = Path(file_path).name
        start_time = time.time()
        print(f"Initiate the extraction process: {file_name}")
        if self.config.get("use_resource_limits", True):
            file_size = os.path.getsize(file_path)
            if not resource_manager.can_process_file(file_size):
                #To update Show size in MB using predefined method for the same
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
        """Extract content and generate training data for a single file, but only keep metadata in memory."""
        # 1. Extract
        extract_result = self._extract_single_file(file_path)
        if not extract_result.get('success'):
            return extract_result

        # Prepare default generation metadata
        extract_result['generation_successful'] = False
        extract_result['generated_count']       = 0
        extract_result['generated_path']        = None

        try:
            text   = extract_result['text']
            images = extract_result['images']

            if not text.strip():
                # nothing to generate
                return extract_result
            generated = generate_data(text,images,self.config)
            # 3. Persist the full output to disk
            for fmt_name in self.config.get('text_formatters', []):
                fmt_cls = get_formatter(fmt_name)
                if not fmt_cls:
                    continue
                formatter = fmt_cls(self.config)
                # assume write() method handles file naming
                formatter.write(generated, batch_idx=1)
                # if formatter exposes last output path, record it
                if hasattr(formatter, 'last_output_path'):
                    extract_result['generated_paths'].append(formatter.last_output_path)


            # 4. Update only the metadata in your result
            extract_result['generation_successful'] = True
            extract_result['generated_count']       = len(generated)

            # 5. (Optional) drop the big blob if it snuck in
            extract_result.pop('generated_data', None)

            return extract_result

        except Exception as e:
            extract_result['generation_error']      = str(e)
            extract_result['generation_successful'] = False
            return extract_result

    def _process_media_directly(self, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process media files (images) end-to-end through vision LLM + generators + formatters."""
        print("🖼️ Direct-to-LLM mode: Processing media files…")
        media_cfg = self.config['dataset']['media']
        gens = media_cfg.get('generators', [])
        fmts = media_cfg.get('formatters', [])

        # bind our per-file work
        def work(path: str) -> Dict[str, Any]:
            return self._process_media_single_file(path, gens, fmts)

        if self.config.get('threads', 1) > 1:
            return self._process_files_parallel(file_paths, work)
        else:
            return self._process_files_sequential(file_paths, work)

    def _process_media_single_file(
        self,
        media_path: str,
        generators: List[str],
        formatters: List[str]
    ) -> Dict[str, Any]:
        """
        1) Send the image to the vision LLM
        2) Run each generator plugin on the LLM’s raw output
        3) Format the combined generator results
        4) Write to disk and return a tiny metadata dict
        """
        result: Dict[str, Any] = {'success': False, 'media_path': media_path}
        try:
            # ——— 1) Vision LLM call ———
            raw = process_media_directly(
                media_path,
                prompt=self.config.get('media_prompt')
            )

            # try to parse JSON if the model returned structured JSON
            try:
                raw_data = json.loads(raw)
            except Exception:
                raw_data = raw

            # ——— 2) Generators ———
            gen_outputs: Dict[str, Any] = {}
            for gen in generators:
                gen_cls = get_generator(gen)
                if not gen_cls:
                    print(f"⚠️  No generator plugin for: {gen}")
                    continue
                plugin = gen_cls(self.config)
                # assume each plugin.generate takes (input, type, prompt_template)
                gen_outputs[gen] = plugin.generate(
                    raw_data,
                    gen,
                    prompt_template=self.config.get('prompts', {}).get('custom', {}).get(gen)
                )

            # ——— 3) Formatters ———
            formatted = gen_outputs
            for fmt in formatters:
                fmt_cls = get_formatter(fmt)
                if not fmt_cls:
                    print(f"⚠️  No formatter plugin for: {fmt}")
                    continue
                formatted = fmt_cls(self.config).format(formatted, data_type=fmt)

            # ——— 4) Persist ———
            out_dir = Path(self.config['output_dir']) / "media_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = formatters[0]
            out_file = out_dir / f"{Path(media_path).stem}_media.{ext}"

            if isinstance(formatted, str):
                out_file.write_text(formatted, encoding='utf-8')
            else:
                out_file.write_text(json.dumps(formatted, ensure_ascii=False, indent=2), encoding='utf-8')

            result.update({
                'success': True,
                'generators_used': generators,
                'formatters_used': formatters,
                'output_file': str(out_file)
            })

        except Exception as e:
            result['error'] = str(e)

        return result

    def _update_stats(self, result: Dict[str, Any]):
        """Update processing statistics"""
        if result['success']:
            self.stats['files_successful'] += 1
            self.stats['total_text_chars'] += result.get('text_chars', 0)
            self.stats['total_images'] += result.get('image_count', 0)
            self.stats['consecutive_failures'] = 0  #  Reset consecutive failures
        else:
            self.stats['files_failed'] += 1
            self.stats['consecutive_failures'] += 1  #  Increment consecutive failures
            self.stats['errors'].append({
                'file': result.get('file_name', 'unknown'),
                'error': result.get('error', 'Unknown error')
            })

            #  Check for quota exceeded error
            error_msg = result.get('error', '').lower()
            if ('quota' in error_msg or '429' in error_msg) and self.config.get('auto_stop_on_quota_exceeded', True):
                self.stats['should_stop'] = True
                self.stats['stop_reason'] = "API quota exceeded"

        #  Check if we should auto-stop
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

    @staticmethod
    def perform_cache_cleanup(config: Optional[Dict[str, Any]] = None, force_clear_if_needed: bool = False) -> Dict[str, Any]:
        """Performs cache cleanup and optionally clears if cleanup didn't fully work."""
        from doc2train.utils.cache import get_cache_stats, cleanup_cache, clear_cache

        print("🧹 Cleaning up cache...")

        # Use config if provided, else default values
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

    def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
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
            batch_results = pipeline.process_files(batch_files)

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
def create_processing_pipeline(config: Dict[str, Any]) -> BaseProcessor:
    """Create appropriate processing pipeline based on configuration"""
    if config.get('benchmark'):
        return PerformanceBenchmark(config)
    elif config.get('batch_size') and config.get('batch_size') < 50:
        return BatchProcessor(config)
    else:
        return ProcessingPipeline(config)
