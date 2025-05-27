import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from file_processor import process_file
from config import MODELS
import multiprocessing
import pdb
def select_model():
    """Select AI provider and model with defaults"""
    provider = os.getenv('DEFAULT_PROVIDER', 'openai')
    model = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')

    if provider not in MODELS:
        print(f"Invalid provider: {provider}. Using default 'openai'.")
        provider = 'openai'

    if model not in MODELS[provider]:
        print(f"Invalid model: {model}. Using default 'gpt-4o-mini'.")
        model = 'gpt-4o-mini'

    return provider, model

def get_files(input_path):
    """Returns a list of files to process"""
    if os.path.isdir(input_path):  # If it's a folder
        return glob.glob(os.path.join(input_path, "*.[pP][dD][fF]")) + \
               glob.glob(os.path.join(input_path, "*.[tT][xX][tT]")) + \
               glob.glob(os.path.join(input_path, "*.[eE][pP][uU][bB]")) + \
               glob.glob(os.path.join(input_path, "*.[sS][rR][tT]")) + \
               glob.glob(os.path.join(input_path, "*.[vV][tT][tT]"))
    elif os.path.isfile(input_path):  # If it's a single file
        return [input_path]
    else:
        print(f"Invalid file or folder: {input_path}")
        return []

def process_file_wrapper(args):
    """Wrapper function for process_file to handle multiple arguments"""
    file, provider, model, start_page, end_page, continue_process, use_ocr = args
    try:
        print(f"Started processing: {file}")
        process_file(file, provider, model, start_page, end_page, continue_process, use_ocr)
        print(f"Completed processing: {file}")
        return (file, True, None)
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return (file, False, str(e))

def parallel_process_files(files_to_process, provider, model, start_page, end_page,continue_process, max_workers=None, use_ocr=False):
    """Process multiple files in parallel"""
    if max_workers is None:
        # Use CPU count as default, but cap it at 4 to prevent API rate limits
        max_workers = min(4, multiprocessing.cpu_count())

    results = []
    args_list = [(file, provider, model, start_page, end_page, continue_process, use_ocr)
                 for file in files_to_process]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file_wrapper, args): args[0]
                         for args in args_list}

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                results.append((file, False, str(e)))

    return results

def main():
    input_path = input("\nEnter file(s) or folder path (PDF, TXT, EPUB, SRT, VTT) [default: sample.pdf]: ").strip() or "sample.pdf"
    start_page = input("Enter start page (default: 1): ").strip() or "1"
    end_page = input("Enter end page (default: full book): ").strip() or None
    max_workers = input("Enter maximum number of files to process simultaneously (default: CPU count, max 4): ").strip()
    use_ocr = input("Enable OCR for scanned PDFs? (y/N): ").strip().lower() in ['y', 'yes', '1']

    # Convert max_workers to int if provided, else None
    max_workers = int(max_workers) if max_workers else None

    # Convert input directly to boolean
    continue_process = input("Continue stalled process? (y/N): ").strip().lower() in ['y', 'yes', '1']

    provider, model = select_model()

    # Support for multiple files separated by commas
    paths = [path.strip() for path in input_path.split(",")]
    files_to_process = []
    for path in paths:
        files_to_process.extend(get_files(path))

    if not files_to_process:
        print("No valid files found. Exiting.")
        return

    print(f"\nFound {len(files_to_process)} files to process")
    print(f"Using {max_workers if max_workers else min(4, multiprocessing.cpu_count())} workers for parallel processing")

    # Process files in parallel
    results = parallel_process_files(
        files_to_process,
        provider,
        model,
        int(start_page),
        int(end_page) if end_page else None,
        continue_process,
        max_workers,
        use_ocr
    )

    # Print summary
    print("\nProcessing Summary:")
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\nSuccessfully processed {len(successful)} files:")
    for file, _, _ in successful:
        print(f"✓ {file}")

    if failed:
        print(f"\nFailed to process {len(failed)} files:")
        for file, _, error in failed:
            print(f"✗ {file} - Error: {error}")

if __name__ == "__main__":
    main()
