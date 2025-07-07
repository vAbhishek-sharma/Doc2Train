# outputs/writers.py

"""
Output Writer system for Doc2Train
- Plugin-based writers (filesystem, jsonl, print, etc.)
- Supports both explicit and smart selection
- Used by pipeline for saving generated and per-file results
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from doc2train.core.registries.writer_registry import get_writer
from doc2train.core.formatters import format_data, smart_format_data
import ipdb
def save_items(
    items: List[Dict],
    output_file: Union[str, Path],
    data_type: str,
    writer_name: str = "filesystem",
    config: Optional[dict] = None,
):
    """
    Save items using the specified writer plugin.
    """
    writer_cls = get_writer(writer_name)
    if not writer_cls:
        raise ValueError(f"Writer '{writer_name}' not found in registry.")
    writer = writer_cls(config)
    writer.write(output_file, items, data_type)

def smart_save_items(
    items: List[Dict],
    output_file: Union[str, Path],
    data_type: str,
    config: Optional[dict] = None,
):
    """
    Smart save: chooses writer plugin by config, file ext, or data type.
    """
    writer_name = None

    # 1. Config override
    if config and config.get("output_writer"):
        writer_name = config["output_writer"]

    # 2. Guess by file extension if not set
    if not writer_name:
        ext = Path(output_file).suffix.lower().lstrip('.')
        if get_writer(ext):
            writer_name = ext

    # 3. Guess by data_type if not set (optional)
    if not writer_name and data_type and get_writer(data_type):
        writer_name = data_type

    # 4. Fallback to filesystem
    if not writer_name:
        writer_name = "filesystem"

    writer_cls = get_writer(writer_name)
    if not writer_cls:
        raise ValueError(f"No writer found for: {writer_name}")
    writer = writer_cls(config)
    writer.write(output_file, items, data_type)

class OutputWriter:
    """
    Main entrypoint for saving results (used in pipeline).
    """
    def __init__(self, config=None):
        self.config = config or {}

    def save_generated_data(self, file_results: Dict[str, Dict]):
        """
        Save generated data from one or more files.
        Expects: {file_path: {data_type: items}}
        """
        output_dir = Path(self.config.get('output_dir', './outputs'))

        output_dir.mkdir(parents=True, exist_ok=True)
        for file_path, data_dict in file_results.items():
            file_stem = Path(file_path).stem
            for data_type, items in data_dict.items():
                # Format the data (using format_data, smart_format_data, etc.)
                format_name = self.config.get("output_format", "jsonl")
                formatted = format_data(items, data_type, format_name, self.config)
                ext = "." + format_name if not format_name.startswith('.') else format_name
                output_file = output_dir / f"{file_stem}_{data_type}{ext}"
                # Use smart writer
                smart_save_items(items, output_file, data_type, self.config)
                # Optionally also save formatted text (for easy reading)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(formatted)

    def save_per_file_data(self, file_output_dir: Path, data: Dict[str, List[Dict]]):
        """
        Save per-file generated data (each data_type to its own file).
        """
        file_output_dir = Path(file_output_dir)
        file_output_dir.mkdir(parents=True, exist_ok=True)
        for data_type, items in data.items():
            format_name = self.config.get("output_format", "jsonl")
            formatted = format_data(items, data_type, format_name, self.config)
            ext = "." + format_name if not format_name.startswith('.') else format_name
            output_file = file_output_dir / f"{data_type}{ext}"
            smart_save_items(items, output_file, data_type, self.config)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(formatted)

    def save_per_file_and_mega_outputs(
        self,
        file_path,
        generated,
        formats,
        batch_dt,
        config,
        generators_key="text_generators"
    ):
        """
        Save both per-file and mega-file outputs for each generator and format.
        Args:
            file_path: input file path
            generated: dict with {generator: [items]}
            formats: list of output formats (e.g., ["jsonl", "csv"])
            batch_dt: datetime string for batch (e.g., "0707251804")
            config: current config (dict)
            generators_key: config key to use for generators ("text_generators" or "media_generators")
        Returns:
            List of all per-file output paths created.
        """
        file_stem = Path(file_path).stem
        output_dir_base = Path(config.get('output_dir', './outputs')) / "data" / batch_dt
        output_dir_base.mkdir(parents=True, exist_ok=True)

        generators = config.get(generators_key, [])

        all_paths = []

        from doc2train.core.registries.formatter_registry import get_formatter

        for fmt in formats:
            for gen in generators:
                section_data = generated.get(gen)
                if not section_data:
                    continue
                ext = "." + fmt if not fmt.startswith('.') else fmt
                # Per-file output
                file_dir = output_dir_base / file_stem
                file_dir.mkdir(parents=True, exist_ok=True)
                per_file_path = file_dir / f"{file_stem}_{gen}{ext}"
                # Format
                formatter_cls = get_formatter(fmt)
                formatter = formatter_cls(config)
                format_method = getattr(formatter, f"format_{gen}", None)
                if format_method:
                    formatted = format_method(section_data)
                else:
                    formatted = formatter.format(section_data, gen)
                with open(per_file_path, "w", encoding="utf-8") as f:
                    f.write(formatted)
                all_paths.append(str(per_file_path))
                # Mega-file append
                mega_path = output_dir_base / f"final_{gen}{ext}"
                if fmt == "jsonl":
                    with open(mega_path, "a", encoding="utf-8") as f:
                        for item in section_data:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                elif fmt == "csv":
                    # TODO: For CSV, ensure correct header handling for appends
                    with open(mega_path, "a", encoding="utf-8") as f:
                        f.write(formatted)
                # Add logic for XML or other formats as needed

        return all_paths

class OutputManager:
    """
    Handles saving all extraction results (not just generated data).
    """
    def __init__(self, config=None):
        self.config = config or {}

    def save_all_results(
        self,
        extraction_results: Dict[str, Any],
        file_name: Optional[str] = 'summary_results'
    ) -> Path:
        """
        Save all results (summary, batch outputs).

        :param extraction_results: The data to serialize as JSON.
        :param file_name:      Optional custom filename (e.g. 'my_results.json').
                            If not provided, defaults to 'summary_results_<timestamp>.json'.
        :return:               Path to the written JSON file.
        """
        # 1. Ensure output directory exists
        output_dir = Path(self.config.get('output_dir', './outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Determine filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_file = output_dir / f"{file_name}_{timestamp}.json"

        # 3. Write JSON
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_results, f, indent=2, ensure_ascii=False)

        # 4. Return the path in case caller wants to know where it went
        return summary_file

# If you ever want to support smart_format_data, add a wrapper like:
def smart_format_data(items, data_type, config=None):
    """
    Format data using best available formatter (helper for pipeline).
    """
    from doc2train.core.registries.formatter_registry import get_formatter
    format_name = config.get("output_format", "jsonl") if config else "jsonl"
    fmt_cls = get_formatter(format_name)
    if not fmt_cls:
        raise ValueError(f"Formatter {format_name} not found")
    return fmt_cls(config).format(items, data_type)
