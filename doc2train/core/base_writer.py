class BaseWriter:
    writer_name = "filesystem"
    priority = 10  # Lower = higher priority

    def __init__(self, config=None):
        self.config = config or {}

    def write(self, output_file, items, data_type):
        """
        Save items (list of dict) to output_file.
        Args:
            output_file: Path to save to
            items: List of data dicts
            data_type: String describing the data type (conversations, qa_pairs, etc)
        """
        raise NotImplementedError("Subclasses must implement write()")
