from doc2train.plugins.writer_plugins.base_writer import BaseWriter
import json

class FilesystemWriter(BaseWriter):
    writer_name = "filesystem"
    priority = 10

    def write(self, output_file, items, data_type):
        # Simple: always write as JSON for demo; you can use config to select format
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
