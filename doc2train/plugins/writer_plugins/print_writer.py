from doc2train.plugins.writer_plugins.base_writer import BaseWriter
import json

class PrintWriter(BaseWriter):
    writer_name = "print"
    priority = 10  # Lower = higher priority, so user can override

    def write(self, output_file, items, data_type):
        print(f"[PrintWriter] Data Type: {data_type}, Output File: {output_file}")
        print(json.dumps(items, indent=2, ensure_ascii=False))
