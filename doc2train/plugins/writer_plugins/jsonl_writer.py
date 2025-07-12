from doc2train.plugins.writer_plugins.base_writer import BaseWriter
import json

class JSONLWriter(BaseWriter):
    writer_name = "jsonl"
    priority = 10

    def write(self, output_file, items, data_type):
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
