from doc2train.plugins.formatter_plugins.base_formatters import BaseFormatter
import csv
import json
import io

class CSVFormatter(BaseFormatter):
    format_name = "csv"
    description = "CSV (flat table) formatter"
    format_name = 10

    def format_conversations(self, conversations):
        return self._dict_list_to_csv(self._flatten_conversations(conversations))

    def format_qa_pairs(self, qa_pairs):
        return self._dict_list_to_csv(qa_pairs)

    def format_embeddings(self, embeddings):
        return self._dict_list_to_csv(embeddings)

    def format_summaries(self, summaries):
        return self._dict_list_to_csv(summaries)

    def _flatten_conversations(self, conversations):
        flattened = []
        for conv in conversations:
            if 'messages' in conv:
                for i, msg in enumerate(conv['messages']):
                    flat = {
                        "conversation_id": conv.get('id', ''),
                        "source_file": conv.get('source_file', ''),
                        "message_index": i,
                        "role": msg.get('role', ''),
                        "content": msg.get('content', ''),
                        "conversation_length": len(conv['messages'])
                    }
                    flattened.append(flat)
        return flattened

    def _dict_list_to_csv(self, dict_list):
        if not dict_list:
            return ""
        output = io.StringIO()
        fieldnames = sorted({key for item in dict_list for key in item.keys()})
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in dict_list:
            row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in item.items()}
            writer.writerow(row)
        return output.getvalue()

    def get_file_extension(self):
        return ".csv"
