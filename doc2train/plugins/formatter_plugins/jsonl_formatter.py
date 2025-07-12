from doc2train.plugins.formatter_plugins.base_formatters import BaseFormatter
import json

class JSONLFormatter(BaseFormatter):
    name = "jsonl"
    description = "JSON Lines formatter"
    priority = 10

    def format_conversations(self, conversations):
        return '\n'.join(json.dumps(conv, ensure_ascii=False) for conv in conversations)

    def format_qa_pairs(self, qa_pairs):
        return '\n'.join(json.dumps(qa, ensure_ascii=False) for qa in qa_pairs)

    def format_embeddings(self, embeddings):
        return '\n'.join(json.dumps(emb, ensure_ascii=False) for emb in embeddings)

    def format_summaries(self, summaries):
        return '\n'.join(json.dumps(summary, ensure_ascii=False) for summary in summaries)

    def get_file_extension(self):
        return ".jsonl"
