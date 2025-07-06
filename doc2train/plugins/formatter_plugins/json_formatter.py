from doc2train.plugins.formatter_plugins.base_formatters import BaseFormatter
import json

class JSONFormatter(BaseFormatter):
    name = "json"
    description = "Standard JSON formatter"
    priority = 10

    def format_conversations(self, conversations):
        return json.dumps(conversations, indent=2, ensure_ascii=False)

    def format_qa_pairs(self, qa_pairs):
        return json.dumps(qa_pairs, indent=2, ensure_ascii=False)

    def format_embeddings(self, embeddings):
        return json.dumps(embeddings, indent=2, ensure_ascii=False)

    def format_summaries(self, summaries):
        return json.dumps(summaries, indent=2, ensure_ascii=False)

    def get_file_extension(self):
        return ".json"
