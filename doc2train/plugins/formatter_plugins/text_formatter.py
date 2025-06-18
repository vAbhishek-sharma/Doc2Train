from doc2train.plugins.formatter_plugins.base_formatters import BaseFormatter

class TextFormatter(BaseFormatter):
    format_name = "txt"
    description = "Human-readable plain text formatter"
    format_name = 10

    def format_conversations(self, conversations):
        output = ["CONVERSATION TRAINING DATA", "=" * 50, f"Total conversations: {len(conversations)}", ""]
        for i, conv in enumerate(conversations, 1):
            output.append(f"CONVERSATION {i}\n" + "-" * 30)
            if 'messages' in conv:
                for message in conv['messages']:
                    role = message.get('role', 'unknown').upper()
                    content = message.get('content', '')
                    output.append(f"{role}: {content}\n")
            if 'source_file' in conv:
                output.append(f"Source: {conv['source_file']}")
            output.append("=" * 50 + "\n")
        return '\n'.join(output)

    def format_qa_pairs(self, qa_pairs):
        output = ["QUESTION & ANSWER TRAINING DATA", "=" * 50, f"Total Q&A pairs: {len(qa_pairs)}", ""]
        for i, qa in enumerate(qa_pairs, 1):
            output.append(f"Q&A PAIR {i}\n" + "-" * 20)
            output.append(f"QUESTION: {qa.get('question', '')}\n")
            output.append(f"ANSWER: {qa.get('answer', '')}\n")
            if 'source_file' in qa:
                output.append(f"Source: {qa['source_file']}")
            output.append("=" * 50 + "\n")
        return '\n'.join(output)

    def format_embeddings(self, embeddings):
        output = ["EMBEDDING TRAINING DATA", "=" * 50, f"Total embedding pairs: {len(embeddings)}", ""]
        for i, emb in enumerate(embeddings, 1):
            output.append(f"EMBEDDING PAIR {i}\n" + "-" * 25)
            output.append(f"Sentence 1: {emb.get('sentence1', '')}")
            output.append(f"Sentence 2: {emb.get('sentence2', '')}")
            output.append(f"Similarity: {emb.get('similarity', 'N/A')}\n")
            if 'source_file' in emb:
                output.append(f"Source: {emb['source_file']}")
            output.append("=" * 50 + "\n")
        return '\n'.join(output)

    def format_summaries(self, summaries):
        output = ["SUMMARY TRAINING DATA", "=" * 50, f"Total summaries: {len(summaries)}", ""]
        for i, summary in enumerate(summaries, 1):
            output.append(f"SUMMARY {i}\n" + "-" * 15)
            output.append(f"SUMMARY: {summary.get('summary', '')}\n")
            if 'original_text' in summary:
                original = summary['original_text']
                if len(original) > 200:
                    original = original[:200] + "..."
                output.append(f"ORIGINAL (excerpt): {original}\n")
            if 'source_file' in summary:
                output.append(f"Source: {summary['source_file']}")
            output.append("=" * 50 + "\n")
        return '\n'.join(output)

    def get_file_extension(self):
        return ".txt"
