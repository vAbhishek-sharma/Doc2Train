from doc2train.plugins.formatter_plugins.base_formatters import BaseFormatter

class MarkdownFormatter(BaseFormatter):
    format_name = "markdown"
    description = "Markdown formatter for documentation-style output"
    format_name = 10

    def format_conversations(self, conversations):
        output = [ "# Conversation Training Data", "", f"**Total conversations:** {len(conversations)}", ""]
        for i, conv in enumerate(conversations, 1):
            output.append(f"## Conversation {i}\n")
            if 'messages' in conv:
                for message in conv['messages']:
                    role = message.get('role', 'unknown').title()
                    content = message.get('content', '')
                    output.append(f"**{role}:** {content}\n")
            if 'source_file' in conv:
                output.append(f"*Source: {conv['source_file']}*\n")
            output.append("---\n")
        return '\n'.join(output)

    def format_qa_pairs(self, qa_pairs):
        output = [ "# Q&A Training Data", "", f"**Total Q&A pairs:** {len(qa_pairs)}", ""]
        for i, qa in enumerate(qa_pairs, 1):
            output.append(f"## Q&A Pair {i}\n")
            output.append(f"**Q:** {qa.get('question', '')}\n")
            output.append(f"**A:** {qa.get('answer', '')}\n")
            if 'source_file' in qa:
                output.append(f"*Source: {qa['source_file']}*\n")
            output.append("---\n")
        return '\n'.join(output)

    def format_embeddings(self, embeddings):
        output = [ "# Embedding Training Data", "", f"**Total embedding pairs:** {len(embeddings)}", ""]
        for i, emb in enumerate(embeddings, 1):
            output.append(f"## Embedding Pair {i}\n")
            output.append(f"**Sentence 1:** {emb.get('sentence1', '')}\n")
            output.append(f"**Sentence 2:** {emb.get('sentence2', '')}\n")
            output.append(f"**Similarity:** {emb.get('similarity', 'N/A')}\n")
            if 'source_file' in emb:
                output.append(f"*Source: {emb['source_file']}*\n")
            output.append("---\n")
        return '\n'.join(output)

    def format_summaries(self, summaries):
        output = [ "# Summary Training Data", "", f"**Total summaries:** {len(summaries)}", ""]
        for i, summary in enumerate(summaries, 1):
            output.append(f"## Summary {i}\n")
            output.append(f"**Summary:** {summary.get('summary', '')}\n")
            if 'original_text' in summary:
                output.append(f"**Original Text:**\n```")
                original = summary['original_text']
                if len(original) > 500:
                    original = original[:500] + "..."
                output.append(original)
                output.append("```\n")
            if 'source_file' in summary:
                output.append(f"*Source: {summary['source_file']}*\n")
            output.append("---\n")
        return '\n'.join(output)

    def get_file_extension(self):
        return ".md"
