from doc2train.plugins.formatter_plugins.base_formatters import BaseFormatter
import xml.etree.ElementTree as ET
import xml.dom.minidom

class XMLFormatter(BaseFormatter):
    format_name = "xml"
    description = "XML formatter for structured data"
    format_name = 10

    def format_conversations(self, conversations):
        root = ET.Element("conversations", count=str(len(conversations)))
        for i, conv in enumerate(conversations):
            conv_elem = ET.SubElement(root, "conversation", id=str(i + 1))
            if 'source_file' in conv:
                conv_elem.set("source", conv['source_file'])
            if 'messages' in conv:
                messages_elem = ET.SubElement(conv_elem, "messages")
                for j, message in enumerate(conv['messages']):
                    msg_elem = ET.SubElement(messages_elem, "message", index=str(j), role=message.get('role', ''))
                    msg_elem.text = message.get('content', '')
        return self._prettify_xml(root)

    def format_qa_pairs(self, qa_pairs):
        root = ET.Element("qa_pairs", count=str(len(qa_pairs)))
        for i, qa in enumerate(qa_pairs):
            qa_elem = ET.SubElement(root, "qa_pair", id=str(i + 1))
            if 'source_file' in qa:
                qa_elem.set("source", qa['source_file'])
            ET.SubElement(qa_elem, "question").text = qa.get('question', '')
            ET.SubElement(qa_elem, "answer").text = qa.get('answer', '')
        return self._prettify_xml(root)

    def format_embeddings(self, embeddings):
        root = ET.Element("embeddings", count=str(len(embeddings)))
        for i, emb in enumerate(embeddings):
            emb_elem = ET.SubElement(root, "embedding_pair", id=str(i + 1), similarity=str(emb.get('similarity', '')))
            if 'source_file' in emb:
                emb_elem.set("source", emb['source_file'])
            ET.SubElement(emb_elem, "sentence1").text = emb.get('sentence1', '')
            ET.SubElement(emb_elem, "sentence2").text = emb.get('sentence2', '')
        return self._prettify_xml(root)

    def format_summaries(self, summaries):
        root = ET.Element("summaries", count=str(len(summaries)))
        for i, summary in enumerate(summaries):
            summary_elem = ET.SubElement(root, "summary", id=str(i + 1))
            if 'source_file' in summary:
                summary_elem.set("source", summary['source_file'])
            ET.SubElement(summary_elem, "summary_text").text = summary.get('summary', '')
            if 'original_text' in summary:
                ET.SubElement(summary_elem, "original_text").text = summary['original_text']
        return self._prettify_xml(root)

    def _prettify_xml(self, elem):
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def get_file_extension(self):
        return ".xml"
