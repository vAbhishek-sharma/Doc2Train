# plugins/generator_plugins/conversation_generator.py

from doc2train.plugins.generator_plugins.base_generator import BaseGenerator

class ConversationGenerator(BaseGenerator):
    generator_name = "conversations"
    types_supported = ["conversations"]
    priority = 10

    def generate(self, chunk, gen_type, prompt_template=None):
        prompt = prompt_template or self.config.get("prompts", {}).get("custom", {}).get("conversations")
        # This is where you'd call your LLM (abstracted for demo)
        return {
            "conversations": [
                {
                    "messages": [
                        {"role": "user", "content": "What is Doc2Train?"},
                        {"role": "assistant", "content": "Doc2Train is a modular document processing tool."}
                    ]
                }
            ]
        }
