# -------------------------------------------------------------
# doc2train/plugins/generator_plugins/conversations_generator.py
from typing import Any, Dict, List, Optional
from doc2train.plugins.generator_plugins.base_generator import BaseGenerator


class ConversationsGenerator(BaseGenerator):
    """
    Generator plugin for dialogue or conversation turns.
    Emits JSON like:
    {
      "conversations": [
        {"speaker": "string", "text": "string"},
        ...
      ]
    }
    """

    name = 'conversations'
    priority = 10
    description = None
    version = "1.0.0"
    author = "Doc2Train Team"
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, gen_type="conversations")
        self.schema = """
        {
          "conversations": [
            {"speaker": "string", "text": "string"},
            …
          ]
        }
        """

    def _parse(self, parsed_json: Any) -> List[Dict[str, Any]]:
        return parsed_json.get("conversations", [])


