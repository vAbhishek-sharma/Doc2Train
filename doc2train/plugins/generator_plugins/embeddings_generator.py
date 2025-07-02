# doc2train/plugins/generator_plugins/embeddings_generator.py
from typing import Any, Dict, List, Optional
from doc2train.plugins.generator_plugins.base_generator import BaseGenerator


class EmbeddingsGenerator(BaseGenerator):
    """
    Generator plugin for embeddings.
    The user is expected to provide a prompt_template that defines a valid JSON
    output structure, for example:
    {
      "embeddings": [
        {"text": "string", "vector": [0.1, 0.2, ...]},
        ...
      ]
    }
    """
    name = 'embeddings'
    priority = 10
    description = None
    version = "1.0.0"
    author = "Doc2Train Team"
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, gen_type="embeddings")
        # no fixed schema: rely on user-provided prompt_template

    def _parse(self, parsed_json: Any) -> List[Dict[str, Any]]:
        # Expect top-level "embeddings" key
        return parsed_json.get("embeddings", [])
