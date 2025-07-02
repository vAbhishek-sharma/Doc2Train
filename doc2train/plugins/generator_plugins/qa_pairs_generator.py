# doc2train/plugins/generator_plugins/qa_pairs_generator.py

from typing import Any, Dict, List, Optional
from doc2train.plugins.generator_plugins.base_generator import BaseGenerator


class QAPairsGenerator(BaseGenerator):
    """
    Generator plugin for question–answer pairs.
    """
    name = 'qa_pairs'
    priority = 10
    description = None
    version = "1.0.0"
    author = "Doc2Train Team"
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, gen_type="qa_pairs")
        # define the JSON schema your LLM should emit
        self.schema = """
        {
          "qa_pairs": [
            {"question": "string", "answer": "string"},
            …
          ]
        }
        """

    def _parse(self, parsed_json: Any) -> List[Dict[str, Any]]:
        """Pull out the 'qa_pairs' array from the JSON."""
        return parsed_json.get("qa_pairs", [])

    def generate(
        self,
        text: str,
        images: Optional[List[Any]] = None,
        prompt_template: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        If you ever need to handle images specially for QA, you can override here.
        Otherwise just defer to BaseGenerator:
        """
        return super().generate(text, images=images, prompt_template=prompt_template)
