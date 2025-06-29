# -------------------------------------------------------------
# doc2train/plugins/generator_plugins/summaries_generator.py
from typing import Any, Dict, List, Optional
from doc2train.plugins.generator_plugins.base_generator import BaseGenerator


class SummariesGenerator(BaseGenerator):
    """
    Generator plugin for summaries.
    Emits JSON like:
    {
      "summaries": ["summary string 1", "summary string 2", ...]
    }
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, gen_type="summaries")
        self.schema = """
        {
          "summaries": ["string", ...]
        }
        """

    def _parse(self, parsed_json: Any) -> List[Dict[str, Any]]:
        items = parsed_json.get("summaries", [])
        # wrap each string in a dict so downstream formatters can key off it
        return [{"summary": s} for s in items]
