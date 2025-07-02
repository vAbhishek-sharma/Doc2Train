# doc2train/core/generator_base.py

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from doc2train.core.extractor import chunk_text
from doc2train.core.registries.llm_registry import get_llm_client
import ipdb

class BaseGenerator(ABC):
    """
    Abstract base for all generator plugins.
    Handles:
      - Splitting text into chunks
      - Building prompts (including optional vision input)
      - Calling the LLM (with JSON‐retry logic)
      - Aggregating & returning parsed results
    """
    generator_name = None
    priority = 10
    description = None
    version = "1.0.0"
    author = "Doc2Train Team"

    def __init__(self, config: Dict[str, Any], gen_type: str):
        self.config    = config
        self.gen_type  = gen_type
        self.client    = get_llm_client(config)
        self.prompts   = config.get("custom_prompts", {})
        self.use_async = config.get("use_async", False)

        # retry behavior
        self.max_retries = config["llm"].get("max_retries", 3)
        self.retry_backoff = config["llm"].get("retry_backoff", 1)
        self.retry_template = config["llm"].get(
            "retry_prompt",
            "Your output wasn't valid JSON.  Please re-emit exactly valid JSON matching this schema:\n\n{schema}\n\nPrevious output:\n{previous}"
        )

    def generate(
        self,
        text: str,
        images: Optional[List[Any]] = None,
        prompt_template: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        1. Chunk the text
        2. For each chunk, format the prompt
        3. Call the LLM (text or vision)
        4. Retry up to max_retries if JSON parse fails
        5. Parse & accumulate via subclass’s _parse()
        """
        chunks = chunk_text(text, config=self.config)
        tpl    = prompt_template or self.prompts.get(self.gen_type, "")
        schema = getattr(self, "schema", "{}")

        all_items: List[Dict[str, Any]] = []

        for chunk in chunks:
            if '{chunk}' in tpl:
                prompt = tpl.format(chunk=chunk)
            elif '{text}' in tpl:
                prompt = tpl.format(text=chunk)
            else:
                prompt = tpl

            raw: Optional[str] = None
            for attempt in range(1, self.max_retries + 1):
                if images is not None:
                    raw = self.client.call_vision(prompt, images, self.config.get('model'))
                else:
                    raw = self.client.call_text(prompt, task=self.gen_type)

                try:
                    parsed = json.loads(raw)
                    break
                except (json.JSONDecodeError, TypeError):
                    if attempt == self.max_retries:
                        raise ValueError(
                            f"Failed to parse JSON after {self.max_retries} tries:\n{raw}"
                        )
                    # build a retry prompt
                    retry = self.retry_template.format(schema=schema, previous=raw or "")
                    prompt = retry
                    time.sleep(self.retry_backoff)

            # delegate parsing of the JSON object to the subclass
            items = self._parse(parsed)
            all_items.extend(items)

        return all_items

    @abstractmethod
    def _parse(self, parsed_json: Any) -> List[Dict[str, Any]]:
        """
        Given a parsed JSON object, extract a list of item dicts.
        """
        ...


