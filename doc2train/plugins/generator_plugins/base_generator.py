from datetime import datetime
import json
import time
from typing import Any, Dict, List, Optional

from abc import ABC
from doc2train.core.extractor import chunk_text
from doc2train.core.registries.llm_registry import get_llm_client


class BaseGenerator(ABC):
    """
    Abstract base for all generator plugins.
    Handles:
      - Splitting text into chunks
      - Building prompts (including optional vision input)
      - Calling the LLM (with JSON-retry logic and markdown cleanup)
      - Aggregating & returning parsed results
      Dispatches by input type: text, vision, audio, video.
    """
    generator_name = None
    priority = 10
    description = None
    version = "1.0.0"
    author = "Doc2Train Team"

    def __init__(self, config: Dict[str, Any], gen_type: str):
        self.config = config
        self.gen_type = gen_type
        self.client = get_llm_client(config)
        self.prompts = config.get("custom_prompts", {})
        self.use_async = config.get("use_async", False)

        # retry behavior
        self.max_retries = config["llm"].get("max_retries", 3)
        self.retry_backoff = config["llm"].get("retry_backoff", 1)
        self.retry_template = config["llm"].get(
            "retry_prompt",
            "Your output wasn't valid JSON. Please re-emit exactly valid JSON matching this schema:\n\n{schema}\n\nPrevious output:\n{previous}"
        )

    def generate(
        self,
        *,
        text: Optional[str] = None,
        input_type: str,
        images: Optional[List[Any]] = None,
        audio: Optional[Any] = None,
        video: Optional[Any] = None,
        prompt_template: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # pick prompt template
        tpl = prompt_template or self.prompts.get(self.gen_type, "")

        if input_type == 'text':
            return self._generate_text(text or "", tpl)
        elif input_type in ('vision', 'direct_vision'):
            return self._generate_vision(images or [], tpl)
        elif input_type == 'audio':
            return self._generate_audio(audio, tpl)
        elif input_type == 'video':
            return self._generate_video(video, tpl)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    def _generate_text(self, text: str, tpl: str) -> List[Dict[str, Any]]:
        chunks = chunk_text(text, config=self.config)
        all_items: List[Dict[str, Any]] = []
        for chunk in chunks:
            # use simple replace to avoid KeyError from stray braces
            if '{chunk}' in tpl:
                prompt = tpl.replace('{chunk}', chunk)
            elif '{text}' in tpl:
                prompt = tpl.replace('{text}', chunk)
            else:
                prompt = tpl

            items = self._call_with_retries(
                lambda p: self.client.call_text_model(p, task=self.gen_type),
                prompt
            )
            all_items.extend(items)
        self._tag_metadata(all_items)
        return all_items

    def _generate_vision(self, images: List[Any], tpl: str) -> List[Dict[str, Any]]:
        prompt = tpl
        items = self._call_with_retries(
            lambda p: self.client.call_vision_model(p, images, self.config.get('model')),
            prompt
        )
        self._tag_metadata(items)
        return items

    def _generate_audio(self, audio: Any, tpl: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("Audio generation not implemented")

    def _generate_video(self, video: Any, tpl: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("Video generation not implemented")

    def _clean_markdown(self, raw: str) -> str:
        """
        Strip markdown code fences like ```json ... ``` if present.
        """
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines)
        return raw

    def _call_with_retries(self, call_fn, prompt: str) -> List[Dict[str, Any]]:
        raw: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            raw = call_fn(prompt)
            cleaned = self._clean_markdown(raw).strip()
            try:
                parsed = json.loads(cleaned)
                return self._parse(parsed)
            except (json.JSONDecodeError, TypeError):
                if attempt == self.max_retries:
                    raise ValueError(
                        f"Failed to parse JSON after {self.max_retries} tries:\n{cleaned}"
                    )
                # update prompt for retry without re-formatting stray braces
                prompt = self.retry_template.replace('{schema}', self.schema).replace('{previous}', cleaned)
                time.sleep(self.retry_backoff)
        return []

    def _tag_metadata(self, items: List[Dict[str, Any]]):
        ts = datetime.utcnow().isoformat() + "Z"
        for item in items:
            item.setdefault("generator", self.gen_type)
            item.setdefault("timestamp", ts)

    def _parse(self, parsed_json: Any) -> List[Dict[str, Any]]:
        """
        Default parse: supports lists of items or dicts with 'items', or keys by gen_type.
        """
        if isinstance(parsed_json, list):
            return parsed_json
        if isinstance(parsed_json, dict):
            if 'items' in parsed_json and isinstance(parsed_json['items'], list):
                return parsed_json['items']
            if self.gen_type in parsed_json and isinstance(parsed_json[self.gen_type], list):
                return parsed_json[self.gen_type]
            return [parsed_json]
        raise ValueError(f"Unexpected parsed type: {type(parsed_json)}")
