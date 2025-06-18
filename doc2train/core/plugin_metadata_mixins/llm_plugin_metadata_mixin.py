class LLMPluginMetadataMixin:
    provider_name = None  # Should be overridden
    priority = 10
    supported_types = ["text"]
    supports_vision = False
    description = ""
    version = "1.0.0"
    author = "Unknown"

    @classmethod
    def validate_metadata(cls, required=("provider_name",)):
        missing = [attr for attr in required if not getattr(cls, attr, None)]
        if missing:
            raise ValueError(f"LLM Plugin '{cls.__name__}' is missing required metadata: {missing}")
