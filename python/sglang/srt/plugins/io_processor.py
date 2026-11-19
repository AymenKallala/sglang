"""Model-specific preparation and postprocessing in the serving process."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sglang.srt.plugins import load_plugins_by_group


@dataclass(frozen=True)
class PreparedInput:
    """One model prompt and the state needed to interpret its output."""

    prompt: str | list[int]
    context: Any = None

    def __post_init__(self):
        if not isinstance(self.prompt, (str, list)) or not self.prompt:
            raise ValueError("A prepared prompt must contain text or token IDs")
        if isinstance(self.prompt, list) and not all(
            type(t) is int for t in self.prompt
        ):
            raise ValueError("A prepared token sequence must contain integer IDs")


@runtime_checkable
class IOProcessor(Protocol):
    """One instance per serving handler. Keep per-input state in PreparedInput."""

    def prepare(self, data: dict) -> PreparedInput: ...

    def postprocess(self, output: Any, context: Any) -> Any: ...


def load_io_processor(task: str, tokenizer_manager) -> IOProcessor | None:
    hf_config = tokenizer_manager.model_config.hf_config
    processors = getattr(hf_config, "sglang_io_processors", {})
    if not isinstance(processors, dict):
        raise ValueError(
            "sglang_io_processors must map task names to entry point names"
        )
    if task not in processors:
        return None

    name = processors[task]
    if not isinstance(name, str) or not name:
        raise ValueError(f"Expected an I/O processor name for {task!r}")
    plugins = load_plugins_by_group("sglang.io_processors", name=name)
    if name not in plugins:
        raise ValueError(f"I/O processor {name!r} for {task!r} is unavailable")
    factory, _ = plugins[name]
    processor = factory(tokenizer=tokenizer_manager.tokenizer, hf_config=hf_config)
    if not isinstance(processor, IOProcessor):
        raise TypeError(f"I/O processor {name!r} must define prepare and postprocess")
    return processor
