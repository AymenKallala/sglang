"""Example for a token classifier trained to score the EOS at each turn."""

import torch
from pydantic import BaseModel, Field

from sglang.srt.plugins.io_processor import PreparedInput


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: list[Message] = Field(min_length=1)


class TurnScores:
    def __init__(self, *, tokenizer, hf_config):
        self.tokenizer = tokenizer
        self.labels = [hf_config.id2label[i] for i in range(len(hf_config.id2label))]

    def prepare(self, data: dict) -> PreparedInput:
        conversation = Conversation.model_validate(data)
        messages = [message.model_dump() for message in conversation.messages]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        positions = [
            i for i, token in enumerate(ids) if token == self.tokenizer.eos_token_id
        ]
        if len(positions) != len(messages):
            raise ValueError("The template must produce one EOS per message")
        return PreparedInput(ids, positions)

    def postprocess(self, output, context) -> dict:
        turns = []
        for position in context:
            scores = (
                torch.tensor(output[position], dtype=torch.float32).sigmoid().tolist()
            )
            turns.append(dict(zip(self.labels, scores, strict=True)))
        return {"turns": turns}
