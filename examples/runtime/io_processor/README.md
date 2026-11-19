# Model I/O processors

An installed package can prepare model inputs and interpret their outputs on the existing Python HTTP server. The example in this directory accepts a conversation on `/v1/classify` and returns named probabilities for each turn. It assumes a supported token classifier that emits independent class logits at every token and a tokenizer whose chat template ends each turn with one EOS token.

Install the example into the server's Python environment, then serve the checkpoint with the processor enabled:

```bash
pip install ./examples/runtime/io_processor
python -m sglang.launch_server --model-path "$MODEL_PATH" \
    --json-model-override-args '{"sglang_io_processors":{"classify":"turn_scores"}}'
```

You can also put the same `sglang_io_processors` mapping in the checkpoint's Hugging Face `config.json`.

```bash
curl -s http://localhost:30000/v1/classify \
    -H 'Content-Type: application/json' \
    -d '{"input":{"messages":[{"role":"user","content":"Hello"}]}}' | jq '.data[0]'
```

For a checkpoint with labels `question` and `greeting`, an example result is:

```json
{"turns": [{"question": 0.12, "greeting": 0.93}]}
```

## Write a processor

Register a factory in the `sglang.io_processors` package entry-point group (see `pyproject.toml`). SGLang calls it at handler startup with the keyword arguments `tokenizer` and `hf_config`. The processor instance can serve concurrent requests.

- `prepare(data)` validates one input and returns a `PreparedInput` containing text or token IDs and optional per-input context. The example carries positions in the exact token IDs it sends to the model.
- `postprocess(output, context)` gets the model output and the context from that input. Keep per-request state in the context.

| Config key | Input to `prepare` | Input to `postprocess` | Return value |
| --- | --- | --- | --- |
| `classify` | The object in `input`. Text and token-ID requests use standard classification. | Raw model scores for the prompt | A JSON object in the standard response's `data` list |
| `rerank` | One `query`, `document`, and optional `instruct` per document | One row from `score_prompts` | One numeric relevance score |

A rerank processor for a decoder model supplies its `label_token_ids` as an instance attribute. It receives the probabilities of those tokens in order. A sequence-classification reranker receives its model scores. Return the same prompt type for all documents in a request. The handler batches the prompts and applies `top_n` and `return_documents`.
