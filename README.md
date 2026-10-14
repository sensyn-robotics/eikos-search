# eikos-search
This is web search tool which can extract exact the answer your question using Eikos(open source Jev).
This is different from general LLM because the system never answer to question directory, but shows the existing article, which is already exists.
And this is different from ordinal Google search because it only shows the exact pharase in the most relevant article.

![pipeline](doc/pipeline.png)

Eikos never generates text: it returns calibrated probabilities over a list of options. eikos-search makes every
candidate sentence of the fetched articles an option, so the answer is always a verbatim slice of an existing page,
or an explicit "no article states this exactly". The full specification is in [doc/specification.tex](doc/specification.tex).

## Setup

```bash
git clone --recursive <this repo>   # Eikos is a pinned submodule in third_party/eikos
uv sync
```

## Usage

```bash
uv run eikos-search config/config.yaml          # web UI at http://127.0.0.1:8000
uv run eikos-search config/config.yaml --repl   # ask in the terminal
```

`POST /api/answer` with `{"question": "..."}` returns the sentence, its paragraph, the source URL and the probability.

## Model

`model.size` (`4b` | `27b`) and `model.precision` (`bf16` | `fp8` | `int4`) in `config/config.yaml` select the Eikos
build; every run prints which one it loaded.

| size | precision | weights | backend |
|---|---|---|---|
| 4b | bf16 | 9.3 GB | transformers or vllm |
| 4b | fp8 / int4 | 5.8 / 4.0 GB | vllm |
| 27b | bf16 | 55.6 GB | transformers or vllm |
| 27b | fp8 / int4 | 31.2 / 19.4 GB | vllm |

For `backend: vllm`, start `third_party/eikos/scripts/serve_vllm.sh <model dir> 8001` and set `model.vllm_url`.

## Tests

```bash
uv run pytest
```
