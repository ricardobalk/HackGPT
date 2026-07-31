# HackGPT

HackGPT is a Streamlit chat interface with provider-neutral model discovery,
streaming, setup prompts, and schema-driven generation controls.

[![HackGPT Demo](static/images/demo.gif)](./static/images/demo.gif)

## Features

- **Multiple providers**: Use OpenAI, Anthropic Claude, or Ollama without changing
  the chat interface.
- **Dynamic model discovery**: The model picker reflects the models available to
  the configured account or Ollama host. Models can be refreshed from the sidebar.
- **Capability-aware configuration**: Every model supplies its own typed option
  schema, including defaults, ranges, choices, and conditional fields. Unsupported
  controls are never sent to a provider.
- **Provider-neutral chat**: Text history and system prompts remain usable when
  switching between compatible providers and models.
- **Streaming responses**: All provider adapters stream response text into the
  existing chat interface.
- **Setup prompts and historical context**: Select a bundled or custom setup prompt
  and optionally add a validated historical knowledge cutoff.

The capability model already represents streaming, system prompts, tools,
structured output, image input, and reasoning. Adding UI for future capabilities
does not require provider-specific branches in the generic option renderer.

## Provider configuration

Credentials and endpoints are read only from environment variables. They are not
entered or displayed in the Streamlit interface.

| Provider | Required | Optional |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Ollama | A reachable Ollama service | `OLLAMA_HOST`, `OLLAMA_API_KEY` |

Copy the example file and fill in only the providers you use:

```bash
cp .env.example .env
```

Docker Compose reads the root `.env` file automatically. For a local Python run,
export the variables into the process environment before starting Streamlit. If
`OLLAMA_HOST` is omitted, the Ollama client uses its local default. When Streamlit
runs in a container and Ollama runs on the host, set `OLLAMA_HOST` to an address
reachable from that container.

Provider availability is account- and deployment-specific. A configured provider
can still report a clear discovery error when credentials are invalid, its endpoint
is unreachable, or it has no compatible chat models.

## Getting started

HackGPT requires Python 3.12 or newer.

### Docker Compose

```bash
docker compose -f docker/docker-compose.dev.yml up
```

Stop or remove the development service with:

```bash
docker compose -f docker/docker-compose.dev.yml stop
docker compose -f docker/docker-compose.dev.yml down
```

### Local development

```bash
python -m pip install -r requirements-dev.txt
streamlit run src/app.py
```

### VS Code Dev Container

Choose **Dev Containers: Reopen in Container** from the Command Palette. The
container installs the development requirements, forwards port 8501, and runs the
`HackGPT: Start Streamlit` workspace task when the folder opens. Its dedicated
integrated terminal is revealed automatically so startup messages, request logs,
and errors remain visible. The forwarded app also opens in a VS Code preview.

Use `Ctrl+C` in that terminal to stop Streamlit. To start it again, run
**Tasks: Run Task** and select `HackGPT: Start Streamlit`. After changing
`.devcontainer/devcontainer.json`, use **Dev Containers: Rebuild and Reopen in
Container** so the updated container settings take effect.

## Quality checks

Run the complete local verification suite from the repository root:

```bash
ruff format --check src tests
ruff check src tests
mypy src tests
pytest
```

## Architecture

- `src/app.py` — Streamlit entry point and provider-registry lifetime.
- `src/hackgpt/providers/types.py` — provider protocol, capabilities, option schemas,
  normalised messages, validation, and user-facing provider errors.
- `src/hackgpt/providers/` — isolated OpenAI, Anthropic, and Ollama adapters plus the
  provider registry.
- `src/hackgpt/ui/options.py` — generic schema-to-Streamlit widget rendering.
- `src/hackgpt/ui/sidebar.py` — provider/model discovery and application controls.
- `src/hackgpt/ui/chat.py` — provider-neutral history and streamed chat rendering.
- `src/hackgpt/prompts.py` — bundled setup-prompt loading and historical context.
- `src/hackgpt/state.py` — session-state initialisation and history normalisation.
- `tests/` — schema, selection, adapter translation, state, and Streamlit smoke tests.

## License

This project is licensed under the terms of the ISC license. See [LICENSE](LICENSE)
for details.
