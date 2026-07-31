# HackGPT

Welcome to HackGPT, a powerful and customizable AI chatbot interface designed specifically for developers. With a ChatGPT-like experience, HackGPT brings a host of key features to enhance your hacking capabilities and empower your development process.

[![HackGPT Demo](static/images/demo.gif)](./static/images/demo.gif)

## Current Features

- **ChatGPT-like Interface**: Immerse yourself in a chat-like environment with responses streamed as they are generated.

- **Model Selection**: Choose from GPT-5.6, GPT-4.1, and GPT-4o models. Compatible models also expose reasoning effort, reasoning mode, and response detail controls.

- **Set-up Prompt Selection**: Unlock more specific responses, results, and knowledge by selecting from a variety of preset set-up prompts. Additionally, craft your own custom set-up prompt for personalized interactions.

- **Knowledge Cut-off**: Ask the selected model to answer from the perspective of a chosen date in the past.

- **On-demand Switching**: Seamlessly switch between GPT-5.6 model tiers and set-up prompts to balance capability, latency, and cost.

## Getting Started

HackGPT is built using Python 3.9 and Streamlit. It can be easily deployed in a containerized environment using the included Docker Compose set-up.

To get started, follow the steps below:

### Docker Compose (recommended)

The easiest way to run HackGPT is to use Docker Compose. To do so, run:

```bash
docker compose -f docker/docker-compose.dev.yml up # -d to run in background
```

To stop the container, press `Ctrl+C` in the terminal window, or run:

```bash
docker compose -f docker/docker-compose.dev.yml stop
```

... in case you used the `-d` flag to run the container in the background.

To remove the container, run:

```bash
docker compose -f docker/docker-compose.dev.yml down
```

### Docker (manually)

If you don't want to use Docker Compose, you can also run the container manually. To do so, run:

```bash
docker run --rm -p 8501:8501 -v "$(pwd)/src:/app" -w /app python:3.9-slim bash -c "pip install -r requirements.txt && streamlit run app.py"
```

### VS Code Dev Container

With the VS Code Dev Containers extension installed, open the Command Palette and select **Dev Containers: Reopen in Container**. The Python dependencies are installed automatically and port 8501 is forwarded for Streamlit.

### Local Python environment

If you don't want to use Docker, you can also run HackGPT in a local Python environment. To do so, run:

```bash
cd ./src
pip install -r requirements.txt
streamlit run app.py
```

## Source layout

- `src/app.py` — minimal Streamlit entry point.
- `src/hackgpt/config.py` — application constants and model metadata.
- `src/hackgpt/openai_service.py` — streamed OpenAI Responses API client.
- `src/hackgpt/prompts.py` — bundled setup-prompt loading and composition.
- `src/hackgpt/state.py` — Streamlit session-state management.
- `src/hackgpt/ui/` — sidebar and chat rendering components.
- `src/hackgpt/data/` — packaged setup-prompt data.

## Feedback

We hope this initial release of HackGPT serves as a valuable tool for developers, encouraging exploration and experimentation with GPT-based models. Your feedback on HackGPT is highly appreciated as it will help us shape future versions of the project.

We're thrilled to have you join us in this journey of unleashing new possibilities and customizability for developers with HackGPT!

## License

This project is licensed under the terms of the ISC license. See [LICENSE](LICENSE) for more details.
