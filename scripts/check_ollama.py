"""Quick preflight: is Ollama reachable and is the configured model pulled?

    python scripts/check_ollama.py

Run this on the machine that will host local generation before the demo.
"""

import sys

import requests

from app.config import settings


def main() -> int:
    host = settings.ollama_host.rstrip("/")
    model = settings.ollama_model
    print(f"Checking Ollama at {host} for model '{model}'...")
    try:
        tags = requests.get(f"{host}/api/tags", timeout=5).json()
    except Exception as exc:
        print(f"  ✗ Could not reach Ollama: {exc}")
        print("    Start it with `ollama serve` (it usually runs as a service after install).")
        return 1

    names = [m.get("name", "") for m in tags.get("models", [])]
    if any(model in n or n.startswith(model) for n in names):
        print(f"  ✓ Ollama is up and '{model}' is available.")
        print("    Try:  python -m app.cli ask \"Summarize Alpha Trading LLC\"")
        return 0

    print(f"  ✗ '{model}' is not pulled. Installed models: {names or '(none)'}")
    print(f"    Pull it with:  ollama pull {model}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
