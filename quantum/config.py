from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG_PATH = Path("config/ibm_config.json")


def load_ibm_token(config_path: Optional[str] = None) -> Optional[str]:
    """
    Load IBM API token from a JSON config file and export to env if not set.

    The JSON must contain {"IBM_API_KEY": "..."}.

    Precedence: existing environment variables win; otherwise, config value
    is set into env var `IBM_API_KEY`.
    Also mirrored into `QISKIT_IBM_TOKEN` and `IBM_APU_KEY` for compatibility.
    """
    # If env already set, respect it.
    token = os.getenv("IBM_API_KEY") or os.getenv("QISKIT_IBM_TOKEN") or os.getenv("IBM_APU_KEY")
    if token:
        return token

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            token = data.get("IBM_API_KEY")
            if token:
                os.environ.setdefault("IBM_API_KEY", token)
                os.environ.setdefault("QISKIT_IBM_TOKEN", token)
                os.environ.setdefault("IBM_APU_KEY", token)
                return token
    except Exception:
        # Silent failure; callers will fall back to Aer
        return None
    return None

