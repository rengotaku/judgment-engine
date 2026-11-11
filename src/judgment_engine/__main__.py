"""`python -m judgment_engine` で起動する（uvicorn の factory を包む）。"""

from __future__ import annotations

import uvicorn

from .app import create_app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8088, log_level="info")


if __name__ == "__main__":
    main()
