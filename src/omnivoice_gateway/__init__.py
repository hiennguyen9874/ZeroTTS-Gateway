import argparse
import json
import logging

from .app import create_app
from .config import load_settings

__all__ = ["create_app", "load_settings", "main"]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {"level": record.levelname, "logger": record.name, "msg": record.getMessage(), **getattr(record, "fields", {})},
            ensure_ascii=False,
        )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="omnivoice-gateway")
    parser.add_argument("--config", help="path to the gateway TOML config (default: $OMNIVOICE_GATEWAY_CONFIG)")
    settings = load_settings(parser.parse_args().config)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1, access_log=False, log_config=None)
