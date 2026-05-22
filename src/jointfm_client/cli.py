"""Command line interface for the JointFM Python SDK."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, TextIO, cast

from jointfm_client.client import JointFMClient
from jointfm_client.configuration import (
    DEFAULT_CLI_DOTENV_PATH,
    DEFAULT_CLI_RETURN_MODE,
    DEFAULT_CLI_TIME_INDEX_MODE,
)
from jointfm_client.contract import ReturnMode, TimeIndexMode
from jointfm_client.exceptions import JointFMError, JointFMResponseError


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``jointfm-client`` command line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = cast(_CommandHandler, args.handler)
    try:
        return handler(args, sys.stdout)
    except (JointFMError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"jointfm-client: {_format_cli_error(error)}", file=sys.stderr)
        return 2


_CommandHandler = Any


def _format_cli_error(error: BaseException) -> str:
    message = str(error)
    if not isinstance(error, JointFMResponseError):
        return message

    details = [f"HTTP {error.status_code}"]
    if error.datarobot_request_id is not None:
        details.append(f"DataRobot request ID: {error.datarobot_request_id}")
    body_excerpt = _single_line(error.response_body_excerpt)
    if body_excerpt != "":
        details.append(f"response excerpt: {body_excerpt}")
    return f"{message} ({'; '.join(details)})"


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jointfm-client",
        description="Call a DataRobot-hosted JointFM deployment from the terminal.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser(
        "health",
        help="Validate settings and print non-secret service metadata as JSON.",
    )
    _add_dotenv_arguments(health_parser)
    health_parser.set_defaults(handler=_health_command)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Submit one JSON request file and write the JSON response file.",
    )
    _add_dotenv_arguments(predict_parser)
    predict_parser.add_argument("request_file", type=Path)
    predict_parser.add_argument("response_file", type=Path)
    predict_parser.set_defaults(handler=_predict_command)

    forecast_parser = subparsers.add_parser(
        "forecast-csv",
        help="Forecast from a CSV history file and write tidy forecast rows as CSV.",
    )
    _add_dotenv_arguments(forecast_parser)
    forecast_parser.add_argument("history_file", type=Path)
    forecast_parser.add_argument("output_file", type=Path)
    forecast_parser.add_argument(
        "--query-times",
        required=True,
        help="Comma-separated future query times.",
    )
    forecast_parser.add_argument(
        "--target-column",
        action="append",
        required=True,
        help="Target column name. Repeat for multiple target columns.",
    )
    forecast_parser.add_argument(
        "--time-index-mode",
        choices=("ordinal", "continuous_float", "absolute_datetime"),
        default=DEFAULT_CLI_TIME_INDEX_MODE,
    )
    forecast_parser.add_argument("--time-column")
    forecast_parser.add_argument(
        "--requested-column",
        action="append",
        help="Requested output column. Defaults to the target columns.",
    )
    forecast_parser.add_argument(
        "--return-mode",
        choices=("mean", "samples", "quantiles"),
        default=DEFAULT_CLI_RETURN_MODE,
    )
    forecast_parser.add_argument("--n-samples", type=int)
    forecast_parser.add_argument(
        "--quantiles",
        help="Comma-separated quantile levels for quantile forecasts.",
    )
    forecast_parser.add_argument("--seed", type=int)
    forecast_parser.set_defaults(handler=_forecast_csv_command)

    return parser


def _add_dotenv_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=DEFAULT_CLI_DOTENV_PATH,
        help="Path to a .env file. Defaults to .env in the current directory.",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Read only process environment variables.",
    )


def _health_command(args: argparse.Namespace, stdout: TextIO) -> int:
    client = JointFMClient.from_env(dotenv_path=_dotenv_path(args))
    health = client.health()
    payload: dict[str, Any] = {"service": asdict(health)}
    if client.settings is not None:
        payload["deployment"] = {
            "selector": client.settings.deployment_selector,
            "deployment_id": client.settings.deployment_id,
            "deployment_url": client.settings.deployment_url,
            "deployment_target": client.settings.deployment_target,
            "health_url": client.settings.health_url,
            "predict_url": client.settings.predict_url,
        }
    _write_json_payload(stdout, payload)
    return 0


def _predict_command(args: argparse.Namespace, stdout: TextIO) -> int:
    del stdout
    request_payload = _read_json_object(args.request_file)
    client = JointFMClient.from_env(dotenv_path=_dotenv_path(args))
    response_payload = client.predict(request_payload)
    _write_json_file(args.response_file, response_payload)
    return 0


def _forecast_csv_command(args: argparse.Namespace, stdout: TextIO) -> int:
    del stdout
    pandas_module = _require_pandas()
    client = JointFMClient.from_env(dotenv_path=_dotenv_path(args))
    frame = pandas_module.read_csv(args.history_file)
    time_index_mode = cast(TimeIndexMode, args.time_index_mode)
    return_mode = cast(ReturnMode, args.return_mode)
    requested_columns = args.requested_column or args.target_column
    result = client.forecast(
        frame,
        query_times=_parse_query_times(args.query_times, time_index_mode),
        time_index_mode=time_index_mode,
        time_column=args.time_column,
        requested_columns=requested_columns,
        return_mode=return_mode,
        n_samples=args.n_samples,
        quantiles=None if args.quantiles is None else _parse_float_list(args.quantiles),
        seed=args.seed,
        target_columns=args.target_column,
    )
    output_frame = result.to_pandas_tidy()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output_file, index=False)
    return 0


def _dotenv_path(args: argparse.Namespace) -> Path | None:
    return None if args.no_dotenv else cast(Path, args.dotenv)


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_payload(stdout: TextIO, payload: Mapping[str, Any]) -> None:
    stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_query_times(raw_value: str, time_index_mode: TimeIndexMode) -> list[Any]:
    values = _parse_string_list(raw_value)
    if time_index_mode == "ordinal":
        return [int(value) for value in values]
    if time_index_mode == "continuous_float":
        return [float(value) for value in values]
    return values


def _parse_float_list(raw_value: str) -> list[float]:
    return [float(value) for value in _parse_string_list(raw_value)]


def _parse_string_list(raw_value: str) -> list[str]:
    values = [value.strip() for value in raw_value.split(",") if value.strip() != ""]
    if not values:
        raise ValueError("comma-separated values must not be empty")
    return values


def _require_pandas() -> Any:
    try:
        import pandas as pandas_module
    except ImportError as error:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "forecast-csv requires installing jointfm-client[dataframe]"
        ) from error
    return pandas_module


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess smoke tests
    raise SystemExit(main())