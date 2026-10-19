"""Command-line entry point.

`skudio` launches the UI bound to localhost on a random free port.
A one-time URL token is passed via env var and expected as a `?token=` query parameter.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import click

import skudio
from skudio.server.net import LOOPBACK, ensure_loopback, pick_free_port
from skudio.server.token import new_token

_TOKEN_ENV = "SKUDIO_TOKEN"
_PROJECT_ENV = "SKUDIO_PROJECT"
_PROJECT_DEFAULT = Path.home() / ".skudio" / "projects" / "default"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(skudio.__version__, prog_name=skudio.NAME)
@click.option(
    "--host",
    default=LOOPBACK,
    show_default=True,
    help="Host to bind the local server to. Must be a loopback address."
)
@click.option(
    "--port", type=int, default=None, help="Port to bind. Default: OS-picked free port.")
@click.option(
    "--no-browser", is_flag=True, help="Do not open the browser.")
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=f"Directory to persist runs into. Default: {_PROJECT_DEFAULT}",
)
@click.option(
    "--no-token", is_flag=True, hidden=True, help="Skip the URL token (dev only, not recommended)."
)
def main(
    host: str, port: int | None, no_browser: bool, project_dir: Path | None, no_token: bool
) -> None:
    """ Launch skudio """
    click.echo(f"{skudio.NAME} v{skudio.__version__}")
    click.echo(skudio.TAGLINE)

    host = ensure_loopback(host)
    if no_token and host != LOOPBACK:
        raise click.UsageError("--no-token requires --host 127.0.0.1")
    if port is None:
        port = pick_free_port()

    resolved_project = _resolve_project_dir(project_dir)

    token = "" if no_token else new_token()
    url = f"http://{host}:{port}/"
    if token:
        url += f"?token={token}"

    click.echo(f"Serving on {url}")
    click.echo(f"Runs saved to {resolved_project}")

    env = os.environ.copy()
    if token:
        env[_TOKEN_ENV] = token
    env[_PROJECT_ENV] = str(resolved_project)

    # TODO: Replace streamlit with react-built served via fastapi
    app_path = Path(__file__).parent / "ui" / "streamlit" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableXsrfProtection", "true",
        "--server.enableCORS", "true",
        "--browser.gatherUsageStats", "false"
    ]

    if no_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url,new=1)

    try:
        subprocess.run(cmd, env=env, check=False)
    except FileNotFoundError:
        click.echo(f"Streamlit not installed. Install:\n   pip install {skudio.NAME}[ui]", err=True)
    sys.exit(1)


def _resolve_project_dir(project_dir: Path | None) -> Path:
    """Resolve `--project-dir` to an absolute Path and ensure it exists."""
    root = (project_dir or _PROJECT_DEFAULT).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root

