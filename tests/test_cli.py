from typer.testing import CliRunner

from vc.cli import app


def test_cli_loads():
    """Imports every command module path; a syntax error anywhere in cli.py fails here."""
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0 and "setup" in r.output and "launch" in r.output
