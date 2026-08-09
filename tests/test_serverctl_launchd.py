import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="launchd is available only on macOS"
)

ROOT = Path(__file__).resolve().parents[1]
SERVERCTL = ROOT / "serverctl.sh"


def _executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


@pytest.fixture
def launchd_environment(tmp_path):
    home = tmp_path / "home & user"
    fake_bin = tmp_path / "bin"
    home.mkdir()
    fake_bin.mkdir()

    state = tmp_path / "launchd-state"
    calls = tmp_path / "launchctl-calls"
    server = fake_bin / "junie-mlx-vlm"
    serverctl = tmp_path / "serverctl.sh"

    serverctl.write_bytes(SERVERCTL.read_bytes())
    serverctl.chmod(0o755)
    _executable(
        server,
        '#!/bin/bash\nexit "${FAKE_SERVER_EXIT:-0}"\n',
    )
    _executable(
        fake_bin / "curl",
        '#!/bin/bash\nexit "${FAKE_CURL_EXIT:-22}"\n',
    )
    _executable(
        fake_bin / "lsof",
        '#!/bin/bash\nexit "${FAKE_LSOF_EXIT:-1}"\n',
    )
    _executable(
        fake_bin / "launchctl",
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >>"$LAUNCHCTL_CALLS"
case "$1" in
  print)
    [ -f "$LAUNCHCTL_STATE" ] || exit 113
    if [ "${FAKE_LAUNCHD_STOPPED:-0}" = "1" ]; then
      printf 'state = not running\n'
    else
      printf 'state = running\n    pid = 4242\n'
    fi
    ;;
  bootstrap)
    touch "$LAUNCHCTL_STATE"
    ;;
  kickstart)
    touch "$LAUNCHCTL_STATE"
    [ "${2:-}" = "-p" ] && printf '4242\n'
    ;;
  bootout)
    rm -f "$LAUNCHCTL_STATE"
    ;;
  *)
    exit 2
    ;;
esac
""",
    )

    label = f"com.junie.mlx-vlm.test.{os.getpid()}"
    config = home / "config & settings.json"
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "JUNIE_LAUNCHD_LABEL": label,
        "JUNIE_SERVER_CONFIG": str(config),
        "LAUNCHCTL_CALLS": str(calls),
        "LAUNCHCTL_STATE": str(state),
    }
    return {
        "env": env,
        "home": home,
        "label": label,
        "config": config,
        "server": server,
        "serverctl": serverctl,
        "state": state,
        "calls": calls,
    }


def _run(command: str, environment: dict, **env_updates):
    env = {**environment["env"], **env_updates}
    return subprocess.run(
        [str(environment["serverctl"]), command],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_start_and_stop_manage_one_launch_agent(launchd_environment):
    context = launchd_environment
    started = _run("start", context)

    assert started.returncode == 0, started.stderr
    plist_path = (
        context["home"]
        / "Library"
        / "LaunchAgents"
        / f"{context['label']}.plist"
    )
    payload = plistlib.loads(plist_path.read_bytes())
    assert payload == {
        "Label": context["label"],
        "ProgramArguments": [str(context["server"])],
        "EnvironmentVariables": {
            "JUNIE_SERVER_CONFIG": str(context["config"])
        },
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 5,
        "StandardOutPath": str(
            context["config"].parent / "junie-mlx-vlm-daemon.log"
        ),
        "StandardErrorPath": str(
            context["config"].parent / "junie-mlx-vlm-daemon.log"
        ),
    }
    first_calls = context["calls"].read_text().splitlines()
    assert first_calls[0].startswith("print gui/")
    assert first_calls[1].startswith("bootstrap gui/")

    duplicate = _run("start", context)
    assert duplicate.returncode == 0
    assert "Already managed by launchd (pid 4242)" in duplicate.stdout
    assert context["calls"].read_text().splitlines()[-2:] == [
        first_calls[0],
        first_calls[0],
    ]

    stopped = _run("stop", context)
    assert stopped.returncode == 0, stopped.stderr
    assert "removed its LaunchAgent" in stopped.stdout
    assert not plist_path.exists()
    assert not context["state"].exists()
    stop_calls = context["calls"].read_text().splitlines()
    assert any(line.startswith("bootout gui/") for line in stop_calls)
    assert stop_calls[-1].startswith("print gui/")


def test_restart_reloads_the_launch_agent(launchd_environment):
    context = launchd_environment
    assert _run("start", context).returncode == 0

    restarted = _run("restart", context)

    assert restarted.returncode == 0, restarted.stderr
    commands = [line.split()[0] for line in context["calls"].read_text().splitlines()]
    assert commands.count("bootstrap") == 2
    assert commands.count("bootout") == 1
    assert context["state"].exists()


def test_start_reloads_a_cleanly_stopped_agent(launchd_environment):
    context = launchd_environment
    assert _run("start", context).returncode == 0

    started = _run("start", context, FAKE_LAUNCHD_STOPPED="1")

    assert started.returncode == 0, started.stderr
    commands = [line.split()[0] for line in context["calls"].read_text().splitlines()]
    assert commands.count("bootstrap") == 2
    assert commands.count("bootout") == 1
    assert context["state"].exists()


def test_start_rejects_a_busy_port(launchd_environment):
    context = launchd_environment

    result = _run("start", context, FAKE_LSOF_EXIT="0")

    assert result.returncode == 1
    assert "port 19239 is already in use" in result.stderr
    assert not context["state"].exists()


def test_start_rejects_matching_gateway_and_worker_ports(launchd_environment):
    context = launchd_environment
    context["config"].write_text('{"port": 19300, "worker_port": 19300}')

    result = _run("start", context)

    assert result.returncode == 1
    assert "gateway port 19300 and worker port 19300 must differ" in result.stderr
    assert not context["state"].exists()


def test_start_rejects_an_unusable_binary(launchd_environment):
    context = launchd_environment

    result = _run("start", context, FAKE_SERVER_EXIT="1")

    assert result.returncode == 1
    assert "exists but cannot be started" in result.stderr
    assert not context["state"].exists()
