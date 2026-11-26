import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "validate.py"), "--repo", str(repo), "--map", "maps/two_rooms.wad"],
        capture_output=True, text=True, timeout=300,
    )


def make_repo(tmp_path: Path, agent_src: str) -> Path:
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "agent.py").write_text(agent_src)
    return tmp_path


def test_template_baseline_passes():
    out = run(ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASSED" in out.stdout


def test_bad_signature_fails(tmp_path):
    out = run(make_repo(tmp_path, "class DoomAgent:\n    def __init__(self, a): pass\n    def reset(self): pass\n    def act(self, o): pass\n"))
    assert out.returncode == 1
    assert "DoomAgent.__init__" in out.stdout


def test_crashing_agent_fails(tmp_path):
    src = (
        "class DoomAgent:\n    def __init__(self, o, a, d): pass\n    def reset(self): pass\n"
        "    def act(self, obs): raise RuntimeError('kaboom')\n"
    )
    out = run(make_repo(tmp_path, src))
    assert out.returncode == 1
    assert "kaboom" in out.stdout


def test_optional_and_varargs_signatures_pass(tmp_path):
    src = (
        "import numpy as np\n"
        "class DoomAgent:\n    def __init__(self, observation_space, action_space, device='cpu', *, extra=1): pass\n"
        "    def reset(self, *args): pass\n    def act(self, obs, **kw): return np.zeros(9, np.int8)\n"
    )
    out = run(make_repo(tmp_path, src))
    assert "DoomAgent interface" in out.stdout and out.returncode == 0, out.stdout


def test_gitignored_submission_files_fail(tmp_path):
    repo = make_repo(tmp_path, "import numpy as np\nfrom lib.policy import X\nclass DoomAgent:\n"
                     "    def __init__(self, o, a, d): pass\n    def reset(self): pass\n"
                     "    def act(self, obs): return np.zeros(9, np.int8)\n")
    (repo / "submission" / "lib").mkdir()
    (repo / "submission" / "lib" / "policy.py").write_text("X = 1\n")
    (repo / ".gitignore").write_text("lib/\n")
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    out = run(repo)
    assert out.returncode == 1 and "ignored by .gitignore" in out.stdout, out.stdout


def test_requirements_rules(tmp_path):
    repo = make_repo(tmp_path, "class DoomAgent:\n    def __init__(self, o, a, d): pass\n    def reset(self): pass\n    def act(self, o): pass\n")
    (repo / "submission" / "requirements.txt").write_text("einops==0.8.1\n--find-links https://evil.example\n")
    out = run(repo)
    assert "unsupported line" in out.stdout and out.returncode == 1
