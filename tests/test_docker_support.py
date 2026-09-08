from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import List, Optional
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"
DEVCONTAINER_PATH = REPO_ROOT / ".devcontainer" / "devcontainer.json"


@dataclass
class DockerInstruction:
    instruction: str
    arguments: str
    stage: Optional[str] = None


@dataclass
class DockerStage:
    base_image: str
    name: Optional[str]
    instructions: List[DockerInstruction]


class DockerfileParser:
    """Parses and validates Dockerfile directives across build stages."""

    def __init__(self, content: str):
        self.content = content
        self.stages = self._parse_stages(content)

    @classmethod
    def from_file(cls, path: Path) -> "DockerfileParser":
        if not path.exists():
            raise FileNotFoundError(f"Dockerfile not found at {path}")
        return cls(path.read_text(encoding="utf-8"))

    def _parse_stages(self, content: str) -> List[DockerStage]:
        stages: List[DockerStage] = []
        current_stage: Optional[DockerStage] = None

        # Join line continuations (lines ending with backslash)
        raw_lines = content.splitlines()
        joined_lines: List[str] = []
        buffer = ""

        for line in raw_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.endswith("\\"):
                buffer += stripped[:-1].strip() + " "
            else:
                buffer += stripped
                joined_lines.append(buffer.strip())
                buffer = ""

        from_pattern = re.compile(r"^FROM\s+(\S+)(?:\s+[aA][sS]\s+(\S+))?", re.IGNORECASE)

        for line in joined_lines:
            parts = line.split(None, 1)
            if not parts:
                continue
            instruction = parts[0].upper()
            args = parts[1] if len(parts) > 1 else ""

            if instruction == "FROM":
                match = from_pattern.match(line)
                if not match:
                    raise ValueError(f"Malformed FROM directive: {line}")
                base_image = match.group(1)
                stage_name = match.group(2)
                current_stage = DockerStage(base_image=base_image, name=stage_name, instructions=[])
                stages.append(current_stage)
            else:
                if current_stage is None:
                    raise ValueError(f"Instruction {instruction} found before any FROM statement.")
                current_stage.instructions.append(
                    DockerInstruction(
                        instruction=instruction,
                        arguments=args,
                        stage=current_stage.name,
                    )
                )

        return stages

    def validate_multistage(self, min_stages: int = 2) -> None:
        if len(self.stages) < min_stages:
            raise ValueError(
                f"Expected at least {min_stages} stages in Dockerfile, found {len(self.stages)}"
            )

    def validate_base_image(self, expected_prefix: str = "python:3.14-slim") -> None:
        if not self.stages:
            raise ValueError("No stages found in Dockerfile")
        for stage in self.stages:
            if not stage.base_image.startswith(expected_prefix):
                raise ValueError(
                    f"Stage '{stage.name}' uses invalid base image '{stage.base_image}', "
                    f"expected prefix '{expected_prefix}'"
                )

    def validate_uv_bundled(self) -> None:
        uv_found = False
        for stage in self.stages:
            for inst in stage.instructions:
                if inst.instruction == "COPY" and "ghcr.io/astral-sh/uv" in inst.arguments:
                    uv_found = True
                    break
        if not uv_found:
            raise ValueError("Dockerfile does not copy uv binary from ghcr.io/astral-sh/uv")

    def validate_cbc_installed(self) -> None:
        cbc_found = False
        for stage in self.stages:
            for inst in stage.instructions:
                if inst.instruction == "RUN" and "coinor-cbc" in inst.arguments:
                    cbc_found = True
                    break
        if not cbc_found:
            raise ValueError("Dockerfile does not install coinor-cbc package")

    def validate_entrypoint(self, expected: str = "romutil") -> None:
        entrypoint_found = False
        for stage in self.stages:
            for inst in stage.instructions:
                if inst.instruction == "ENTRYPOINT" and expected in inst.arguments:
                    entrypoint_found = True
                    break
        if not entrypoint_found:
            raise ValueError(f"Dockerfile does not configure ENTRYPOINT invoking '{expected}'")


def load_devcontainer(path: Path = DEVCONTAINER_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Devcontainer file not found at {path}")
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


class TestDockerSupportPositive:
    """Positive test cases verifying required Dockerfile and devcontainer configuration."""

    def test_dockerfile_exists(self):
        assert DOCKERFILE_PATH.is_file(), f"{DOCKERFILE_PATH} does not exist"

    def test_dockerignore_exists_and_contains_rules(self):
        assert DOCKERIGNORE_PATH.is_file(), f"{DOCKERIGNORE_PATH} does not exist"
        content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
        assert ".git" in content
        assert ".venv" in content
        assert "__pycache__" in content

    def test_dockerfile_multistage_structure(self):
        parser = DockerfileParser.from_file(DOCKERFILE_PATH)
        parser.validate_multistage(min_stages=2)
        stage_names = [s.name for s in parser.stages]
        assert "builder" in stage_names, "Expected stage named 'builder'"
        assert "runtime" in stage_names, "Expected stage named 'runtime'"

    def test_dockerfile_python_base_images(self):
        parser = DockerfileParser.from_file(DOCKERFILE_PATH)
        parser.validate_base_image("python:3.14-slim")

    def test_dockerfile_copies_uv(self):
        parser = DockerfileParser.from_file(DOCKERFILE_PATH)
        parser.validate_uv_bundled()

    def test_dockerfile_installs_coinor_cbc(self):
        parser = DockerfileParser.from_file(DOCKERFILE_PATH)
        parser.validate_cbc_installed()

        # Verify clean package caches
        runtime_stage = [s for s in parser.stages if s.name == "runtime"][0]
        run_commands = [i.arguments for i in runtime_stage.instructions if i.instruction == "RUN"]
        assert any("coinor-cbc" in cmd and "rm -rf /var/lib/apt/lists/*" in cmd for cmd in run_commands), (
            "Runtime stage must clean apt lists after installing coinor-cbc"
        )

    def test_dockerfile_entrypoint_and_cmd(self):
        parser = DockerfileParser.from_file(DOCKERFILE_PATH)
        parser.validate_entrypoint("romutil")

        runtime_stage = [s for s in parser.stages if s.name == "runtime"][0]
        entrypoint_inst = [i for i in runtime_stage.instructions if i.instruction == "ENTRYPOINT"]
        assert len(entrypoint_inst) == 1
        assert "romutil" in entrypoint_inst[0].arguments

        cmd_inst = [i for i in runtime_stage.instructions if i.instruction == "CMD"]
        assert len(cmd_inst) >= 1
        assert "--help" in cmd_inst[0].arguments

    def test_dockerfile_volume_and_workdir(self):
        parser = DockerfileParser.from_file(DOCKERFILE_PATH)
        runtime_stage = [s for s in parser.stages if s.name == "runtime"][0]

        workdirs = [i.arguments for i in runtime_stage.instructions if i.instruction == "WORKDIR"]
        assert "/data" in workdirs, "Runtime stage must set WORKDIR to /data"

        volumes = [i.arguments for i in runtime_stage.instructions if i.instruction == "VOLUME"]
        assert any("/data" in v for v in volumes), "Runtime stage must declare /data VOLUME"

    def test_dockerfile_path_env_setup(self):
        parser = DockerfileParser.from_file(DOCKERFILE_PATH)
        runtime_stage = [s for s in parser.stages if s.name == "runtime"][0]

        envs = [i.arguments for i in runtime_stage.instructions if i.instruction == "ENV"]
        assert any("PATH=" in env and "/app/.venv/bin" in env for env in envs), (
            "Runtime stage must add /app/.venv/bin to PATH"
        )

    def test_devcontainer_exists(self):
        assert DEVCONTAINER_PATH.is_file(), f"{DEVCONTAINER_PATH} does not exist"

    def test_devcontainer_valid_json(self):
        data = load_devcontainer()
        assert isinstance(data, dict)
        assert "name" in data
        assert "ROMUtil" in data["name"]

    def test_devcontainer_build_references_dockerfile(self):
        data = load_devcontainer()
        assert "build" in data, "devcontainer.json must contain 'build' mapping"
        build_cfg = data["build"]
        dockerfile_rel = build_cfg.get("dockerfile")
        assert dockerfile_rel is not None

        # Resolve relative to .devcontainer directory
        resolved_dockerfile = (DEVCONTAINER_PATH.parent / dockerfile_rel).resolve()
        assert resolved_dockerfile == DOCKERFILE_PATH.resolve()
        assert resolved_dockerfile.exists()

    def test_devcontainer_post_create_command(self):
        data = load_devcontainer()
        post_create = data.get("postCreateCommand", "")
        assert "uv sync" in post_create, "postCreateCommand must run 'uv sync'"

    def test_devcontainer_vscode_settings_and_extensions(self):
        data = load_devcontainer()
        customizations = data.get("customizations", {})
        vscode = customizations.get("vscode", {})

        settings = vscode.get("settings", {})
        assert "python.defaultInterpreterPath" in settings
        assert "python.testing.pytestEnabled" in settings
        assert settings["python.testing.pytestEnabled"] is True

        extensions = vscode.get("extensions", [])
        assert "ms-python.python" in extensions


class TestDockerSupportNegative:
    """Negative test cases verifying parser rejection of non-conforming Docker and devcontainer configs."""

    def test_missing_dockerfile_raises_error(self, tmp_path):
        non_existent = tmp_path / "Dockerfile.missing"
        with pytest.raises(FileNotFoundError):
            DockerfileParser.from_file(non_existent)

    def test_instruction_before_from_raises_error(self):
        malformed = "RUN echo hello\nFROM python:3.14-slim"
        with pytest.raises(ValueError, match="found before any FROM"):
            DockerfileParser(malformed)

    def test_single_stage_rejected(self):
        single_stage = (
            "FROM python:3.14-slim\n"
            "RUN apt-get update && apt-get install -y coinor-cbc\n"
            "ENTRYPOINT [\"romutil\"]\n"
        )
        parser = DockerfileParser(single_stage)
        with pytest.raises(ValueError, match="Expected at least 2 stages"):
            parser.validate_multistage(min_stages=2)

    def test_wrong_base_image_rejected(self):
        wrong_base = (
            "FROM python:3.11-slim AS builder\n"
            "RUN echo ok\n"
            "FROM python:3.11-slim AS runtime\n"
            "RUN echo ok\n"
        )
        parser = DockerfileParser(wrong_base)
        with pytest.raises(ValueError, match="invalid base image"):
            parser.validate_base_image("python:3.14-slim")

    def test_missing_uv_copy_rejected(self):
        no_uv = (
            "FROM python:3.14-slim AS builder\n"
            "RUN pip install uv\n"
            "FROM python:3.14-slim AS runtime\n"
            "RUN echo ok\n"
        )
        parser = DockerfileParser(no_uv)
        with pytest.raises(ValueError, match="does not copy uv binary"):
            parser.validate_uv_bundled()

    def test_missing_cbc_rejected(self):
        no_cbc = (
            "FROM python:3.14-slim AS builder\n"
            "COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv\n"
            "FROM python:3.14-slim AS runtime\n"
            "ENTRYPOINT [\"romutil\"]\n"
        )
        parser = DockerfileParser(no_cbc)
        with pytest.raises(ValueError, match="does not install coinor-cbc"):
            parser.validate_cbc_installed()

    def test_missing_entrypoint_rejected(self):
        no_ep = (
            "FROM python:3.14-slim AS builder\n"
            "COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv\n"
            "FROM python:3.14-slim AS runtime\n"
            "CMD [\"python\", \"-m\", \"romutil\"]\n"
        )
        parser = DockerfileParser(no_ep)
        with pytest.raises(ValueError, match="does not configure ENTRYPOINT"):
            parser.validate_entrypoint("romutil")

    def test_malformed_devcontainer_json_raises_error(self, tmp_path):
        bad_json = tmp_path / "devcontainer.json"
        bad_json.write_text("{ \"name\": \"broken\", ", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_devcontainer(bad_json)

    def test_devcontainer_missing_file_raises_error(self, tmp_path):
        non_existent = tmp_path / "devcontainer.json"
        with pytest.raises(FileNotFoundError):
            load_devcontainer(non_existent)
