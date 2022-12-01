#
# When the hook scripts script are run, their current working directory is the root of the generated project
#
# SEE https://cookiecutter.readthedocs.io/en/stable/advanced/hooks.html

import shutil
import sys
from pathlib import Path
import os
from contextlib import contextmanager


if sys.version_info < (3, 8):
    raise ValueError(
        f"Unsupported python version, got {sys.version_info} and expected >=3.8"
    )


SELECTED_DOCKER_BASE = "{{ cookiecutter.docker_base }}"
SELECTED_GIT_REPO = "{{ cookiecutter.git_repo }}"


OSPARC_METADATA_PATH = Path(".osparc") / "metadata.yml"


def create_dockerfile():
    folder_name = Path("docker") / SELECTED_DOCKER_BASE.split(":")[0]

    # list folders
    # NOTE: it needs to be a list as we delete the folders

    for folder in list(f for f in Path("docker").glob("*") if f.is_dir()):
        if folder != folder_name:
            shutil.rmtree(folder)


def create_ignore_listings():
    # .gitignore
    common_gitignore = Path("Common.gitignore")
    python_gitignore = Path("Python.gitignore")

    gitignore_file = Path(".gitignore")
    gitignore_file.unlink(missing_ok=True)
    shutil.copyfile(common_gitignore, gitignore_file)

    if "python" in SELECTED_DOCKER_BASE:
        with gitignore_file.open("at") as fh:
            fh.write("\n")
            fh.write(python_gitignore.read_text())

    common_gitignore.unlink()
    python_gitignore.unlink()

    # .dockerignore
    common_dockerignore = Path("Common.dockerignore")
    dockerignore_file = Path(".dockerignore")
    dockerignore_file.unlink(missing_ok=True)
    shutil.copyfile(common_dockerignore, dockerignore_file)

    # appends .gitignore above
    with dockerignore_file.open("at") as fh:
        fh.write("\n")
        fh.write(gitignore_file.read_text())

    common_dockerignore.unlink()


def create_repo_folder():
    if SELECTED_GIT_REPO != "github":
        shutil.rmtree(".github")
    if SELECTED_GIT_REPO != "gitlab":
        shutil.rmtree(".gitlab")


@contextmanager
def context_print(
    msg,
):
    print("-", msg, end="...", flush=True)
    yield
    print("DONE")


def check_python():
    is_pyconfig = "python" in SELECTED_DOCKER_BASE

    for folder in ("src", ".osparc"):
        for fp in Path(folder).rglob("Python.*"):
            if fp.is_file():
                if is_pyconfig:
                    fp.rename(fp.parent / fp.name.removeprefix("Python."))
                else:
                    fp.unlink(missing_ok=True)

    if is_pyconfig:
        shutil.rmtree("service.cli", ignore_errors=True)
        Path(".osparc/docker-compose.overwrite.yml").unlink(missing_ok=True)

        # settings.json contains metadata
        if (Path(".osparc") / "settings.json").exists():
            OSPARC_METADATA_PATH.unlink(missing_ok=True)


def main():
    print("Starting post-gen-project hook:", flush=True)
    try:
        with context_print("Pruning docker/ folder to selection"):
            create_dockerfile()

        with context_print("Updating .gitignore and .dockerignore configs"):
            create_ignore_listings()

        with context_print("Updating service binder"):
            check_python()

        with context_print("Adding config for selected external repository"):
            create_repo_folder()

    except Exception as exc:  # pylint: disable=broad-except
        print("ERROR", exc)
        return os.EX_SOFTWARE
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
