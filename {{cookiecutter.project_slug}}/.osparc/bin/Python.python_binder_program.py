#!/usr/bin/env python
#
# This code loads and runs side-by-side with user's code
# therefore dependencies and python compatibility is critical
#
#   binder --> user-code
#   osparc-base-image
#

ALERT_PREFIX = "🚨"
TIP_PREFIX = "🔊 TIP:"

import sys

if sys.version_info < (3, 8):
    raise ValueError(
        f"{ALERT_PREFIX} Unsupported python version, got {sys.version_info} and expected >=3.8"
    )

import importlib  # nopycln: import
import importlib.util  # nopycln: import
import inspect
import json
import logging
import os
import sys
from contextlib import suppress
from copy import deepcopy
from inspect import Parameter, Signature
from pathlib import Path
from textwrap import indent
from typing import Any, Callable, Mapping, Optional
from typing import get_args, get_origin


try:
    # .osparc/requirements.txt
    import rich
    import typer
    import yaml
    from pydantic import (
        BaseModel,
        BaseSettings,
        ValidationError,
        validate_arguments,
        validator,
    )
    from pydantic.decorator import ValidatedFunction
    from pydantic.tools import schema_of
    from rich.console import Console

except ImportError as err:
    err.msg += f".\n {TIP_PREFIX} did you install osparc python dependencies, i.e. 'pip install -r .osparc/requirements.txt'??"
    raise


THIS_FILEPATH = Path(sys.argv[0] if __name__ == "__main__" else __file__).resolve()
THIS_FILEPATH = THIS_FILEPATH.with_name(THIS_FILEPATH.name.removeprefix("Python."))
DOT_OSPARC_DIR = THIS_FILEPATH.parent.parent

assert DOT_OSPARC_DIR.exists()  # nosec
assert DOT_OSPARC_DIR.name == ".osparc"  # nosec

error_console = Console(stderr=True)


# ----------------------------------------------------------------------------------------------------------
# DISCOVER & BIND
# ----------------------------------------------------------------------------------------------------------


class DotOsparcSettings(BaseModel):
    publish_functions: list[str]
    metadata: dict[str, Any]


def discover_published_functions(
    functions_dotted_names: list[str], *, dot_osparc_dir: Path
) -> list:
    IMPORT_MODULE_EXCEPTIONS = (AttributeError, ModuleNotFoundError, FileNotFoundError)

    def _import_module_from_path(module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    published = []
    for dotted_name in functions_dotted_names:
        parts = dotted_name.split(".")
        module_name = ".".join(parts[:-1])
        func_name = parts[-1]

        assert dotted_name == f"{module_name}.{func_name}"  # nosec

        try:
            module = None
            # namespaces to path: a.b.c.f  -> a/b/c
            namespaces_path = "/".join(parts[:-1])

            for guess_dir in (
                dot_osparc_dir.parent,
                dot_osparc_dir.parent / "src",
            ):
                # module can be a package 'a/b/__init__.py' or a file 'a/b.py'
                for guess_module_path in (
                    guess_dir / (namespaces_path + ".py"),
                    guess_dir / namespaces_path / "__init__.py",
                ):
                    with suppress(*IMPORT_MODULE_EXCEPTIONS):
                        module = _import_module_from_path(
                            module_name, guess_module_path
                        )
                        if module:
                            break
                if module:
                    break

            if module is None:
                raise ImportError(
                    f"Cannot find module {module_name}.{func_name}",
                    name=module_name,
                )

            published.append(getattr(module, func_name))

        except AttributeError as exc:
            error_console.log(
                f"{ALERT_PREFIX} Skipping function '{dotted_name}':\n{indent(f'{exc}', prefix='->')}"
            )

    return published


# ----------------------------------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------------------------------


class SchemaResolver:
    @classmethod
    def name_type(cls, parameter_annotation):
        try:
            if issubclass(parameter_annotation, float):
                name = "number"
            elif issubclass(parameter_annotation, int):
                name = "integer"
            elif issubclass(parameter_annotation, str):
                name = "string"
            else:
                name = f"{parameter_annotation}".replace("typing.", "")
        except TypeError:
            name = f"{parameter_annotation}".replace("typing.", "")

        return name

    @classmethod
    def _replace_value_in_dict(cls, item: Any, original_schema: dict[str, Any]):
        #
        # Taken and adapted from https://github.com/samuelcolvin/pydantic/issues/889#issuecomment-850312496
        # TODO: check https://github.com/gazpachoking/jsonref

        if isinstance(item, list):
            return [cls._replace_value_in_dict(i, original_schema) for i in item]
        elif isinstance(item, dict):
            if "$ref" in item.keys():
                # Limited to something like "$ref": "#/definitions/Engine"
                definitions = item["$ref"][2:].split("/")
                res = original_schema.copy()
                for definition in definitions:
                    res = res[definition]
                return res
            else:
                return {
                    key: cls._replace_value_in_dict(i, original_schema)
                    for key, i in item.items()
                }
        else:
            return item

    @classmethod
    def resolve_refs(cls, schema: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in str(schema):
            # NOTE: this is a minimal solution that cannot cope e.g. with
            # the most generic $ref with might be URLs. For that we will be using
            # directly jsonschema python package's resolver in the near future.
            # In the meantime we can live with this
            return cls._replace_value_in_dict(deepcopy(schema), deepcopy(schema.copy()))
        return schema


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dump_dot_osparc_config(core_func: Callable, settings_metadata: dict[str, Any]):
    def _create_inputs(parameters: Mapping[str, Parameter]) -> dict[str, Any]:
        inputs = {}
        for parameter in parameters.values():
            # should only allow keyword argument
            assert parameter.kind == parameter.KEYWORD_ONLY
            assert parameter.annotation != Parameter.empty

            # build each input
            description = getattr(
                parameter.annotation,
                "description",
                parameter.name.replace("_", " ").capitalize(),
            )

            # FIXME: files are represented differently!
            content_schema = schema_of(
                parameter.annotation,
                title=parameter.name.capitalize(),
            )

            data = {
                "label": parameter.name,
                "description": description,
                "type": "ref_contentSchema",
                "contentSchema": SchemaResolver.resolve_refs(content_schema),
            }

            if parameter.default != Parameter.empty:
                # TODO: what if partial-field defaults?
                data["defaultValue"] = parameter.default

            inputs[parameter.name] = data
        return inputs

    def _create_outputs(return_annotation: Any) -> dict[str, Any]:
        def _as_args_tuple(return_annotation: Any) -> tuple:
            if return_annotation == Signature.empty:
                return tuple()

            origin = get_origin(return_annotation)

            if origin and origin is tuple:
                # multiple outputs
                return_args_types = get_args(return_annotation)
            else:
                # single output
                return_args_types = (return_annotation,)
            return return_args_types

        # TODO: add extra info on outputs?
        outputs = {}

        return_args_types = _as_args_tuple(return_annotation)
        for index, return_type in enumerate(return_args_types, start=1):
            name = f"out_{index}"

            if return_type is None:
                continue

            display_name = f"Out{index} {SchemaResolver.name_type(return_type)}"
            content_schema = schema_of(return_type, title=display_name)
            data = {
                "label": display_name,
                "description": "",
                "type": "ref_contentSchema",
                "contentSchema": SchemaResolver.resolve_refs(content_schema),
            }
            outputs[name] = data
        return outputs

    if inspect.isgeneratorfunction(core_func):
        raise NotImplementedError(f"Cannot process function iterators as {core_func}")

    signature = inspect.signature(core_func)
    inputs = _create_inputs(signature.parameters)
    outputs = _create_outputs(signature.return_annotation)

    # TODO: sync this with metadata and runtime models!
    config_folder = DOT_OSPARC_DIR / "services" / core_func.__name__
    config_folder.mkdir(parents=True, exist_ok=True)

    # NOTE settings metadata is for the entire repo
    service_name = f"{settings_metadata['name']}-{core_func.__name__}"
    service_key = f"{settings_metadata['key']}-{core_func.__name__}"

    assert service_key.endswith(service_name)  # nosec

    def _update_metadata_file():
        metadata = deepcopy(settings_metadata)
        metadata_path = config_folder / "metadata.yml"

        if not metadata_path.exists():
            # init
            metadata.update(
                **{
                    "name": service_name,
                    "key": service_key,
                }
            )
        else:
            # previous version takes precedence
            prev_metadata = yaml.safe_load(metadata_path.read_text())
            metadata.update(prev_metadata)

        metadata.update(
            **{
                "inputs": inputs,
                "outputs": outputs,
            }
        )

        with metadata_path.open("wt") as fh:
            yaml.safe_dump(metadata, fh, indent=1, sort_keys=False)

        rich.print(f"Updated {metadata_path}")
        return metadata

    def _update_runtime_file():
        runtime_path = config_folder / "runtime.yml"
        runtime = {"settings": []}
        with suppress(FileNotFoundError):
            runtime = yaml.safe_load(runtime_path.read_text())

        delete = [
            item for item in runtime["settings"] if item["name"] == "ContainerSpec"
        ]
        for item in delete:
            runtime["settings"].remove(item)

        runtime["settings"].append(
            {
                "name": "ContainerSpec",
                "type": "ContainerSpec",
                "value": {
                    "Command": [
                        THIS_FILEPATH.name,
                        core_func.__name__,
                    ]
                },
            },
        )

        with runtime_path.open("wt") as fh:
            yaml.safe_dump(runtime, fh, indent=1, sort_keys=False)

        rich.print(f"Updated {runtime_path}")
        return runtime

    def _update_docker_compose_override_file():
        compose_specs_path = config_folder / "docker-compose.overwrite.yml"
        compose_specs = {
            "version": "3.7",
            "services": {
                f"{service_name}": {
                    "build": {
                        "dockerfile": "docker/{{ cookiecutter.docker_base.split(':')[0] }}/Dockerfile",
                        "target": "production",
                    },
                    # for debugging
                    "volumes": [
                        "./validation/input:/input:ro",
                        "./validation/output:/output:rw",
                    ],
                }
            },
        }

        with compose_specs_path.open("wt") as fh:
            yaml.safe_dump(compose_specs, fh, indent=1, sort_keys=False)

        rich.print(f"Updated {compose_specs_path}")
        return compose_specs

    _update_metadata_file()
    _update_runtime_file()
    _update_docker_compose_override_file()


def echo_jsonschema(core_func: Callable):
    vfunc = ValidatedFunction(function=core_func, config=None)
    assert vfunc.model  # nosec

    rich.print("json-schema for the inputs of {core_func.__name__}")
    rich.print(vfunc.model.schema_json(indent=1))


# ----------------------------------------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------------------------------------

run_logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(levelname)s [binder_program]: %(message)s", level=logging.INFO
)


class RuntimeSettings(BaseSettings):
    """Environment expected at runtime inside the container


    This interface is defined by the sidecar
    """

    INPUT_FOLDER: Path
    OUTPUT_FOLDER: Path
    LOG_FOLDER: Optional[Path] = None

    SC_BUILD_TARGET: Optional[str] = None
    SC_COMP_SERVICES_SCHEDULED_AS: Optional[str] = None
    SC_USER_ID: Optional[int] = None
    SC_USER_NAME: Optional[str] = None

    SIMCORE_MEMORY_BYTES_LIMIT: Optional[int] = None
    SIMCORE_NANO_CPUS_LIMIT: Optional[int] = None

    @validator("INPUT_FOLDER", "OUTPUT_FOLDER")
    @classmethod
    def check_dir_exists(cls, v):
        if v is None or not v.exists():
            raise ValueError(
                f"Folder {v} does not exists."
                "Expected predefined and created by sidecar"
            )
        return v

    @validator("INPUT_FOLDER")
    @classmethod
    def check_input_dir(cls, v):
        f = v / "inputs.json" if v else None
        if f is None or not f.exists():
            raise ValueError(
                f"File {f} does not exists."
                "Expected predefined and created by sidecar"
            )
        return v

    @validator("OUTPUT_FOLDER")
    @classmethod
    def check_output_dir(cls, v: Path):
        if not os.access(v, os.W_OK):
            raise ValueError(f"Do not have write access to {v}: {v.stat()}")
        return v

    @property
    def input_file(self) -> Path:
        return self.INPUT_FOLDER / "inputs.json"

    @property
    def output_file(self) -> Path:
        return self.OUTPUT_FOLDER / "outputs.json"


def run_service(core_func: Callable):
    # TODO: App class? with workflow embedded? split setup + run

    vfunc = ValidatedFunction(function=core_func, config=None)
    assert vfunc.model  # nosec

    # envs and inputs (setup by sidecar)
    try:
        settings = RuntimeSettings()  # captures  settings TODO: move
        run_logger.info("Settings setup by sidecar %s", settings.json(indent=1))

        inputs: BaseModel = vfunc.model.parse_file(settings.input_file)

    except json.JSONDecodeError as exc:
        assert settings
        raise ValueError(
            f"Invalid input file ({settings.input_file}) json format: {exc}"
        ) from exc

    except ValidationError as err:
        raise ValueError(f"Invalid inputs for {core_func.__name__}: {err}") from err

    # executes
    returned_values = vfunc.execute(inputs)

    # outputs (expected by sidecar)
    # TODO: verify outputs match with expected?
    # TODO: sync name
    if not isinstance(returned_values, tuple):
        returned_values = (returned_values,)

    outputs = {
        f"out_{index}": value for index, value in enumerate(returned_values, start=1)
    }
    settings.output_file.write_text(json.dumps(outputs))


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------


def create_group(core_func: Callable, settings: DotOsparcSettings) -> typer.Typer:
    app = typer.Typer(help=core_func.__doc__)

    @app.command()
    def run():
        """runs service"""
        run_service(core_func)

    @app.command()
    def config(
        dot_osparc_config: bool = True,
        jsonschema_inputs: bool = False,
    ):
        """echos configurations"""

        # TOOLING
        if jsonschema_inputs:
            echo_jsonschema(core_func)
            return

        # TOOLING
        elif dot_osparc_config:
            dump_dot_osparc_config(core_func, settings.metadata)
            return

    return app


def create_cli(expose: list[Callable], settings: DotOsparcSettings) -> typer.Typer:
    if not expose:
        raise ValueError(
            "No published functions could be exposed.\n"
            f"{TIP_PREFIX} Include path to the package in PYTHONPATH environment variable"
        )

    app = typer.Typer(
        name="osparc python function binder",
        help="Binds a service interface to a python function signature",
    )
    for func in expose:
        app.add_typer(create_group(func, settings), name=func.__name__)

    return app


def load_settings(settings_path: Path) -> DotOsparcSettings:
    settings = DotOsparcSettings.parse_raw(settings_path.read_text())

    if not settings.publish_functions:
        value = input(
            f"Initializing {settings_path} ... \n"
            "Name a function to expose (e.g. {{ cookiecutter.project_package_name }}.my_function): "
        )
        settings.publish_functions = [value]

        settings_path.write_text(settings.json(indent=1))
        run_logger.info(" %s updated %s", ALERT_PREFIX, settings_path)

    return settings


if __name__ == "__main__":
    try:
        settings = load_settings(settings_path=DOT_OSPARC_DIR / "settings.json")

        main = create_cli(
            expose=discover_published_functions(
                settings.publish_functions, dot_osparc_dir=DOT_OSPARC_DIR
            ),
            settings=settings,
        )
        main()
    except Exception as error:  # pylint: disable=broad-except
        run_logger.exception("Stopping application. %s", error)
        sys.exit(os.EX_SOFTWARE)
