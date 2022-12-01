# {{ cookiecutter.project_slug }}

{{ cookiecutter.project_short_description }}
{% if "python" in cookiecutter.docker_base %}

```console
$ make help

$ make config target=demo
$ make build target=demo
$ make shell target=demo
```
{% else %}
## Usage

```console
$ make help

$ make build
$ make info-build
$ make tests
```

## Workflow

1. The source code shall be copied to the [src]({{ cookiecutter.project_slug }}/src/{{ cookiecutter.project_package_name }}) folder.
1. The [Dockerfile]({{ cookiecutter.project_slug }}/src/Dockerfile) shall be modified to compile the source code.
2. The [.osparc](.osparc) is the configuration folder and source of truth for metadata: describes service info and expected inputs/outputs of the service.
3. The [execute]({{ cookiecutter.project_slug }}/service.cli/execute) shell script shall be modified to run the service using the expected inputs and retrieve the expected outputs.
4. The test input/output shall be copied to [validation]({{ cookiecutter.project_slug }}/validation).
5. The service docker image may be built and tested as ``make build tests`` (see usage above)
{%- endif %}
