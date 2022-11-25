# cookiecutter-osparc-service

![Build Status](https://github.com/ITISFoundation/cookiecutter-osparc-service/workflows/Github-CI%20Push/PR/badge.svg)

Cookiecutter to generate an oSparc compatible service for the oSparc simcore platform. Currently only for **computational services** supported.


## Requirements
- GNU Make
- Python3
- Python3-venv (recommended to work in a virtual environment)
- [``cookiecutter``](https://python-package-generator.readthedocs.io/en/master/)

```console
sudo apt-get update
sudo apt-get install -y make python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install cookiecutter
```

## Usage

Prepare a python environment (optional)

```console
python3 -m venv .venv
source .venv/bin/activate
```

and generate a new Cookiecutter template layout
```console
cookiecutter gh:ITISFoundation/cookiecutter-osparc-service
```

For more information, check [FAQ](./FAQ.md)


## Development

```console
git clone https://github.com/ITISFoundation/cookiecutter-osparc-service.git
cd cookiecutter-osparc-service
make devenv
source .venv/bin/activate
make play
make tests
```

## License

This project is licensed under the terms of the [MIT License](/LICENSE)


---
<p align="center">
<image src="https://github.com/ITISFoundation/osparc-simcore-python-client/blob/4e8b18494f3191d55f6692a6a605818aeeb83f95/docs/_media/mwl.png" alt="Made with love at www.z43.swiss" width="20%" />
</p>
