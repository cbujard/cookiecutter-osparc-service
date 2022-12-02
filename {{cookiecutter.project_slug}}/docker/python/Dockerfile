FROM {{ cookiecutter.docker_base }} as base
#
#  USAGE:
#     cd services/{{ cookiecutter.project_slug }}
#     docker build -f Dockerfile -t {{ cookiecutter.project_slug }}:prod --target production ../../
#     docker run {{ cookiecutter.project_slug }}:prod
#

LABEL maintainer={{ cookiecutter.git_username }}

# non-root user
ENV SC_USER_ID 8004
ENV SC_USER_NAME scu
RUN adduser \
  --uid ${SC_USER_ID}\
  --disabled-password\
  --gecos "" \
  --shell /bin/sh \
  --home /home/${SC_USER_NAME}\
  ${SC_USER_NAME}


RUN apt-get update \
  && apt-get -y install --no-install-recommends \
  jq \
  && rm --recursive --force /var/lib/apt/lists/*

RUN python -m pip --no-cache install --upgrade pip wheel setuptools


# --------------------------Production stage -------------------
# Final cleanup up to reduce image size and startup setup
# Runs as ${SC_USER_NAME} (non-root user)
#
#  + /home/${SC_USER_NAME}     $HOME = WORKDIR
#    + .osparc [${SC_USER_NAME}:${SC_USER_NAME}]
#    + docker  [${SC_USER_NAME}:${SC_USER_NAME}]
#    + {{ cookiecutter.project_package_name }} [${SC_USER_NAME}:${SC_USER_NAME}]
#
FROM base as production

ENV SC_BUILD_TARGET production
ENV SC_BOOT_MODE production

{% if cookiecutter.project_type == "computational" %}
ENV INPUT_FOLDER="/input" \
  OUTPUT_FOLDER="/output"
{% elif cookiecutter.project_type == "dynamic" %}
{# will not be necessary when the dy-sidecar comes in #}
ENV SIMCORE_NODE_UUID="-1" \
  SIMCORE_USER_ID="-1" \
  SIMCORE_NODE_BASEPATH="" \
  STORAGE_ENDPOINT="=1" \
  POSTGRES_ENDPOINT="-1" \
  POSTGRES_USER="-1" \
  POSTGRES_PASSWORD="-1" \
  POSTGRES_DB="-1"
{%- endif %}

WORKDIR /home/${SC_USER_NAME}

# uncomment and adapt to install runtime dependencies
RUN apt-get update \
  && apt-get -y install --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# copy docker bootup scripts
COPY --chown=${SC_USER_NAME}:${SC_USER_NAME} \
  docker/{{ cookiecutter.docker_base.split(":")[0] }}/*.sh \
  docker/

{%- if cookiecutter.project_type == "computational" %}
# copy .osparc folder
COPY --chown=${SC_USER_NAME}:${SC_USER_NAME} \
  .osparc/ \
  .osparc/

# install .osparc/requirements
RUN python -m pip --no-cache install -r .osparc/requirements.txt


# necessary to be able to call run directly without sh in front
ENV PATH="/home/${SC_USER_NAME}/.osparc/bin:${PATH}"
{%- endif %}

# copy binaries from build
COPY --chown=${SC_USER_NAME}:${SC_USER_NAME} \
  src \
  src


ENTRYPOINT [ "/bin/sh", "docker/entrypoint.sh" ]
