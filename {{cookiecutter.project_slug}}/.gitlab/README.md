# Gitlab

add the following in your __gitlab-ci.yml__ file:

```yaml
include:
  - local: '/services/{{ cookiecutter.project_slug }}/ci/gitlab-ci.yml'
```
