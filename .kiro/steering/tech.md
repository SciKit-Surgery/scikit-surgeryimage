# Tech Stack

## Language & Runtime

- Python (supports 2 and 3, CI targets 3.13)

## Build & Packaging

- setuptools with `setup.py`
- Versioneer for git-tag-based versioning (PEP 440, tag prefix `v`)
- Universal wheel (`bdist_wheel universal=1`)

## Core Dependencies

- numpy >= 2.0.0
- opencv-contrib-python-headless >= 4.10.0.84, < 4.13.0.90
- scikit-surgerycore >= 0.1.0

## Dev Dependencies

- pytest (test runner)
- tox (test orchestration)
- coverage / coveralls (code coverage)
- pylint < 3.1.0 (linting)
- sphinx + sphinx_rtd_theme (documentation)
- mock, pyfakefs, parameterized (test utilities)
- pyinstaller (packaging)

## Common Commands

```bash
# Run tests with coverage
tox -e test

# Run linting
tox -e lint

# Build docs
tox -e docs

# Run tests directly (without tox)
coverage run -a --source ./sksurgeryimage -m pytest -v -s ./tests/
coverage report -m

# Lint directly
pylint --rcfile=tests/pylintrc --ignore _version.py sksurgeryimage

# Install in development mode
pip install -e .
```

## Key Lint Rules

- Max line length: 100 characters
- Naming: snake_case for functions, methods, variables, arguments; PascalCase for classes; UPPER_CASE for constants
- Indent: 4 spaces
- Max function arguments: 5
- Max locals: 15
- Pylint must score 10/10 (fail-under=10)
- `cv2.*` members are whitelisted (generated-members)
- `numpy` and `cv2` are allowed as extension packages
