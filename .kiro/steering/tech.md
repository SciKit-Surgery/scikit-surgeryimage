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

The tox `test` environment creates a virtualenv at `.tox/test/` with all dependencies installed. To run tests directly without re-creating the venv, use `.tox/test/bin/python3`:

```bash
# Run tests with coverage (full tox, creates/reuses venv)
tox -e test

# Run linting (full tox)
tox -e lint

# Build docs
tox -e docs

# Run tests directly using the tox venv (fast, skips venv setup)
.tox/test/bin/python3 -m pytest -v -s ./tests/

# Run a specific test file
.tox/test/bin/python3 -m pytest -v -s ./tests/calibration/test_dotty_grid_point_detector.py

# Lint directly using the tox venv
.tox/test/bin/python3 -m pylint --rcfile=tests/pylintrc --ignore _version.py sksurgeryimage

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
