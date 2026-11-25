.PHONY: develop-scallopy develop-scallopy-plus test clean

develop-scallopy:
	uv pip install --reinstall-package scallopy --editable .

develop-scallopy-plus:
	uv pip install --reinstall-package scallopy --editable . --config-setting build-args="--features torch-tensor"

test:
	uv run --no-project python tests/test.py

clean:
	rm -f scallopy/scallopy.cpython*
