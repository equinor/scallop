.PHONY: build install develop clean

build:
	uv build --clear

install: build
	uv pip install --reinstall-package scallop dist/*.whl

develop:
	uv pip install --editable .

clean:
	rm -rf dist
