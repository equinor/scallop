.PHONY: build install develop test clean

develop:
	$(MAKE) -C .. develop-codeql

install:
	$(MAKE) -C .. install-codeql

build:
	$(MAKE) -C .. build-codeql

test:
	uv run --no-project python tests/test.py

clean:
	rm -rf dist
