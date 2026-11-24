.PHONY: build install develop test clean

develop:
	$(MAKE) -C .. develop-gpt

install:
	$(MAKE) -C .. install-gpt

build:
	$(MAKE) -C .. build-gpt

test:
	uv run --no-project python tests/test.py

clean:
	rm -rf dist
