.PHONY: build install develop test clean

develop:
	$(MAKE) -C .. develop-gemini

install:
	$(MAKE) -C .. install-gemini

build:
	$(MAKE) -C .. build-gemini

test:
	uv run --no-project python tests/test.py

clean:
	rm -rf dist
