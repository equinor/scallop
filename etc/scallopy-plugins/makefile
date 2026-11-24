DIRECTORY = $(sort $(patsubst %/pyproject.toml,%,$(wildcard */pyproject.toml)))
DEVELOP_TARGETS = $(patsubst %,develop-%,$(DIRECTORY))
INSTALL_TARGETS = $(patsubst %,install-%,$(DIRECTORY))
BUILD_TARGETS = $(patsubst %,build-%,$(DIRECTORY))

.PHONY: develop install build clean $(DEVELOP_TARGETS) $(INSTALL_TARGETS) $(BUILD_TARGETS)

develop: $(DEVELOP_TARGETS)

$(DEVELOP_TARGETS): develop-%:
	cd $* && uv pip install --editable .

install: $(INSTALL_TARGETS)

$(INSTALL_TARGETS): install-%: build-%
	cd $* && uv pip install --reinstall-package scallop-$* dist/*.whl

build: $(BUILD_TARGETS)

$(BUILD_TARGETS): build-%:
	cd $* && uv build --clear

clean:
	rm -rf $(addsuffix /dist,$(DIRECTORY))
