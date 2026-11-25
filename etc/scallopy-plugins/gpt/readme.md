# Scallop GPT

## Installation

To install from source, please use the following command:

``` bash
$ make -C etc/scallopy-plugins install-gpt
```

For developers, you can do

``` bash
$ make -C etc/scallopy-plugins develop-gpt
```

As such, you can edit in-source and have the changes reflected in your execution.

## Usage

The default chat model is `gpt-5.4-mini`. Override it with `--openai-gpt-model`.

The Scallop GPT plugin provides the following constructs:

- (Foreign Attribute) `@gpt`
- (Foreign Attribute) `@gpt_extract_info`
- (Foreign Attribute) `@gpt_encoder`
- (Foreign Predicate) `gpt(bound input: String, output: String)`
- (Foreign Function) `$gpt(String) -> String`

### Foreign Attribute `@gpt`

### Foreign Attribute `@gpt_extract_info`

The default model and its snapshots use strict JSON schemas based on the
relation's output fields. Other explicitly selected models keep JSON mode.

### Foreign Attribute `@gpt_encoder`

Text-only features do not require PyTorch. For embeddings, use a Scallopy
build with tensor support and install `uv pip install './etc/scallopy-plugins/gpt[embeddings]'`.
The encoder uses the CPU unless the optional `scallop-gpu` plugin selects a GPU.

### Foreign Predicate `gpt`

### Foreign Function `$gpt`
