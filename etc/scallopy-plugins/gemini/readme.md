# Scallop Gemini

This implementation is entirely based on the Scallop GPT Plugin code.
The main goal of this adaptation is to make the use for small tasks more accessible.

## Installation

To install from source, please use the following command:

``` bash
$ make -C etc/scallopy-plugins install-gemini
```

For developers, you can do

``` bash
$ make -C etc/scallopy-plugins develop-gemini
```

As such, you can edit in-source and have the changes reflected in your execution.

## Usage

The Scallop Gemini plugin provides the following constructs:

- (Foreign Attribute) `@gemini_extract_info`
- (Foreign Attribute) `@gemini_encoder`
- (Foreign Predicate) `gemini(bound input: String, output: String)`
- (Foreign Function) `$gemini(String) -> String`

### Foreign Attribute `@gemini_extract_info`

This attribute uses structured JSON output based on the relation's output fields.
If you override the default Gemini model, choose one that supports structured output.

### Foreign Attribute `@gemini_encoder`

Text-only features do not require PyTorch. For embeddings, use a Scallopy
build with tensor support and install `uv pip install './etc/scallopy-plugins/gemini[embeddings]'`.
The encoder uses the CPU unless the optional `scallop-gpu` plugin selects a GPU.

The encoder uses `gemini-embedding-001` by default and returns a 768-element
embedding tensor. Embeddings from different models are not interchangeable.

### Foreign Predicate `gemini`

### Foreign Function `$gemini`
