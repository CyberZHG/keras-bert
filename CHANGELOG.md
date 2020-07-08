# Changelog

## [Unreleased]

## [0.85.0] - 2020-07-09

### Fixed

- Compatible with Keras 2.4.3

## [0.82.0] - 2020-06-02

### Removed

- Adapter

## [0.78.0] - 2019-09-17

### Fixed

- Compatible with Keras 2.3.0

## [0.70.0] - 2019-07-16

### Added

- Try to find the indices of tokens in the original text.

## [0.69.0] - 2019-07-16

### Added

- [Adapter](https://arxiv.org/pdf/1902.00751.pdf)

## [0.60.0] - 2019-06-10

### Added

- `trainable` can be a list of prefixes of layer names

## [0.58.0] - 2019-06-10

### Fixed

- Use `math_ops` for tensorflow backend
- Assign names to variables in warmup optimizer 

## [0.56.0] - 2019-06-04

### Changed

- Docs about `training` and `trainable`

### Fixed

- Missing `trainable=False` when `training=True`

## [0.54.0] - 2019-05-29

### Added

- Support eager mode with tensorflow backend

## [0.43.0] - 2019-05-12

### Added

- Support `tf.keras`

## [0.40.0] - 2019-04-29

### Added

- Warmup optimizer

## [Older Versions]

### Added

- BERT implementation
- Load official model
- Tokenizer
- Demos
