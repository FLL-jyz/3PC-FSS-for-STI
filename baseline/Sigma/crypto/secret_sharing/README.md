# secret_sharing

## Offline Triple Files (2PC)

The 2PC multiplication path in `twopc_protocols.h` supports loading Beaver triples from
preprocessed files instead of generating triples online.

### Environment Variables

- `SIGMA_OFFLINE_TRIPLE_EW_FILE`: element-wise multiplication triples for `mul(...)`
- `SIGMA_OFFLINE_TRIPLE_MM_FILE`: matrix multiplication triples for `matMul(...)`

If an env var is missing, execution throws an exception.

### File Format

Each triple request consumes three tensors in sequence:

1. tensor `a`
2. tensor `b`
3. tensor `c`

All values are parsed as whitespace-separated integers from text. Newlines are optional;
spaces/tabs/newlines are all valid separators.

For a tensor with `N` elements, one request consumes `3 * N` numbers.

### Sequential Consumption

The loader maintains per-file read offsets inside the process:

- first call reads the first triple in the file
- next call continues from where the previous call ended

This allows a single file to hold many triples for repeated protocol calls.

### Example

For one `mul` call with shape `[2]`, file content can be:

`1 2   3 4   5 6`

which maps to:

- `a = [1, 2]`
- `b = [3, 4]`
- `c = [5, 6]`
