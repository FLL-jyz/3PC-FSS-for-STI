# Bundled FSS implementation

This directory contains the FSS implementation used by the 3PC protocols in
this repository. `include/` contains the public headers, `src/` contains the
library implementation, and `third_party/cryptoTools/` contains the required
cryptoTools sources and their license.

CMake builds this copy as the `fss` target. Eigen is read from the neighboring
EzPC checkout at `../EzPC/SCI/extern/eigen` by default; configure the project
with `-DEZPC_ROOT=/path/to/EzPC` when EzPC is located elsewhere. The build does
not update or write files in EzPC.

The FSS implementation is based on LLAMA's FSS code. See `LICENSE` and
`third_party/cryptoTools/LICENSE` for the respective license terms.
