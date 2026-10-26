# 3PC FSS for Secure Transformer Inference (STI)

This is project focuses on secure three-party computation (3PC) for Transformer model inference based on our 3PC FSS.

## Directory Structure

*   **`application/`**: Neural network layer implementations (`nn`).
*   **`common/`**: Common utilities, including network communication, ring arithmetic, and math functions.
*   **`crypto/`**: Cryptographic protocols (Arithmetic Secret Sharing, Replicated Secret Sharing).
*   **`FSS/`**: The bundled Function Secret Sharing implementation used by the 3PC protocols. `FSS/include/` contains its public headers, `FSS/src/` contains the library implementation, and `FSS/third_party/cryptoTools/` contains the cryptoTools sources required to build it.
*   **`secure_model/`**: Secure 3PC specific logic and party management.
*   **`tests/`**: Test files for various models (CNN, LLM) and benchmarks.
*   **`external/`**: External dependencies (e.g., `cnpy`).
*   **`eval_bash.sh`**: Helper script for building and running the three-party tests.
*   **`main.cpp`**: The main entry point for the executable.
*   **`baseline/`**: Additional implementations used for comparison:
    *   **`baseline/Sigma/`**: Sigma implementation and evaluation code.
    *   **`baseline/Puma/`**: Puma baseline implementation and evaluation code.
    *   **`baseline/Shaft/`**: Shaft implementation and evaluation code.

Each baseline is self-contained and has its own `README.md` with build and
evaluation instructions. Build a baseline from its own directory rather than
from the repository root.

## Requirements

Running this project requires a Linux environment with the following dependencies:

*   **Compiler**: g++ (C++17 compatible)
*   **Build System**: CMake ($\ge$ 3.16), make
*   **Libraries**:
    *   OpenMP
    *   OpenSSL
    *   [cnpy](https://github.com/rogersce/cnpy) (Integrated in `external/`)
    *   EzPC checkout in `../EzPC` by default, with `SCI/extern/eigen` available

The FSS implementation and cryptoTools sources are included in `FSS/` and are built with this project. You do not need to copy or modify `FSS/` inside EzPC. CMake reads Eigen from the adjacent EzPC checkout and does not modify EzPC source files. Set `EZPC_ROOT` if your EzPC checkout is elsewhere.

To install dependencies on Ubuntu:
```bash
sudo apt update
sudo apt install build-essential cmake libssl-dev libomp-dev
```

## Compilation

### EzPC and FSS configuration

Place the EzPC checkout next to this repository so that the directory layout is:

```text
parent-directory/
├── EzPC/
└── 3PC-FSS-for-STI/
```

The EzPC checkout must contain `SCI/extern/eigen`. If the EzPC submodules have
not been initialized yet, initialize them from the EzPC directory:

```bash
cd ../EzPC
git submodule update --init --recursive
cd ../3PC-FSS-for-STI
```

The root `CMakeLists.txt` adds the bundled `FSS/` directory and builds the
`fss` static library from source. By default it looks for Eigen at
`../EzPC/SCI/extern/eigen`. For another EzPC location, pass an absolute or
relative path with `-DEZPC_ROOT`:

```bash
cmake -S . -B build -DEZPC_ROOT=/path/to/EzPC
```

No prebuilt `libfss.a` is required, and no source file under EzPC needs to be
edited.

You can build the project manually using standard CMake commands:

```bash
cmake -S . -B build -DEZPC_ROOT=../EzPC
cmake --build build -j
```

This will generate the `Triforce` executable in the `build/` directory.

## Running Tests

You can use the provided script `eval_bash.sh` to build and run tests easily, or run the executable manually.

### Using `eval_bash.sh`

The script automatically handles the build process and can launch all three parties for local testing.

**Usage:**
```bash
sh eval_bash.sh <test_name> [party_id]
```

*   **`<test_name>`**: The name of the test to run (see supported tests below).
*   **`[party_id]`** (Optional): The ID of the party (0, 1, or 2).
    *   If specified, it runs a single party instance.
    *   If omitted, it launches all 3 parties locally in the background.

**Examples:**

1.  **Run Microbenchmarks (All 3 parties):**
    ```bash
    sh eval_bash.sh bench
    ```

2.  **Run RSS Test for Party 0:**
    ```bash
    sh eval_bash.sh rss 0
    ```

### Manual Execution

If you prefer to run manually after building:

```bash
# General syntax
./build/Triforce <test_name> <party_id>

# Example: Running locally (requires 3 terminals)
./build/Triforce bench 0
./build/Triforce bench 1
./build/Triforce bench 2
```

## Supported Tests

The `Triforce` executable supports the following `test_name` arguments:

| Test Name | Description |
|-----------|-------------|
| `rss` | Evaluation of Replicated Secret Sharing (RSS) protocols. |
| `bench` | Microbenchmarking of core secure operations (ReLU, MatMul, Softmax, etc.). |
| `cnn3pc` | Secure inference for CNN models (e.g., AlexNet, ResNet50). |
| `llm3pc` | Secure inference for Transformer models (BERT, GPT2). |
| `llmacc` | Accuracy evaluation for LLMs. |
| `offline_cnn3pc` | Offline phase testing for CNNs. |
| `offline_llm3pc` | Offline phase testing for LLMs. |

## Data Preparation

For inference tests (`llmacc`, `llm3pc`, etc.), you need to prepare model and data shares.

1.  Use `tests/pt2npz.py` to convert PyTorch models and data into shared format.
2.  Shares should be placed in:
    *   `data/model_shares/`
    *   `data/data_shares/`
