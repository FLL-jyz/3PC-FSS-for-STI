# Triforce

TFssformer is a project focused on maliciously secure three-party computation (3PC) for Transformer model inference.

## Directory Structure

*   **`application/`**: Neural network layer implementations (`nn`).
*   **`common/`**: Common utilities, including network communication, ring arithmetic, and math functions.
*   **`crypto/`**: Cryptographic protocols (Arithmetic Secret Sharing, Replicated Secret Sharing).
*   **`secure_model/`**: Secure 3PC specific logic and party management.
*   **`tests/`**: Test files for various models (CNN, LLM) and benchmarks.
*   **`external/`**: External dependencies (e.g., `cnpy`).
*   **`script/`**: Contains helper scripts like `eval_bash.sh`. 
*   **`main.cpp`**: The main entry point for the executable.

## Requirements

Running this project requires a Linux environment with the following dependencies:

*   **Compiler**: g++ (C++17 compatible)
*   **Build System**: CMake ($\ge$ 3.16), make
*   **Libraries**:
    *   OpenMP
    *   OpenSSL
    *   [cnpy](https://github.com/rogersce/cnpy) (Integrated in `external/`)

To install dependencies on Ubuntu:
```bash
sudo apt update
sudo apt install build-essential cmake libssl-dev libomp-dev
```

## Compilation

You can build the project manually using standard CMake commands:

```bash
mkdir build
cd build
cmake ..
make -j
```

This will generate the `TFssformer` executable in the `build/` directory.

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
./build/TFssformer <test_name> <party_id>

# Example: Running locally (requires 3 terminals)
./build/TFssformer bench 0
./build/TFssformer bench 1
./build/TFssformer bench 2
```

## Supported Tests

The `TFssformer` executable supports the following `test_name` arguments:

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


