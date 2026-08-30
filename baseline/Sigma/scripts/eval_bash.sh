show_help() {
    echo
    echo "Usage:"
    echo "  sh eval_bash.sh <test_name> [party_id]"
    echo
    echo "Description:"
    echo "  This script builds and optionally runs the Sigma tests."
    echo
    echo "Arguments:"
    echo "  <test_name>     Required unless only building. Choose one of:"
    echo "      twopc       Evaluation of two-party secret sharing options"
    echo "      bench       Microbenchmark of core secure ops"
    echo "      cnn2pc      Secure inference for CNN models (AlexNet, ResNet50)"
    echo "      cnn2pc      2PC experiment mode for CNN models"
    echo "      llm2pc      Secure inference for Transformer models"
    echo "      llm2pc      2PC experiment mode for Transformer models"
    echo "      llmacc      Accuracy evaluation for BERT/GPT2"
    echo "      offline_cnn2pc   2PC offline communication simulation for CNN"
    echo "      offline_llm2pc   2PC offline communication simulation for LLM"
    echo "      -h, --help  Show this help message and exit"
    echo
    echo "  [party_id]      Optional. Choose 0, 1, or 2 to run a specific party"
    echo "                  Leave blank to run all 3 parties locally (default)"
    echo
    echo "Behavior:"
    echo "  - If <test_name> is not specified, only the project will be built and no tests will be executed."
    echo
}

# === Check for help flags ===
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
esac

# === Setup ===
export OMP_NUM_THREADS=1

cd ..

if [ ! -d "build" ]; then
    mkdir build
fi

cd ./build || { echo "[ERROR] Failed to enter build directory"; exit 1; }

# 运行 cmake 和 make
cmake .. || { echo "[ERROR] cmake failed"; exit 1; }

make -j || { echo "[ERROR] make failed"; exit 1; }

if [ $# -eq 1 ]; then
    echo "[INFO] Running test '$1' for 2 parties..."
    ./Sigma $1 0 &
    ./Sigma $1 1 &

elif [ $# -gt 1 ]; then
  echo "[INFO] Running test '$1' for party $2..."
  ./Sigma $1 $2
fi

wait

echo "[INFO] Completed."
