#!/usr/bin/env bash

set -u

show_help() {
    echo
    echo "Usage:"
    echo "  ./scripts/run_three_party_logged.sh [test_name] [run_tag]"
    echo
    echo "Description:"
    echo "  Launch a 3-party local test with one log file per party."
    echo
    echo "Arguments:"
    echo "  test_name   Optional. Default: llm3pc"
    echo "  run_tag     Optional. Log directory suffix. Default: timestamp"
    echo
}

case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
TEST_NAME="${1:-llm3pc}"
RUN_TAG="${2:-$(date +%Y%m%d_%H%M%S)}"
BINARY="$BUILD_DIR/puma"
LOG_DIR="$BUILD_DIR/logs/${TEST_NAME}_${RUN_TAG}"
PID_FILE="$LOG_DIR/pids.txt"
SUMMARY_FILE="$LOG_DIR/summary.txt"

mkdir -p "$LOG_DIR"

if [[ ! -x "$BINARY" ]]; then
    echo "[ERROR] Missing executable: $BINARY"
    echo "[INFO] Please build first with: cmake -S . -B build && cmake --build build -j"
    exit 1
fi

echo "[INFO] Root directory: $ROOT_DIR"
echo "[INFO] Build directory: $BUILD_DIR"
echo "[INFO] Binary: $BINARY"
echo "[INFO] Logs: $LOG_DIR"

pkill -f "$BINARY $TEST_NAME" >/dev/null 2>&1 || true

pids=()
cleanup_needed=1

cleanup() {
    if [[ "$cleanup_needed" -eq 0 ]]; then
        return
    fi
    for pid in "${pids[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
}

trap cleanup INT TERM

for party_id in 0 1 2; do
    log_file="$LOG_DIR/party_${party_id}.log"
    echo "[INFO] Starting party $party_id -> $log_file"
    (
        cd "$BUILD_DIR" || exit 1
        exec "$BINARY" "$TEST_NAME" "$party_id"
    ) >"$log_file" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "party_${party_id}=$pid" >> "$PID_FILE"
    sleep 1
done

party_finished=(0 0 0)
party_status=(255 255 255)
failure_detected=0

while :; do
    all_done=1
    for index in 0 1 2; do
        pid="${pids[$index]}"
        if [[ "${party_finished[$index]}" -eq 1 ]]; then
            continue
        fi

        all_done=0
        if kill -0 "$pid" >/dev/null 2>&1; then
            continue
        fi

        if wait "$pid"; then
            party_status[$index]=0
        else
            party_status[$index]=$?
        fi
        party_finished[$index]=1

        if [[ "${party_status[$index]}" -ne 0 && "$failure_detected" -eq 0 ]]; then
            failure_detected=1
            echo "[WARN] Party $index exited with status ${party_status[$index]}; stopping remaining parties."
            for other_index in 0 1 2; do
                other_pid="${pids[$other_index]}"
                if [[ "$other_index" -ne "$index" ]] && [[ "${party_finished[$other_index]}" -eq 0 ]] && kill -0 "$other_pid" >/dev/null 2>&1; then
                    kill "$other_pid" >/dev/null 2>&1 || true
                fi
            done
        fi
    done

    if [[ "$all_done" -eq 1 ]]; then
        break
    fi
    sleep 1
done

cleanup_needed=0

{
    echo "test_name=$TEST_NAME"
    echo "run_tag=$RUN_TAG"
    echo "log_dir=$LOG_DIR"
    echo "party_0_status=${party_status[0]}"
    echo "party_1_status=${party_status[1]}"
    echo "party_2_status=${party_status[2]}"
} > "$SUMMARY_FILE"

echo "[INFO] Summary written to $SUMMARY_FILE"
for party_id in 0 1 2; do
    echo "[INFO] party $party_id exit status: ${party_status[$party_id]}"
done

if [[ "${party_status[0]}" -eq 0 && "${party_status[1]}" -eq 0 && "${party_status[2]}" -eq 0 ]]; then
    exit 0
fi

exit 1