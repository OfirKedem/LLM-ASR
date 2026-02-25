#!/usr/bin/env bash
set -euo pipefail

YAML_PATH="sweep.yaml"

usage() {
  echo "Usage:"
  echo "  $0 create                    # create sweep from ${YAML_PATH}"
  echo "  $0 agent SWEEP_ID            # launch single wandb agent for given sweep ID"
  echo "  $0 agents SWEEP_ID [GPUS]    # launch one agent per GPU (default: \$CUDA_VISIBLE_DEVICES or 0)"
  echo ""
  echo "Examples:"
  echo "  SWEEP_ID=ofirkedem/speach-project/abc123 $0 agents"
  echo "  $0 agents ofirkedem/speach-project/abc123 0,1,2,3"
}

if [[ "${1:-}" == "create" ]]; then
  wandb sweep "${YAML_PATH}"
elif [[ "${1:-}" == "agent" ]]; then
  if [[ $# -lt 2 ]]; then
    usage
    exit 1
  fi
  SWEEP_ID="$2"
  wandb agent "${SWEEP_ID}"
elif [[ "${1:-}" == "agents" ]]; then
  SWEEP_ID="${2:-${SWEEP_ID:-}}"
  GPU_LIST="${3:-${CUDA_VISIBLE_DEVICES:-0}}"
  if [[ -z "${SWEEP_ID}" ]]; then
    echo "Error: SWEEP_ID required (argument or env SWEEP_ID)" >&2
    usage
    exit 1
  fi

  IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
  NUM_GPUS=${#GPU_ARRAY[@]}

  echo "========================================="
  echo "Multi-GPU W&B Sweep Agent Launcher"
  echo "========================================="
  echo "Sweep ID: $SWEEP_ID"
  echo "GPU IDs: $GPU_LIST"
  echo "Number of agents: $NUM_GPUS"
  echo "Stdout/stderr: discarded (logs in wandb UI)"
  echo ""

  PIDS=()
  for gpu_id in "${GPU_ARRAY[@]}"; do
    gpu_id="${gpu_id// /}"
    echo "Launching agent on GPU $gpu_id (CUDA_VISIBLE_DEVICES=$gpu_id)..."
    CUDA_VISIBLE_DEVICES="$gpu_id" nohup wandb agent "${SWEEP_ID}" > /dev/null 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "  GPU $gpu_id: PID $pid"
    sleep 1
  done

  echo ""
  echo "========================================="
  echo "All agents launched!"
  echo "========================================="
  echo ""
  echo "Monitor progress: wandb UI (sweep runs)"
  echo ""
  echo "Check running agents:"
  echo "  ps aux | grep 'wandb agent'"
  echo ""
  echo "Kill all agents (if needed):"
  echo "  pkill -f 'wandb agent'"
  echo ""
else
  usage
  exit 1
fi

