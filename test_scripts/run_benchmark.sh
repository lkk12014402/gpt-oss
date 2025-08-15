set -ex

export OPENAI_API_KEY="empty"
export CUDA_VISIBLE_DEVICES=7

source /sdp/lkk/gpt-oss-20b/.venv/bin/activate

which python

function start_vllm() {
    local model_path=$1
    # vllm_server_gpt-oss-20b.log
    local model_name=$2
    local log_file=$3

    VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DISTRIBUTED_DEBUG=INFO vllm serve $model_path --port 8000 --served-model-name $model_name &> $log_file 2>&1 &
    
    SERVER_PID=$!
    
    # echo "vLLM server PID: $SERVER_PID"
    sleep 5s
    # echo "Waiting vllm ready"
    n=0
    until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
	n=$((n+1))
	if grep -q "Application startup complete" $log_file; then
	    break
	fi
	sleep 5s
    done

    echo $SERVER_PID

}

function benchmark() {
    local model_path=$1
    local task=$2
    local effort=$3
    cd /sdp/lkk/gpt-oss-20b/commits/gpt-oss
    
    /usr/bin/python -m gpt_oss.evals --model $model_path --eval $task --base-url http://localhost:8000/v1 --reasoning-effort $effort --output_dir /sdp/lkk/gpt-oss-20b/results/

}

function stop() {
    local pid=$1
    kill -9 $pid
}

function main() {
    # model_path="/sdp/lkk/gpt-oss-20b/gpt-oss-20b"
    # log_file="vllm_server_gpt-oss-20b.log"
    efforts=("low" "medium" "high")
    tasks=("mgsm" "mmlu" "gpqa" "aime25")

    # --------------------------------------------------------------------------
    model_path="/sdp/lkk/gpt-oss-20b/gpt-oss-20b"
    model_name="openai/gpt-oss-20b"
    log_file="vllm_server_gpt-oss-20b.log"
    pid=$(start_vllm "$model_path" "$model_name" "$log_file")
    echo "vLLM server PID: $pid"
    for task in "${tasks[@]}"; do
        for effort in "${efforts[@]}"; do
	    log_file="vllm_server_openai_gpt-oss-20b_$task_$effort.log"
	    # pid=$(start_vllm "$model_path" "$model_name" "$log_file")
	    benchmark $model_name $task $effort
	    # stop $pid
	    sleep 10
	done
    done
    stop $pid
    # --------------------------------------------------------------------------


}

main



exit
# sglang python3 -m sglang.launch_server --model ./gpt-oss-20b-bf16 --port 8009
#
model_path="/sdp/lkk/gpt-oss-20b/gpt-oss-20b-bf16"

/usr/bin/python -m sglang.launch_server --model $model_path --port 8000

cd /sdp/lkk/gpt-oss-20b/commits/gpt-oss

python -m gpt_oss.evals --model gpt-oss-20b --eval gpqa --base-url http://localhost:8003/v1 --reasoning-effort low
