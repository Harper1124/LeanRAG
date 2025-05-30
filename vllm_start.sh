
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup  vllm serve /cpfs04/user/zhangyaoze/vllm_weight/bge-m3 --served-model-name bge --port 8000 --max-model-len 8192 --served-model-name bge --gpu-memory-utilization 0.1 >logs/embed.log 2>&1 &
if [ -z "$1" ]; then
  echo "Usage: $0 <num_gpus>"
  exit 1
fi

for ((i=0;i<$1;i++)); do
  (
    export CUDA_VISIBLE_DEVICES=$i
    nohup vllm serve /cpfs04/user/zhangyaoze/vllm_weight/Qwen3-14b --served-model-name Qwen3-14b\
      --port $((8001 + i)) --max-model-len 32768 --served-model-name Qwen3-14b > logs/qwen3_14b_$i.log 2>&1 &
  )
done

