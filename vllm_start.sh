
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup  vllm serve /cpfs04/user/zhangyaoze/vllm_weight/bge-m3 --served-model-name bge --port 8000 --max-model-len 8192 --served-model-name bge --gpu-memory-utilization 0.1 >logs1/embed.log 2>&1 &
# if [ -z "$1" ]; then
#   echo "Usage: $0 <num_gpus>"
#   exit 1
# fi

# for ((i=0;i<$1;i++)); do
#   (
#     export CUDA_VISIBLE_DEVICES=$i
#     nohup vllm serve /cpfs04/user/zhangyaoze/vllm_weight/Qwen3-14b --served-model-name Qwen3-14b\
#       --port $((8001 + i))  --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' --max-model-len 65536  > logs1/qwen3_14b_$i.log 2>&1 &
#   )
# done
# CUDA_VISIBLE_DEVICES=0,1 nohup  vllm serve /cpfs04/user/zhangyaoze/vllm_weight/Qwen3-32b --served-model-name Qwen3-32b --port 8001 --tensor-parallel-size 2   >q32.log 2>&1 &
# --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072
CUDA_VISIBLE_DEVICES=0,1 nohup  vllm serve /root/Qwen2.5 --served-model-name qwen2.5 --port 8001 >logs1/qwen2.5.log --tensor-parallel-size 2 2>&1 &