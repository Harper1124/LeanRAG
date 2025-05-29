export CUDA_VISIBLE_DEVICES=0,1

vllm serve /mnt/workspace/workspace/models/Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --served-model-name Qwen2.5-7B --host 0.0.0.0 --port 8002 --max-model-len 16384 --uvicorn-log-level debug --disable-log-requests