## Instructions

1. Launch the batch inference [template](https://console.anyscale.com/v2/template-preview/batch-llm)
<img width="536" alt="image1" src="https://github.com/user-attachments/assets/c47183bb-2865-4adb-87ff-d46be3296f46">


2. Update the Head Node Type to the desired machine type
<img width="357" alt="image2" src="https://github.com/user-attachments/assets/cbb37e50-4040-4088-94f3-792f0f02551d">


| Model | Node Type |
| :---- | :---- |
| neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 | g6e.12xlarge (at least 2 L40S GPU) |
| neuralmagic/Meta-Llama-3.1-7B-Instruct-FP8 | g6e.xlarge (at least 1 L40S GPU)  |

3. Get the benchmark code to the workspace (Upload to the workspace directly by dragging to the File Explorer tab) [benchmark-blog](https://drive.google.com/drive/folders/1N7fS3VuroMK3-3eJzm__KV57evtnOrGg?usp=sharing)

4. Run the benchmark with 

```
cd benchmark-blog
bash run_70b.sh 
# bash run_8b.sh
```

## Model Configs

**neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8** 

```
model_id: neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
llm_engine: vllm
accelerator_type: L40S
engine_kwargs:
  tensor_parallel_size: 1
  pipeline_parallel_size: 2
  max_num_seqs: 128
  use_v2_block_manager: True
  enable_prefix_caching: False
  preemption_mode: "recompute"
  block_size: 16
  kv_cache_dtype: "auto"
  enforce_eager: True
  gpu_memory_utilization: 0.95
  enable_chunked_prefill: True
  max_num_batched_tokens: 256
  max_seq_len_to_capture: 32768
  dtype: float16
runtime_env:
  env_vars:
    vllm_configure_logging: 1
    vllm_use_ray_compiled_dag: 1
    vllm_use_ray_spmd_worker: 1
    vllm_use_ray_compiled_dag_nccl_channel: 1
    enable_anyscale_prefix_optimizations: "0"
    vllm_attn_backend: "FLASH_ATTN"
    vllm_disable_logprobs: 1
```

**neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 Model config:**

```
model_id: neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8
llm_engine: vllm
accelerator_type: L40S
engine_kwargs:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  max_num_seqs: 224
  use_v2_block_manager: True
  enable_prefix_caching: False
  preemption_mode: "recompute"
  block_size: 16
  kv_cache_dtype: "auto"
  enforce_eager: False
  gpu_memory_utilization: 0.95
  enable_chunked_prefill: True
  max_num_batched_tokens: 3072
  max_seq_len_to_capture: 32768
runtime_env:
  env_vars:
    VLLM_ATTENTION_BACKEND: "FLASH_ATTN"
    ENABLE_ANYSCALE_PREFIX_OPTIMIZATIONS: "0"
```
