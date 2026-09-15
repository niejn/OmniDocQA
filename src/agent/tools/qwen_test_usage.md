 """
    modelscope download --model Qwen/Qwen3-Reranker-4B README.md --local_dir ./dir
    
    modelscope download `
    --model Qwen/Qwen3-Reranker-8B `
    --local_dir C:\Users\julien\.cache\huggingface\hub\Qwen3-Reranker-8B
    
    modelscope download --model Qwen/Qwen3-Reranker-8B README.md --local_dir ./dir
    
    python src/agent/tools/qwen_reranker.py `
    --model C:\Users\julien\.cache\huggingface\hub\Qwen3-Reranker-8B `
    --batch-size 1 `
    --max-length 4096
    """