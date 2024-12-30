from huggingface_hub import hf_hub_download

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置为hf的国内镜像网站

from huggingface_hub import snapshot_download

model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# while True 是为了防止断联
while True:
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=True,  # 在local-dir指定的目录中都是一些“链接文件”
            ignore_patterns=["*.bin"],  # 忽略下载哪些文件
            local_dir=model_name,
            token="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",   # huggingface的token
            resume_download=True
        )
        break
    except:
        pass

