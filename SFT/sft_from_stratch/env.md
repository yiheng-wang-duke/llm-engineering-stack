
### Create new environment
```
conda create -n llm_sft python=3.12
conda activate llm_sft
```


Although other library versions might work as well, it's strongly recommended to install libraries with the same version, as this environment is verified.
### Install torch
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### Install transformers
```
pip install transformers==5.7.0
```

### Install accelerate
```
pip install accelerate==1.13.0
```

### Install flash attention
```
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
```

```
deepspeed-0.19.1
```