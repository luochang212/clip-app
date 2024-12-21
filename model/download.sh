conda install pytorch -y
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese --local-dir ./Taiyi-CLIP-Roberta-102M-Chinese
huggingface-cli download --resume-download openai/clip-vit-base-patch32 --local-dir ./clip-vit-base-patch32