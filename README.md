# requirement
```
transformers-4.34.1
fairscale-0.4.13
pycocoevalcap-1.2
```

# Start
1. dwnload the checkpoint
```python
mkdir checkpoint
wget -O checkpoint/model_base_caption_capfilt_large.pth https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth
```



2. run training process\
If you run the code for the 1st time, vit model checkpoint will be downloaded automatically.
```python
python src/train_caption.py
```




