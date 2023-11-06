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



# Result
## Configuration

I use batchsize 8, epoch 50, random shuffle dataloader w/o bootstrap, with metric BLEU-3, BLEU-4, ROUGE and CIDEr \
and choose the best checkpoiint with the highest $\text{BLEU-3} + \text{ ROUGE}$.

<img width="973" alt="image" src="https://github.com/YiandLi/Student_Image_Caption/assets/72687714/f1a9446e-8af6-4f33-b401-59a484f7899c">

| Metric   |      Min |      Max |
|:---------|---------:|---------:|
| BLEU-3   | 0.048443 | 0.319001 |
| BLEU-4   | 0.028146 | 0.286257 |
| ROUGE    | 0.165428 | 0.392557 |
| CIDEr    | 0.139804 | 1.24426  |

## Case study

You can download the generated caption result with :
```
wget https://www.kaggleusercontent.com/kf/149532039/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Ez4BLOrBONnDKR93O4VamA.HAz1zq_fXRQwZXpLvO3LmBmSHRZgkvL6I-sHeq4G3VNoLoqSBrH1UO-qryp8ASSUxfYZNVvq7vof3OZ3ogUMGK_bLifkv9g-hBMJEb885xb1dLmQsljhDqgvmhLREXMUCSSFINv6iE0fS2YnpPoanpgSxB-KgXe93woBSS-N_Dgj-LDx9r6MDAF8K5O1xoCNr8T1CUJt73kLGoZD24JysEklRCxjq9_4gUB0asmaM72sIFyKsRMGdROLXaVrIqJigQRHRHMLCp_ciM-Ib_nCiWDWEOaDRsOSj6lbK9hnuKcepK7UOhmr2RkA49EZeT3FpJV4zolacfEpBRZsKvmYSZsHZv_enSq0MhW8DMyVXZGwD0pf18zdB1Bnm1z1ZJfOdfFpnW902Bo7C3ShpC_ob--8zyi9UAtUMweYSn8IUPDoYILpsLY1yk9ntVaTyEYoDYKrGV3JItIvDhJdM6yiE1oGFFAZCVY1mhPNWEq20Vd1FxBZq1X55Tl6qUwENYUVpwgteGVUVkihEn0njEI7W9b_Orc29ixRvTclUqt_xHN5R2P12ZYz7FEShGIGgi7j2fJ_uEdrKn22U0QUZ3klnaaVMtuEtJQP1n3j-M7-xpw6mHmoCiRcn3QXBu4adj0_79zgxGJOu8PzVaL4qSDPgwrH2WN5PzuyqLV2SEg8fa0.kCvQMf842RWhGYvNHOJc7Q/Student_Image_Caption/output/result.txt
```

