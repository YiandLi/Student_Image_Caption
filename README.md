
Name: Yiin.Liu

Mail: Yliu8258@usc.edu

--- 

# Description
This task aims to prviddee actionable feedback to students' note images.

I solve it as an Image Caption Task and choose to fine tune [BLIP](https://github.com/salesforce/BLIP) on the give dataset.

## Model
BLIP has shwon great performance as a backbone on different multi-modal tasks. It separates encoding of images and text. The Image Encoder is a vision transformer(Vit). For the Image-grounded text decoder, it replaces the bi-directional self-attention layers in the text encoder with causal self-attention layers. A special [Decode] token is used to signal the beginning of a sequence.

When pre-training, BLIP jointly optimizes three objectives during pre-training, including two understanding-based objectives (ITC, ITM) and one generation-based objective (LM), on large-scale image-text datasets automatically collected from the web. 

When do fine tune for this task, I could direcly use the pre-trained image-caption module.

## Potential Drawbacks
1. This model is pretrained on a noisy web dataset, where the web texts often inaccurately describe the visual content of the images, resulting in a noisy source of supervision.

2. While the pre-trained model demonstrates high generality due to its training on a general-topic dataset, it lacks performance in this education-specific scenario.

3. The images in this task are occasionally blurry and are related to geometry and algebraic problems, leading to a potential lack of prior knowledge for the model.

4. The labeled descriptions in this task vary in length and mode (many of them share the common prefix "The student draws"/"x number lines"/...), which limits the complexity during fine-tuning, although it may ease the generation task.

## Improvements

1. For the model: Explore advanced pre-training techniques using domain-specific data related to education, which can enhance the model's performance in the education-specific scenario. So that we can incorporate transfer learnin, allowing the model to leverage task-specific knowledge and improve its performance on educational content.
For instance, one approach is to combine various tasks to train a multi-modal model with interleaved data. This approach enables the model to develop support for multi-modal chatting abilities.

2. For the data: Augment the dataset with high-quality, high-resolution images to minimize the impact of image blurriness and provide the model with clearer visual information.

3. Other solutions: Utilize a combination of rule-based and deep learning methods to improve the model's understanding of geometric and algebraic concepts, allowing it to generate more contextually relevant and accurate descriptions.

## Fine-tuned Checkpoint
The trained model is in the folder: `output/checkpoint_best.pth` of size 2.69 GB .

Or download with:
```
wget https://www.kaggleusercontent.com/kf/149532039/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..zE4geo0b-Ds2GtEfjWiRQA.SQocM1U6OLJEKxauiq42KPKtKVMaLtLFWDEYJSmlWbPLESYhdxmGHTI4QnuKGGx7hXJOBWtdinIUj-q3bdkQoBr09MFpTJiaWEel8YUmhwYYH2DCJOLHSWfsjMCiUUfEILJ5omel8C5EKmJdcdXzH6XQT1MtOa1jxcBN8bCw02HeVPmFP022Xa0cUYTJAgeZmF34q9P4jFFCR7FWAAtWpD9SnFoBWdLkZu_m0PBsOu47etasK39fdToTOeUZSIC3dROSu0sRZ2k79QoKcjSouOcVG36zw-FtliT0FluIz9Z9zuBrFpx_Z9efkLGJBivQXUQRUPxYCP9ZPuEQlWlfjRpq-LgwC7Zt-aHrKHsD_o2EKN42LnuQYVwgvNk_NFa3sZWbjv_98E9OFR1XMWMcaYkRs6kgw8o6QY6BQjn-Cs0vnpoct0e_y8q4GgwMPUOZXteADhBAvLKzeEHb8MTh2AhIvJjkTUrvwVAehhLi2WBlhP11qjClIpKDMDn_dVafZuntyRUxPTreSQrZdxvSCDAhAgwLJOJKqDNgTyw0VyALXPd8Qr0I2ZtwrAOfXlJnwn73p5U_PIsVjOY9e1Z_Q_PIzO6q23ry6YvqcdzyRgbkusc2xr9t6gISqWHFmG7Nbcv1__YY0UOxzOMKG-iYQ2gQ1BBq0BOeC7_k-wG0X0o.9SEUlFOjJtNTSK5q6ttXGw/Student_Image_Caption/output/checkpoint_best.pth
```


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

I use batchsize 8, epoch 50, random shuffle dataloader w/o bootstrap, to fine-tune vit-base combined with capfilt_large module,with metric BLEU-3, BLEU-4, ROUGE and CIDEr \
and choose the best checkpoiint with the highest $\text{BLEU-3} + \text{ ROUGE}$.


## Metric result

<img width="973" alt="image" src="https://github.com/YiandLi/Student_Image_Caption/assets/72687714/f1a9446e-8af6-4f33-b401-59a484f7899c">

| Metric   |      Start |      Max |
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

