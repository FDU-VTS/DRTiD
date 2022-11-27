# Cross-Field Transformer for Two-field Fundus Images

This repo covers the source code of paper Cross-Field Transformer for Two-field Fundus Images (BIBM 2022).
![](https://github.com/FDU-VTS/DRTiD/blob/main/src/CrossFiT.png)


## Dataset
Download the Diabetic Retinopathy Two-field image Dataset (DRTiD) from [https://github.com/FDU-VTS/DRTiD](https://github.com/FDU-VTS/DRTiD)

## Train

### Single-field methods
field 1 / field 2 / field 1&2
```
python main_base.py --pretrained True --dataset drtid --fusion_category single --fusion_type 1 --visname drtid_single_1
python main_base.py --pretrained True --dataset drtid --fusion_category single --fusion_type 2 --visname drtid_single_2
python main_base.py --pretrained True --dataset drtid --fusion_category single --fusion_type 3 --visname drtid_single_3
```


### Two-field methods
1. feature-level fusion (average pooling / max pooling / concatenation)
```
python main_base.py --pretrained True --dataset drtid --fusion_category fusion2 --fusion_type avg --visname drtid_fusion2_avg
python main_base.py --pretrained True --dataset drtid --fusion_category fusion2 --fusion_type max --visname drtid_fusion2_max
python main_base.py --pretrained True --dataset drtid --fusion_category fusion2 --fusion_type cat --visname drtid_fusion2_cat
```

2. decision-level fusion (average / max)
```
python main_base.py --pretrained True --dataset drtid --fusion_category fusion3 --fusion_type avg --visname drtid_fusion3_avg
python main_base.py --pretrained True --dataset drtid --fusion_category fusion3 --fusion_type max --visname drtid_fusion3_max
```


### CrossFiT
CrossFiT with 3-layer CFA, hidden size=1024, max pooling, APE, FAM threshold p=0.06
```
python main_crossfit.py --pretrained True --dataset drtid --xfmer_hidden_size 1024 --xfmer_layer 3 --pool max --p_threshold 0.06 --visname drtid_res50_l3_d1024_p0.06_APE_max
```


