# DRTiD

![](https://github.com/FDU-VTS/DRTiD/blob/main/src/intro2.png)

We propose a new benchmark dataset, namely Diabetic Retinopathy Two-field image Dataset (DRTiD), consisting of 3,100 two-field fundus images. Two-field images contains a pair of macula-centric and optic disc-centric images. We provide annotations of DR severity grades and localization of macula & optic disc.


### Related Papers


### Related Codes
Source code of CrossFiT: [https://github.com/FDU-VTS/DRTiD/tree/main/CrossFiT](https://github.com/FDU-VTS/DRTiD/tree/main/CrossFiT)

## 📢📢📢 Now DRTiD is publicly available

### Get Password

To get the password of the compressed package, an application email must be sent to jlhou18@fudan.edu.cn (Junlin Hou) with a specified form like below, otherwise may be ignored.

**Title of Email**:

DRTiD: your_organization: your_name

The string of 'DRTiD' can not be empty. It is the fixed form and a special sign we use to identifying your downloading intention from other disturbers like spams. The contents appending to DRTiD can help us identifying you more easily.

**Body of Email**:

Organization Detail: Your Organization Details

Main Works: Your Main Works

Usages: Your Usages About This Data Set

Attachment: Please Download, Read, Sign an EULA ([GoogleDrive](https://drive.google.com/file/d/1X1o-uAwTwHajtwTfBK1X-Y97amvOgrQP/view?usp=sharing)), and Attach it To Your Email.

### Dataset Structure

```
DRTiD
├── Original Images
└── Groundtruths
    ├── DR_grade
    │   ├── a. DR_grade_Training.csv
    │   └── b. DR_grade_Testing.csv
    └── Optic_Macula_Localization
        └──op_ma_localization.csv

```

### Instructions

An DR open access dataset for research only.

By using the DRTiD dataset, you are obliged to reference the following paper:
```
@inproceedings{hou2022cross,
  title={Cross-Field Transformer for Diabetic Retinopathy Grading on Two-field Fundus Images},
  author={Hou, Junlin and Xu, Jilan and Xiao, Fan and Zhao, Rui-Wei and Zhang, Yuejie and Zou, Haidong and Lu, Lina and Xue, Wenwen and Feng, Rui},
  booktitle={2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={985--990},
  year={2022},
  organization={IEEE Computer Society}
}
```
