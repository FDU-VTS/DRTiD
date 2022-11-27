# DRTiD

![](https://github.com/FDU-VTS/DRTiD/blob/main/src/intro.png)

We propose a new benchmark dataset, namely Diabetic Retinopathy Two-field image Dataset (DRTiD), consisting of 3,100 two-field fundus images. Two-field images contains a pair of macula-centric and optic disc-centric images. We provide annotations of DR & DME severity grades and localization of macula & optic disc.


### Related Papers


### Related Codes
[Source code of CrossFiT](https://github.com/FDU-VTS/DRTiD/)

### Now DRTiD is publicly available

### Get Password

To get the password of the compressed package, an application email must be sent to jlhou18@fudan.edu.cn (Junlin Hou) with a specified form like below, otherwise may be ignored.

**Title of Mail**:

DRTiD: your_organization: your_name

The string of 'DRTiD' can not be empty. It is the fixed form and a special sign we use to identifying your downloading intention from other disturbers like spams. The contents appending to DRTiD can help us identifying you more easily.

**Body of Mail**:

Organization Detail: Your Organization Details

Main Works: Your Main Works

Usages: YourUsages About This Data Set

### Dataset Structure

```
DRTiD
├── Original Images
└── Groundtruths
    ├── 1. DR_grade
    │   ├── a. DR_grade_Training.csv
    │   └── b. DR_grade_Testing.csv
    ├── 2. DME_grade
    │   ├── a. DME_grade_Training.csv
    │   └── b. DME_grade_Testing.csv
    └── 3. Localization
        ├── a. Localization_Training.csv
        └── b. Localization_Testing.csv
```

### Instructions

An DR open access dataset for research only.

By using the DRTiD dataset, you are obliged to reference the following paper:
