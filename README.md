# DRTiD

![](https://github.com/FDU-VTS/DRTiD/blob/main/src/intro2.png)

We propose a new benchmark dataset, namely Diabetic Retinopathy Two-field image Dataset (DRTiD), consisting of 3,100 two-field fundus images. Two-field images contains a pair of macula-centric and optic disc-centric images. We provide annotations of DR severity grades and localization of macula & optic disc.


### Related Paper
Junlin Hou, Jilan Xu, Fan Xiao, Rui-Wei Zhao, Yuejie Zhang, Haidong Zou, Lina Lu, Wenwen Xue, Rui Feng. Cross-Field Transformer for Diabetic Retinopathy Grading on Two-field Fundus Images. 2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE Computer Society, 2022: 985-990. [[paper](https://ieeexplore.ieee.org/abstract/document/9995459)][[arxiv](https://arxiv.org/abs/2211.14552)]

### Related Code
Source code of CrossFiT: [https://github.com/FDU-VTS/DRTiD/tree/main/CrossFiT](https://github.com/FDU-VTS/DRTiD/tree/main/CrossFiT)

### How to get DRTiD?

Fill in the information in the application form [wenjuanxing (问卷星)](https://www.wjx.cn/vm/ex4eUEY.aspx#) or [GoogleForm](https://docs.google.com/forms/d/e/1FAIpQLSfEYIwk5G1Y6sAcDEjkt7qmv5FWcdG9Jn_P-gSnZ77MWvRM3A/viewform?usp=sf_link) and the download link will be feedbacked after accurate filling.

<!--
### Download Dataset

Please download the DRTiD dataset from [Google](https://drive.google.com/file/d/1p9GBaXcq65rBNkY9Mou8zbzF0RIsLo-6/view?usp=share_link) or [Baidu](https://pan.baidu.com/s/12b11WYXMOKMl1POZUImoTg?pwd=4a54).

### Get password

To get the password, an application email must be sent to jlhou18@fudan.edu.cn (Junlin Hou) with a specified form like below.

**Title of Email**:

DRTiD: your_organization: your_name


The string of 'DRTiD' can not be empty. It is the fixed form and a special sign we use to identifying your downloading intention from other disturbers like spams. The contents appending to DRTiD can help us identifying you more easily.


**Body of Email**:


1. Organization Detail: Your Organization Details

2. Main Works: Your Main Works


1. Usages: Your Usages About This Data Set

2. **Attachment: Please Download ([GoogleDrive](https://drive.google.com/file/d/1X1o-uAwTwHajtwTfBK1X-Y97amvOgrQP/view?usp=sharing) or [Baidu](https://pan.baidu.com/s/1zYrwkkmKKuhUwbEFs-lZJw?pwd=84ns)), Read, Sign an EULA (please add a handwritten or electronic signature), and Attach it To Your Email.**
-->


### Dataset Structure

```
DRTiD
├── Original Images
└── Ground Truths
    ├── DR_grade
    │   ├── a. DR_grade_Training.csv
    │   └── b. DR_grade_Testing.csv
    └── Optic_Macula_Localization
        └──op_ma_localization.csv

```

All images from the DRTiD dataset are of gradable quality and annotated by three experienced ophthalmologists.

We also provide the initial version, without image quality check and label re-verification by ophthalmologists. The initial labels are provided by community screening.


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

### Contact

If you have any questions, please feel free to contact Dr. Junlin Hou (jlhou18@fudan.edu.cn).
