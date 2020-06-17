# SS-DCNet
This is the repository for SS-DCNet, presented in our paper:

[**From Open Set to Closed Set: Supervised Spatial Divide-and-Conquer for Object Counting**](https://arxiv.org/abs/2001.01886)

[Haipeng Xiong](https://scholar.google.com/citations?user=AEW8GxcAAAAJ&hl=zh-CN)<sup>1</sup>, [Hao Lu](https://sites.google.com/site/poppinace/)<sup>2</sup>, Chengxin Liu<sup>1</sup>,
Liang Liu<sup>1</sup>, [Chunhua Shen](http://cs.adelaide.edu.au/~chhshen/)<sup>2</sup>, Zhiguo Cao<sup>1</sup> 

<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>The University of Adelaide, Australia

## Contributions
- We propose to transform open-set counting into a closed set problem via S-DC. A theoretical analysis of why such a transformation makes sense is also presented;
- We investigate the explicit supervision for S-DC, which leads to a novel SS-DCNet. SS-DCNet is applicable to both regression-based and classification-based counters and can produce visually clear spatial divisions;
- We report state-of-the-art counting performance over 5 challenging datasets with remarkable relative improvements. We also show good transferablity of SS-DCNet via cross-dataset evaluations on crowd counting datasets.

## Results
The mean average error (MAE) of SS-DCNet against its previous version, S-DCNet, over 5 challenging counting datasets are reported on the table:

|      | SHA | SHB |UCF_CC_50 | UCF-QNRF | TRANCOS | MTC|
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|S-DCNet (cls)|58.3|6.7|204.2|104.4|2.92|5.6|   
|SS-DCNet (cls)|56.1|6.6|179.2|81.9|2.42|3.9|

## Environment
Please install required packages according to `requirements.txt`.

## Data
Testing data for ShanghaiTech and UCF-QNRF dataset have been preprocessed. You can download the processed dataset from:

ShanghaiTech PartA [[Baidu Yun]](https://pan.baidu.com/s/1s34zLNARwgsxmQ1JV2xN3A) with code: po1v or [[Google Drive]](https://drive.google.com/open?id=1bYL9t9vWiez-fVJEBWonxxNDHU63gpF2)

ShanghaiTech PartB [[Baidu Yun]](https://pan.baidu.com/s/1s34zLNARwgsxmQ1JV2xN3A) with code: po1v or [[Google Drive]](https://drive.google.com/open?id=1bYL9t9vWiez-fVJEBWonxxNDHU63gpF2)

UCF-QNRF [[Baidu Yun]](https://pan.baidu.com/s/1s34zLNARwgsxmQ1JV2xN3A) with code: po1v or [[Google Drive]](https://drive.google.com/open?id=1bYL9t9vWiez-fVJEBWonxxNDHU63gpF2)

## Model
Pretrained weights can be downloaded from:

ShanghaiTech PartA [[Baidu Yun]](https://pan.baidu.com/s/1vL0r5ntWHQ_fKlUg6J-zqw) with code: weng or [[Google Drive]](https://drive.google.com/open?id=1TRJr9YuP1dFpnbQvSSQHqIqhLFdElo_Q)

ShanghaiTech PartB [[Baidu Yun]](https://pan.baidu.com/s/1vL0r5ntWHQ_fKlUg6J-zqw) with code: weng or [[Google Drive]](https://drive.google.com/open?id=1TRJr9YuP1dFpnbQvSSQHqIqhLFdElo_Q)

UCF-QNRF [[Baidu Yun]](https://pan.baidu.com/s/1vL0r5ntWHQ_fKlUg6J-zqw) with code: weng or [[Google Drive]](https://drive.google.com/open?id=1TRJr9YuP1dFpnbQvSSQHqIqhLFdElo_Q)


## A Quick Demo
1. Download the code, data and model.

2. Organize them into one folder. The final path structure looks like this:
```
-->The whole project
    -->data
        -->SH_partA
        -->SH_partB
        -->UCF-QNRF_ECCV18
    -->model
        -->SHA
        -->SHB
        -->QNRF
    -->Network
        -->base_Network_module.py
        -->merge_func.py
        -->class_func.py
        -->SSDCNet.py
    -->all_main.py
    -->main_process.py
    -->Val.py
    -->load_data_V2.py
    -->IOtools.py
```

3. Run the following code to reproduce our results. The MAE will be SHA: 55.571, SHB: 6.645 and QNRF: 81.864 . Have fun:)
       
       for ShanghaiTech PartA:
       python all_main.py --dataset SHA 
       
       for ShanghaiTech PartB:
       python all_main.py --dataset SHB 
       
       for UCF-QNRF:
       python all_main.py --dataset QNRF 
       


## References
If you find this work or code useful for your research, please cite:
```
@misc{xiong2020open,
    title={From Open Set to Closed Set: Supervised Spatial Divide-and-Conquer for Object Counting},
    author={Haipeng Xiong and Hao Lu and Chengxin Liu and Liang Liu and Chunhua Shen and Zhiguo Cao},
    year={2020},
    eprint={2001.01886},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
and 
```
@inproceedings{xhp2019SDCNet,
  title={From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer},
  author={Xiong, Haipeng and Lu, Hao and Liu, Chengxin and Liang, Liu and Cao, Zhiguo and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019},
  pages = {8362-8371}
}
```
