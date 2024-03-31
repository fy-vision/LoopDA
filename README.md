# LoopDA: Constructing Self-loops to Adapt Nighttime Semantic Segmentation
**[[WACV23 Paper]](https://openaccess.thecvf.com/content/WACV2023/papers/Shen_LoopDA_Constructing_Self-Loops_To_Adapt_Nighttime_Semantic_Segmentation_WACV_2023_paper.pdf)**
## Abstract
>Due to the lack of training labels and the difficulty of annotating, dealing with adverse driving conditions such as nighttime has posed a huge challenge to the perception system of autonomous vehicles. Therefore, adapting knowledge from a labelled daytime domain to an unlabelled nighttime domain has been widely researched. In addition to labelled daytime datasets, existing nighttime datasets usually provide nighttime images with corresponding daytime reference images captured at nearby locations for reference. The key challenge is to minimize the performance gap between the two domains. In this paper, we propose LoopDA for domain adaptive nighttime semantic segmentation. It consists of self-loops that result in reconstructing the input data using predicted semantic maps, by rendering them into the encoded features. In a warm-up training stage, the self-loops comprise of an inner-loop and an outer-loop, which are responsible for intra-domain refinement and inter-domain alignment, respectively. To reduce the impact of day-night pose shifts, in the later self-training stage, we propose a co-teaching pipeline that involves an offline pseudo-supervision signal and an online reference-guided signal `DNA' (Day-Night Agreement), bringing substantial benefits to enhance nighttime segmentation. Our model outperforms prior methods on Dark Zurich and Nighttime Driving datasets for semantic segmentation.

## Setup Environment

To run our scripts, we suggest setting up the following virtual environment:

```shell
python -m venv ~/venv/loopda
source ~/venv/loopda/bin/activate
```

The required python packages of this environment can be installed by the following command:

```shell
pip install -r requirements.txt
```

## Usage
**Clone** this github repository:
```bash
  git clone https://github.com/fy-vision/LoopDA
  cd LoopDA
```
###Datasets
The datasets are store in ./data, and the strcture is shown as follows 
```none
|──data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── DarkZurich
│   │   ├── Dark_Zurich_train_anon
│   │   │   ├── rgb_anon/val/night
│   │   ├── Dark_Zurich_train_anon
│   │   │   ├── rgb_anon/train/night
│   │   │   ├── rgb_anon/train/day
│   ├── NighttimeDrivingTest
│   │   ├── gtCoarse_daytime_trainvaltest
│   │   │   ├── test/night

```

### Running Evaluation Scripts for LoopDA:
We provide evaluation scripts with PSPNet as follows,

```shell
python evaluate_val.py --data_path_val PATH-TO-VALSET \
                       --weight_dir PATH-TO-WEIGHT_DIR \
                       --data_list_path_val_img PATH-TO-VAL_IMG_LIST \
                       --data_list_path_val_lbl PATH-TO-VAL_LBL_LIST \
                       --dataset_name DATASET_NAME \
                       --cuda_device_id 0
```

Details are given in the script.


## Citation
If you like this work and would like to use our code or models for research, please consider citing our paper:
```
@inproceedings{shen2023loopda,
  title={LoopDA: Constructing self-loops to adapt nighttime semantic segmentation},
  author={Shen, Fengyi and Pataki, Zador and Gurram, Akhil and Liu, Ziyuan and Wang, He and Knoll, Alois},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3256--3266},
  year={2023}
}
```
## Acknowledgement
Our implementation is inspired by [TridentAdapt](https://github.com/HMRC-AEL/TridentAdapt).
