# DanceProfessionalization

## Environment Installation 

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n dance-pro python=3.7 -y
conda activate dance-pro
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
# Recommended torch==1.6.0 and cuda==10.1. Higher versions might cause unknown problems.
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install other dependencies.**
```shell
pip install librosa==0.8.1
pip install pymel
pip install pyyaml
```



## Data Preparation
All data and preparation procedures are included in the `data/` dictionary.
```bash
cd data 
```

You can download refined dance data [HERE](https://github.com/zhouqiu/DanceData) which comes from [AIST++](https://google.github.io/aistplusplus_dataset/) originally and symlink the dataset root to `$DanceProfessionalization/data/`.

Here, the total data structure is shown like this:
```
DanceProfessionalization/
├── blender/
├── data/
    ├── AIST++_bvh/
    ├── mp3/
    ├── real_set/
    ├── train_val_testset/
├── model/
├── utils/
```


Using the pre-generated data `data/train_val_testset/` is recommended. You can also run the following command to generate synthenized data on your own.
```bash
sh start_syn.sh
```



## Training & Testing
All training and testing procedures are included in the `model/` dictionary.
```bash
cd model
```

For our two-stage settings, train the DTW model of the first stage using the command below:
```bash
python train_dtw.py --net_path dtw --useTripletloss --isConv --isTrans
```

Then test the DTW model (optional but not mandatory):
```bash
python test_dtw.py --net_path dtw --gpu_ids 0 --isConv --isTrans --total_length 1000 --result_path dtw --model_epoch 400
```

Before starting the second phase, generate middle results of the DTW model:
```bash
# for default AIST++ dataset
sh dtw_aist.sh  
# if using real data
sh dtw_real.sh  
```

For the second stage, initially train and then finetune the DANCE model:
```bash
python train_dance.py --net_path dance --dtw_path ./all_results/dtw_result  --velo_w 1 --iters 100
python train_dance.py --net_path dance --dtw_path ./all_results/dtw_result --isFinetune --velo_w 1 --continueTrain --model_epoch 100 --iters 100
```


Then test the DANCE model:
```bash
# for default AIST++ dataset
python test_dance.py --net_path dance --dtw_path ./all_results/dtw_result --isFinetune --gpu_ids 0 --result_path testset_result --model_epoch 200  
# if using real data
python test_dance.py --net_path dance --non_path ../data/real_set --dtw_path ./all_results/dtw_real_result --isFinetune --gpu_ids 0 --result_path realset_result --model_epoch 200 --save_pkg_num 1  
```



## Visualization
You can visualize the dance results in the `blender/` dictionary.
```bash
cd blender
```

For visualization, you should firstly install [Blender](https://docs.blender.org/manual/zh-hans/dev/getting_started/installing/index.html) and add `path-to-install-Blender/` to your system variable path.
Then run the following script to visualize your results.

Remember to replace your custom music path, dance bvh path and result save path before running. ^_^
```bash
demo.bat
```


## Acknowledgement
The following researches have made great contributions to this work.

* [Unpaired Motion Style Transfer from Video to Animation](https://deepmotionediting.github.io/style_transfer) 
* [A Deep Learning Framework For Character Motion Synthesis and Editing](https://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing) 

### Citation
If this work is useful to your research, please cite our papers:
```
@article{zhou2023let,
  title={Let’s all dance: Enhancing amateur dance motions},
  author={Zhou, Qiu and Li, Manyi and Zeng, Qiong and Aristidou, Andreas and Zhang, Xiaojing and Chen, Lin and Tu, Changhe},
  journal={Computational Visual Media},
  volume={9},
  number={3},
  pages={531--550},
  year={2023},
  publisher={Springer}
}

```
