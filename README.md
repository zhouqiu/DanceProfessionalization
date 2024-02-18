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

**c. Install numpy.**
```shell
pip install numpy==1.18.5
# 1.21.6 ? 
```

**d. Install librosa.**
```shell
pip install librosa==0.8.1
```

pip install pymel
pip install pyyaml
pip install python-probe

## Data Preparation
You can download dance data [HERE](https://github.com/zhouqiu/DanceData) and symlink the dataset root to `$DanceProfessionalization/`.

Here, the total code structure is shown like this:
```
DanceProfessionalization
├── AIST++_bvh/
├── code_blender/
├── code_model/
├── code_syn_non_data/
├── mp3/
├── real_set/
├── train_val_testset/
├── utils/
```


Then Run the following command to generate synthenized data.
```bash
cd code_syn_non_data 
sh start_syn.sh
```



## Training & Testing
```bash
cd code_model
```


#### train alignment DTW
```bash
python train_dtw.py --net_path dtw --gpu_ids 0,1 --useTripletloss --isConv --isTrans
python test_dtw.py --net_path dtw --gpu_ids 0 --isConv --isTrans --total_length 1000 --result_path dtw --model_epoch 400
```


#### generate DTW result
```bash
sh dtw_aist.sh
```


#### train enhancement autoencoder
```bash
python train_dance.py --net_path dance --dtw_path ./all_results/dtw_result  --velo_w 1 --gpu_ids 0,1 --iters 100
python train_dance.py --net_path dance --dtw_path ./all_results/dtw_result --isFinetune --velo_w 1 --gpu_ids 0,1 --continueTrain --model_epoch 100 --iters 100
```


#### test
```bash
python test_dance.py --net_path dance --dtw_path ./all_results/dtw_result --isFinetune --gpu_ids 0 --result_path testset_result --model_epoch 200
```


---------------------------------------------------------- real data ----------------------------------------------------------

#### generate DTW result
```bash
sh dtw_real.sh
```


#### test
```bash
python test_dance.py --net_path dance --non_path ./real_set --dtw_path ./all_results/dtw_real_result --isFinetune --gpu_ids 0 --result_path realset_result --model_epoch 200 --save_pkg_num 1
```



## Visualization
```bash
cd code_blender
demo.bat
```


## Acknowledgement
This project is not possible without multiple great open-sourced code bases. We list some notable examples below.
* [AIST++](https://xxx) 
* [Blender](https://xxx) 

