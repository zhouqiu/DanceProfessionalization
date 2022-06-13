# DanceProfessionalization

## Prepare:Environment
#### python 3.7
#### cuda 10.1
#### pytorch 1.6.0
#### numpy 1.18.5
#### librosa 0.8.1

## 1.generate training data & testing data
#### cd code_syn_non_data 
#### sh start_syn.sh



## 2.training & testing model
#### cd code_model

### train alignment DTW
#### python train_dtw.py --net_path dtw --gpu_ids 0,1 --useTripletloss --isConv --isTrans
#### python test_dtw.py --net_path dtw --gpu_ids 0 --isConv --isTrans --total_length 1000 --result_path dtw --model_epoch 400

### generate DTW result
#### sh dtw_aist.sh

### train enhancement autoencoder
#### python train_dance.py --net_path dance --dtw_path ./all_results/dtw_result  --velo_w 1 --gpu_ids 0,1 --iters 100
#### python train_dance.py --net_path dance --dtw_path ./all_results/dtw_result --isFinetune --velo_w 1 --gpu_ids 0,1 --continueTrain --model_epoch 100 --iters 100

### test
#### python test_dance.py --net_path dance --dtw_path ./all_results/dtw_result --isFinetune --gpu_ids 0 --result_path testset_result --model_epoch 200

---------------------------------------------------------- real data ----------------------------------------------------------

### generate DTW result
#### sh dtw_real.sh

### test
#### python test_dance.py --net_path dance --non_path ./real_set --dtw_path ./all_results/dtw_real_result --isFinetune --gpu_ids 0 --result_path realset_result --model_epoch 200 --save_pkg_num 1



## 3.blend
#### cd code_blender
#### demo.bat

