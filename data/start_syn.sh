
# synthesis non-professional data
python generate_long_seq_dataset.py --pkg_num non_1 --mode test
python generate_long_seq_dataset.py --pkg_num non_2 --mode test
python generate_long_seq_dataset.py --pkg_num non_3 --mode test
python generate_long_seq_dataset.py --pkg_num non_4 --mode test

python generate_long_seq_dataset.py --pkg_num non_1 --mode val
python generate_long_seq_dataset.py --pkg_num non_2 --mode val
python generate_long_seq_dataset.py --pkg_num non_3 --mode val
python generate_long_seq_dataset.py --pkg_num non_4 --mode val

python generate_long_seq_dataset.py --pkg_num non_1 --mode train
python generate_long_seq_dataset.py --pkg_num non_2 --mode train
python generate_long_seq_dataset.py --pkg_num non_3 --mode train
python generate_long_seq_dataset.py --pkg_num non_4 --mode train

















