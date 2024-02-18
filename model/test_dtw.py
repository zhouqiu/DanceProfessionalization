from .dataset.long_dataset_dance import Dataset_AIST
from torch.utils.data import DataLoader
from .models.dtw_net import DTWModel
from utils.util import mkdir
from .options.test_options import TestOptions

import os

import numpy as np


if __name__ == '__main__':

    opt = TestOptions().parse()
    gpu_ids_str = ""
    for g in opt.gpu_ids:
        gpu_ids_str += str(g) + ","
    print("visible gpu ids:{}".format(gpu_ids_str))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    print("Prepare dataset:")
    testset = Dataset_AIST(opt.non_path, opt.music_path, opt.dataset_mode, opt.total_length, isSlice=opt.isSlice)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    static_dict = testset.getStatic()
    print("Prepare dataset done.")

    print("Prepare model:")
    model = DTWModel(opt)
    print("Prepare model done.")

    savepath = os.path.join(opt.result_root, opt.result_path)
    mkdir(savepath)

    targets = []
    preds = []
    dists = []
    sample_names = []
    music_names = []
    real_lengths = []

    for i, data in enumerate(testloader, 0):

        filename = testset.get_filename(i)

        out_dict = model.test(data)
        music_name = filename.split("_")[-2] + ".mp3"
        sample_names.append(filename)
        music_names.append(music_name)

        real_length = int(out_dict["real_length"].squeeze(0).detach().cpu().numpy())
        target = out_dict["target_mat"].detach().cpu().numpy()
        pred = out_dict["pred_mat"].detach().cpu().numpy()
        dist = out_dict["dist"].detach().cpu().numpy()

        target = np.squeeze(target)[:real_length, :real_length]
        pred = np.squeeze(pred)[:real_length, :real_length]
        dist = np.squeeze(dist)[:real_length, :real_length]

        targets.append(target)
        preds.append(pred)
        dists.append(dist)
        real_lengths.append(real_length)


    print("number: {}".format(len(targets)))
    print("shape:{}".format(targets[0].shape))

    data_dict = {}
    data_dict[opt.dataset_mode] = {"targets": targets, "preds": preds, "dists":dists,
                         "sample_names": sample_names, "real_lengths":real_lengths}
    np.savez_compressed(os.path.join(savepath, "matrixes.npz"), **data_dict)



