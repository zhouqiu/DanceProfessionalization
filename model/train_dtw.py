from torch.utils.data import DataLoader
from .models.dtw_net import DTWModel
from .dataset.long_dataset_align import Dataset_AIST
from utils.util import print_current_losses
from .options.train_options import TrainOptions

import os


if __name__ == '__main__':

    opt = TrainOptions().parse()
    gpu_ids_str = ""
    for g in opt.gpu_ids:
        gpu_ids_str += str(g) + ","
    print("visible gpu ids:{}".format(gpu_ids_str))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    print("Prepare dataset:")
    trainset = Dataset_AIST(opt.non_path, opt.music_path, opt.dataset_mode, opt.total_length, isSlice=opt.isSlice, slice_pad=opt.slice_pad)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    print("Prepare dataset done.")

    print("Prepare model:")
    model = DTWModel(opt)
    print("Prepare model done.")

    log_name = os.path.join(opt.net_root, opt.net_path, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('================ Training Loss ================\n')

    start = 1
    finish = start + opt.iters
    if opt.continueTrain:
        start += opt.model_epoch
        finish += opt.model_epoch
    print("Training: epoch from {} to {}.".format(start, finish))
    for epoch in range(start, finish):
        for i, data in enumerate(trainloader, 0):
            model(data)
            model.optimize()
            loss_dict = model.getlossDict()
            if (i+1) % 100 == 0:
                print(" processed data {}.".format(i+1))

        print_current_losses(epoch, loss_dict, log_name)

        if epoch % opt.store_freq == 0:
            model.save_networks(epoch)
