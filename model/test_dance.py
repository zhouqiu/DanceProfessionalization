from .dataset.long_dataset_dance import Dataset_AIST
from torch.utils.data import DataLoader
from .models.dance_net import AISTModel
from utils.util import mkdir, save_result_from_long
from .options.test_options import TestOptions
import os

import numpy as np

from utils.util import get_glb, remove_fs,fix_on_floor,fix_on_floor_for_aist



if __name__ == '__main__':

    opt = TestOptions().parse()
    gpu_ids_str = ""
    for g in opt.gpu_ids:
        gpu_ids_str += str(g) + ","
    print("visible gpu ids:{}".format(gpu_ids_str))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    print("Prepare dataset:")
    testset = Dataset_AIST(opt.non_path, opt.dtw_path, opt.dataset_mode, opt.total_length, isSlice=opt.isSlice)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    static_dict = testset.getStatic()
    print("Prepare dataset done.")

    print("Prepare model:")
    model = AISTModel(opt)
    print("Prepare model done.")

    savepath = os.path.join(opt.result_root, opt.result_path)
    mkdir(savepath)
    if not opt.isSlice:
        in_path = os.path.join(savepath, "input")
        out_path = os.path.join(savepath, "output")
        pro_path = os.path.join(savepath, "pro")
        mkdir(in_path)
        mkdir(out_path)
        mkdir(pro_path)


    for i, data in enumerate(testloader, 0):
        pkg_num = int(data["pkg_num"].squeeze(0).detach().cpu().numpy())
        if not opt.isSlice and pkg_num == opt.save_pkg_num:
            out_dict = model.test(data)

            filename = testset.get_filename(i)
            music_name = filename.split("_")[-2] + ".mp3"
            real_length = int(out_dict["real_length"].squeeze(0).detach().cpu().numpy())

            nft = out_dict["nft"].squeeze(0).detach().cpu().numpy().transpose()  # [T,4]
            oft = out_dict["oft"].squeeze(0).detach().cpu().numpy().transpose()  # [T,4]

            nquat = data["nquat"].squeeze(0).detach().cpu().numpy().reshape(-1, opt.total_length).transpose().reshape(opt.total_length, -1, 4)  # [T,J,4]
            nquat = nquat[:real_length]
            pquat = data["pquat"].squeeze(0).detach().cpu().numpy().reshape(-1, opt.total_length).transpose().reshape(opt.total_length, -1, 4) #[T,J,4]
            pquat = pquat[:real_length]
            ndirect = out_dict["direct_in"].squeeze(0).detach().cpu().numpy().reshape(-1,
                                                                                      opt.total_length).transpose().reshape(
                opt.total_length, -1, 3)  # [T,20,3]
            ndirect = ndirect[:real_length]
            odirect = out_dict["direct_out"].squeeze(0).detach().cpu().numpy().reshape(-1,opt.total_length).transpose().reshape(opt.total_length, -1, 3)  # [T,20,3]
            odirect = odirect[:real_length]
            pdirect = out_dict["direct_pro"].squeeze(0).detach().cpu().numpy().reshape(-1,
                                                                                       opt.total_length).transpose().reshape(
                opt.total_length, -1, 3)  # [T,20,3]
            pdirect = pdirect[:real_length]

            nrtpos = out_dict["nrtpos"].squeeze(0).detach().cpu().numpy().reshape(3, -1).transpose()# [T,3]
            nrtpos = nrtpos[:real_length]
            ortpos = out_dict["ortpos"].squeeze(0).detach().cpu().numpy().reshape(3, -1).transpose() # [T,3]
            ortpos = ortpos[:real_length]
            prtpos = out_dict["prtpos"].squeeze(0).detach().cpu().numpy().reshape(3, -1).transpose()# [T,3]
            prtpos = prtpos[:real_length]

            off_len = data["off_len"].squeeze(0).detach().cpu().numpy()  # [20]
            parents = data["parents"].squeeze(0).detach().cpu().numpy()  # [20]
            nglb = get_glb(ndirect, nrtpos, off_len, parents)  # [T,J,3]
            oglb = get_glb(odirect, ortpos, off_len, parents)
            pglb = get_glb(pdirect, prtpos, off_len, parents)  # [T,J,3]
            if filename[0] != 'g':
                nquat, nrtpos = remove_fs(nquat, nglb, nrtpos, nft)
                nrtpos = fix_on_floor(nglb, nrtpos)
            else:
                nrtpos = fix_on_floor_for_aist(nglb, nrtpos)
            fixed_ik_quat, fixed_rtpos = remove_fs(nquat, oglb, ortpos, oft)
            prtpos = fix_on_floor(pglb, prtpos)

            if filename[0] != 'g': #for real data ,modity rtpos
                print("for real data ,modify rtpos!")
                mean_prt = np.mean(prtpos, axis=0) #[3]
                mean_nrt = np.mean(nrtpos, axis=0) #[3]
                mean_ort = np.mean(ortpos, axis=0) #[3]
                mean_fixrt = np.mean(fixed_rtpos, axis=0) #[3]
                print("mean x:{}, z:{}".format(mean_prt[0], mean_prt[2]))

                nrtpos[:,0] = nrtpos[:,0] - mean_nrt[0] + mean_prt[0]
                nrtpos[:, 2] = nrtpos[:, 2] - mean_nrt[2] + mean_prt[2]
                ortpos[:,0] = ortpos[:,0] - mean_ort[0] + mean_prt[0]
                ortpos[:,2] = ortpos[:,2] - mean_ort[2] + mean_prt[2]
                fixed_rtpos[:,0] = fixed_rtpos[:,0] - mean_fixrt[0] + mean_prt[0]
                fixed_rtpos[:, 2] = fixed_rtpos[:, 2] - mean_fixrt[2] + mean_prt[2]

            print("current : {}".format(i))
            save_result_from_long(nquat, nrtpos, in_path, static_dict, filename, frameRate=60, suffix='_in')
            save_result_from_long(fixed_ik_quat, fixed_rtpos, out_path, static_dict, filename, frameRate=60, suffix='_out')
            save_result_from_long(pquat, prtpos, pro_path, static_dict, filename, frameRate=60, suffix='_pro')






