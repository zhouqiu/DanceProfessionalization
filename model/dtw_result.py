from torch.utils.data import DataLoader
from .models.dtw_net import DTWModel
from utils.util import mkdir
from .options.test_options import TestOptions
import os

import numpy as np
from torch.utils.data import Dataset
import librosa
from utils.util import get_files
from utils.Quaternions_old import Quaternions


class Dataset_AIST(Dataset):
    def __init__(self, non_pkg_path, music_path, phase='train', total_length=3000):
        print("non_pkg_path: {}, total_length:{}".format(non_pkg_path, total_length))

        self.static = np.load(non_pkg_path, allow_pickle=True)["static"].item()
        non_dataset = np.load(non_pkg_path, allow_pickle=True)[phase].item()
        # [N, T, J, 4] [N, T, 3] [N, T, J, 4] [N, T, 3] [N]  [N T]
        self.n_all_quats, self.n_all_rtposes,  self.n_all_filenames, self.all_targets, self.all_matrixes = \
            non_dataset["quats"], non_dataset["rt_poses"], non_dataset["filenames"], non_dataset["targets"], non_dataset["matrixes"]

        self.length = len(self.n_all_quats)
        print("the length of set: {}.".format(self.length))
        print("read motion data finished.")

        self.chosen_joints = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26]  # Jo = 21

        self.all_global_pos = self.__get_all_global_pos(self.n_all_quats, self.n_all_rtposes)  # [N, T, Jo, 3]

        self.all_velocity = self.__get_all_velocity(self.all_global_pos)  # [N,T,Jo,3]
        self.all_acce = self.__get_all_accele(self.all_velocity)  # [N,T,Jo,3]

        # self.all_ori_targets = self.__get_all_ori_targets(self.all_targets)  # [N T]
        # self.all_matrixes = self.__get_matrix(self.all_ori_targets)  # [N, Ta, Tm]

        print("process motion data finished.")

        self.music_dict, sample_rate = self.get_origin_music(music_path)
        self.all_melsDB, _, _ = self.music_match(self.n_all_quats, self.n_all_filenames, self.music_dict, sample_rate, 60) #[N, 80, T]  [N, T] [N, T]

        print("the length of music set: {}.".format(len(self.all_melsDB)))
        print("read music data finished.")

        self.total_length = total_length


    def music_match(self, quats, filenames, music_dict, mel_sr, frame_rate):
        num_mels = 80  # number of mel bins
        # mel_sr = 44100  # audio sampling rate
        mel_n_fft = 1024  # 2048         # n_fft
        mel_hop = 736  # 512             # hop length

        music_clips = []
        onset_clips = []
        beat_seq_clips = []
        for i in range(self.length):
            frames = len(quats[i])
            music_name = filenames[i].split("_")[-2]
            waveform = music_dict[music_name]
            waveform = waveform[:int(frames / frame_rate * mel_sr)]
            audio_feature = librosa.feature.melspectrogram(y=waveform,
                                                           sr=mel_sr,
                                                           n_fft=mel_n_fft,
                                                           hop_length=mel_hop,
                                                           n_mels=num_mels)
            # make sure: music feature time length == motion time length
            if audio_feature.shape[-1] > frames:
                audio_feature = audio_feature[:, :frames]
            if audio_feature.shape[1] < frames:
                audio_feature = np.concatenate((audio_feature, audio_feature[:, -(frames - audio_feature.shape[1]):]),
                                               axis=-1)
            audio_feature = librosa.amplitude_to_db(audio_feature)

            # onset_env = librosa.onset.onset_strength(sr=mel_sr, S=audio_feature, aggregate=np.median)  # [Ta]
            # _, beats = librosa.beat.beat_track(sr=mel_sr, onset_envelope=onset_env)

            # normal
            if np.max(audio_feature) > np.min(audio_feature):
                audio_feature = (audio_feature - np.mean(audio_feature)) / (
                            np.max(audio_feature) - np.min(audio_feature))
            else:
                audio_feature = np.zeros_like(audio_feature)
            music_clips.append(audio_feature)

            # beat_seq = np.zeros_like(onset_env)  # [T]
            # beat_seq[beats.tolist()] = np.ones(len(beats))
            # beat_seq_clips.append(beat_seq)
            #
            # # normal
            # if np.max(onset_env) > np.min(onset_env):
            #     onset_env = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env))
            # else:
            #     onset_env = np.zeros_like(onset_env)
            #     # print(onset_env)
            # onset_clips.append(onset_env)

        return music_clips, onset_clips, beat_seq_clips

    def get_origin_music(self, music_path):
        # num_mels = 80  # number of mel bins
        mel_sr = 44100  # audio sampling rate
        # mel_n_fft = 1024  # 2048         # n_fft
        # mel_hop = 256  # 512             # hop length

        music_dict = {}
        music_list = get_files(music_path, '.mp3')
        for i in range(len(music_list)):
            # print(music_list[i])
            musicname = music_list[i]
            waveform, audio_sample_rate = librosa.load(musicname, mel_sr)
            key = musicname.split("/")[-1].split(".")[0]
            music_dict[key] = waveform
        print("music number:{}".format(len(music_list)))
        return music_dict, mel_sr

    def __forward_rotations(self, rotations, rtpos=None, trim=False):
        """
        input: rotations [T, J, 4], rtpos [T, 3]
        output: positions [T, J, 3]
        """
        parents = self.static['parents']
        offsets = self.static['offsets']

        transforms = Quaternions(rotations).transforms()  # [T, J, 3, 3]
        glb = np.zeros(rotations.shape[:-1] + (3,))  # [T, J, 3]
        if rtpos is not None:
            glb[..., 0, :] = rtpos
        for i, pi in enumerate(parents):
            if pi == -1:
                continue
            glb[..., i, :] = np.matmul(transforms[..., pi, :, :],
                                       offsets[i])
            # print(offsets[i])
            glb[..., i, :] += glb[..., pi, :]
            transforms[..., i, :, :] = np.matmul(transforms[..., pi, :, :],
                                                 transforms[..., i, :, :])
        if trim:
            glb = glb[..., self.chosen_joints, :]
        return glb

    def __get_all_local_pos(self, quats):
        all_local_pos = []
        for i in range(self.length):
            # local position 21 joints
            local_position = self.__forward_rotations(quats[i], trim=True)  # [T, Jo, 3]
            local_position = local_position.reshape(len(local_position), -1)

            all_local_pos.append(local_position)
        return all_local_pos

    def __get_all_global_pos(self, quats, rtposes):
        all_global_pos = []
        for i in range(self.length):
            # 求local position
            global_position = self.__forward_rotations(quats[i], rtposes[i], trim=True)  # [T, Jo, 3]
            all_global_pos.append(global_position)
        return all_global_pos  # [N, T, Jo, 3]

    def __get_all_velocity(self, local_poses):  # [N, T, J, 3]
        velos = []
        # chosen_joints = [4, 8, 18, 23]
        for i in range(self.length):
            local_pos = local_poses[i]  # [T, J, 3]
            delta = local_pos[1:] - local_pos[:-1]
            delta = np.abs(np.concatenate((delta[0, np.newaxis, :], delta), axis=0))  # [T,J,3]
            velos.append(delta)
        # velos = np.array(velos)
        return velos  # [N,T,J, 3]

    def __get_all_accele(self, velos):  # [N, T, J, 3]
        acces = []
        # chosen_joints = [4, 8, 18, 23]
        for i in range(self.length):
            acc = velos[i]  # [T, 3]
            delta = acc[1:] - acc[:-1]
            delta = np.abs(np.concatenate((delta[0, np.newaxis, :], delta), axis=0))  # [T,J,3]
            acces.append(delta)
        # acces = np.array(acces)
        return acces  # [N,T,J,3]

    def __get_all_ori_targets(self, targets):
        ori_targets = []
        for i in range(self.length):
            ori_target = self.anti_target(targets[i])
            ori_targets.append(ori_target)
        return ori_targets

    def anti_target(self, target):
        target = target.reshape((-1))

        frames = len(target)
        # ori_target = np.linspace(-1, 1, frames)
        # blank = np.linspace(-1, 1, frames)
        ori_target = np.linspace(0, frames - 1, frames)
        blank = np.linspace(0, frames - 1, frames)
        ori_ori_target = np.zeros_like(target)

        i = 0
        j = 0
        while j < frames:
            while i < frames - 1:
                if ori_target[j] >= target[i]:
                    if ori_target[j] <= target[i + 1]:
                        break
                i += 1

            ori_ori_target[j] = (1 / (target[i + 1] - target[i])) * (
                    (ori_target[j] - target[i]) * blank[i + 1] + (target[i + 1] - ori_target[j]) * blank[i])
            j += 1

        return ori_ori_target

    def __get_matrix(self, targets):
        ori_matrixes = []
        for i in range(self.length):
            mat = self.matrix(targets[i])
            ori_matrixes.append(mat)
        return ori_matrixes  # [N, Ta, Tm]

    def matrix(self, target):
        # b = (target + 1) / 2 * (len(target) - 1)
        b = target
        b[-1] = np.floor(b[-1])
        # print("length {} ori_target:{} {}".format(len(b), b[0], b[-1]))
        # print(b)

        mat = np.zeros((len(b), len(b)))
        for i in range(len(b)):
            # print("bi={}".format(b[i]))
            lower = int(np.floor(b[i]))
            upper = int(np.ceil(b[i]))

            if lower == upper:
                mat[i][lower] = 1
            else:
                mat[i][lower] = upper - b[i]
                mat[i][upper] = b[i] - lower
                # newquat[k] = quat[lower] * (upper - orig_index) + quat[upper] * (orig_index - lower)
                # new_target[k] = target[lower] * (upper - orig_index) + target[upper] * (orig_index - lower)

        return mat  # [Ta, Tm]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        i = index % self.length

        velo = self.all_velocity[i]  # [T,Jo,3]
        velo = velo / np.amax(np.absolute(velo))
        acce = self.all_acce[i]  # [T,Jo,3]
        acce = acce / np.amax(np.absolute(acce))
        agg = np.concatenate((velo[..., np.newaxis], acce[..., np.newaxis]), axis=-1)  # [T,Jo,3,2]
        agg = agg.reshape((len(agg), -1)).transpose()

        real_length = len(self.n_all_quats[i])  # [T]


        matrix = self.all_matrixes[i].transpose()  # [Tm, Ta]
        audio_feature = self.all_melsDB[i]  # [80, T]


        # beat_seq = self.all_beat_seq[i]  # [T]

        if real_length < self.total_length:
            padding_mask = np.ones(self.total_length)
            padding_mask[:real_length] = np.zeros(real_length)

            new_matrix = np.zeros((self.total_length, self.total_length))
            new_matrix[:real_length, :real_length] = matrix.copy()
            new_agg = np.zeros((agg.shape[0], self.total_length))
            new_agg[:, :real_length] = agg.copy()
            new_audio_feature = np.zeros((audio_feature.shape[0], self.total_length))
            new_audio_feature[:, :real_length] = audio_feature.copy()
        else:# real_length >= self.total_length:
            padding_mask = np.zeros(self.total_length)

            new_matrix = matrix[:self.total_length, :self.total_length].copy()
            new_agg = agg[:, :self.total_length].copy()
            new_audio_feature = audio_feature[:, :self.total_length].copy()

        motion_len = real_length if real_length < self.total_length else self.total_length

        return {
            "matrix": new_matrix,  # [Tm, Ta]

            "motion_feature": new_agg,  # [126, T]
            "audio_feature": new_audio_feature,  # [80, Tm]

            # "beat_seq": new_beat_seq,  # [T]

            "padding_mask": padding_mask,  # [ T]
            "real_length": motion_len,  # [1]
            'filename': self.n_all_filenames[i],

        }


    def get_filename(self, index):
        i = index % self.length
        return self.n_all_filenames[i]

    def getStatic(self):
        return self.static

    def getLength(self):
        return self.length


if __name__ == '__main__':

    opt = TestOptions().parse()
    gpu_ids_str = ""
    for g in opt.gpu_ids:
        gpu_ids_str += str(g) + ","
    print("visible gpu ids:{}".format(gpu_ids_str))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    print("Prepare dataset:")

    store_pkg_name = "dtw_" + opt.non_path.split("/")[-1][4:]
    print("store_pkg_name: {}.".format(store_pkg_name))

    testset = Dataset_AIST(opt.non_path, opt.music_path, opt.dataset_mode, opt.total_length)
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
    real_lengths = []

    for i, data in enumerate(testloader, 0):

        filename = testset.get_filename(i)
        print("{}: current :{}".format(i,filename))

        out_dict = model.test(data)
        sample_names.append(filename)

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

    np.savez_compressed(os.path.join(savepath, store_pkg_name), **data_dict)