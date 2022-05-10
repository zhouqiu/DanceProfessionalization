import numpy as np
from utils.Quaternions_old import Quaternions

from torch.utils.data import Dataset
from utils.util import get_files
import librosa


class Dataset_AIST(Dataset):
    def __init__(self, non_path, music_path, phase='train', total_length=3000, isSlice=False, slice_pad=30):
        print("dataset mode:{}, total_length:{}, isSlice:{}, slice_pad:{}.".format(phase, total_length, isSlice, slice_pad))

        non_list = get_files(non_path, '.npz')
        self.static = np.load(non_list[0], allow_pickle=True)["static"].item()
        n_all_quats = []
        n_all_rtposes = []
        n_all_filenames = []

        all_targets = []
        all_matrixes = []
        all_pkg_nums = []

        pkgs = set()
        for i in range(len(non_list)):
            if non_list[i].split("_")[-2] == phase:
                pkg_num = int(non_list[i].split("/")[-1].split(".")[0].split("_")[1])
                pkgs.add(pkg_num)
                print(non_list[i])
                print("pkg num:{}".format(pkg_num))
                non_dataset = np.load(non_list[i], allow_pickle=True)[phase].item()

                n_quats, n_rt_poses, n_filenames, targets, matrixes = \
                    non_dataset["quats"], non_dataset["rt_poses"], non_dataset["filenames"],non_dataset["targets"], non_dataset["matrixes"]

                pkg_nums = pkg_num * np.ones(len(n_quats))
                pkg_nums = pkg_nums.tolist()

                if isSlice:
                    n_quats = n_quats[::slice_pad]
                    n_rt_poses = n_rt_poses[::slice_pad]
                    n_filenames = n_filenames[::slice_pad]
                    targets = targets[::slice_pad]
                    matrixes = matrixes[::slice_pad]
                    pkg_nums = pkg_nums[::slice_pad]

                n_all_quats.extend(n_quats)
                n_all_rtposes.extend(n_rt_poses)
                n_all_filenames.extend(n_filenames)
                all_targets.extend(targets)
                all_matrixes.extend(matrixes)
                all_pkg_nums.extend(pkg_nums)

        self.package_nums = len(pkgs)
        self.length = len(n_all_quats)
        self.package_length = self.length // self.package_nums
        print("the length of set: {}, pkg num: {}, the length of pkg: {}.".format(self.length, self.package_nums,
                                                                                  self.package_length))
        self.n_all_quats = n_all_quats  # [N, T, J, 4]
        self.n_all_rtposes = n_all_rtposes  # [N, T, 3]
        self.n_all_filenames = n_all_filenames  # [N]
        self.all_targets = all_targets  # [N T]
        self.all_matrixes = all_matrixes #[N, Ta, Tm]
        self.all_pkg_nums = all_pkg_nums  # [N]
        print("read motion data finished.")

        self.chosen_joints = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26]  # Jo = 21

        self.all_global_pos = self.__get_all_global_pos(self.n_all_quats, self.n_all_rtposes)  # [N, T, Jo, 3]

        self.all_velocity = self.__get_all_velocity(self.all_global_pos)  # [N,T,Jo,3]
        self.all_acce = self.__get_all_accele(self.all_velocity)  # [N,T,Jo,3]

        print("process motion data finished.")


        self.music_dict, sample_rate = self.get_origin_music(music_path)
        self.all_melsDB, _, _ = self.music_match(self.n_all_quats, self.n_all_filenames, self.music_dict, sample_rate, 60) #[N, 80, T]  [N, T] [N, T]

        print("the length of music set: {}.".format(len(self.all_melsDB)))
        print("read music data finished.")

        self.total_length = total_length


    def music_match(self, quats, filenames, music_dict, mel_sr, frame_rate):
        num_mels = 80  # number of mel bins
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

            # normal
            if np.max(audio_feature) > np.min(audio_feature):
                audio_feature = (audio_feature - np.mean(audio_feature)) / (
                            np.max(audio_feature) - np.min(audio_feature))
            else:
                audio_feature = np.zeros_like(audio_feature)
            music_clips.append(audio_feature)

        return music_clips, onset_clips, beat_seq_clips

    def get_origin_music(self, music_path):
        mel_sr = 44100  # audio sampling rate

        music_dict = {}
        music_list = get_files(music_path, '.mp3')
        for i in range(len(music_list)):
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
            # local position
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
        b = target
        b[-1] = np.floor(b[-1])

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
            "pkg_num": self.all_pkg_nums[i],  # [1]
            "matrix": new_matrix,  # [Tm, Ta]

            "motion_feature": new_agg,  # [126, T]
            "audio_feature": new_audio_feature,  # [80, Tm]

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








