import numpy as np
from utils.Quaternions_old import Quaternions
import os
import math

from torch.utils.data import Dataset
from utils.util import get_files
import librosa


class Dataset_AIST(Dataset):
    def __init__(self, non_path, dtw_path, phase='train', total_length=3000, isSlice=False, slice_pad=30):
        print("dataset mode:{}, total_length:{}, isSlice:{}, slice_pad:{}.".format(phase, total_length, isSlice, slice_pad))

        non_list = get_files(non_path, '.npz')
        self.static = np.load(non_list[0], allow_pickle=True)["static"].item()
        n_all_quats = []
        n_all_rtposes = []
        n_all_filenames = []
        p_all_quats = []
        p_all_rtposes = []
        all_targets = []
        all_pkg_nums = []
        all_matrixes = []

        pkgs = set()
        for i in range(len(non_list)):
            if non_list[i].split("_")[-2] == phase:
                pkg_num = int(non_list[i].split("/")[-1].split(".")[0].split("_")[1])
                pkgs.add(pkg_num)
                print(non_list[i])
                print("pkg num:{}".format(pkg_num))
                non_dataset = np.load(non_list[i], allow_pickle=True)[phase].item()

                n_quats, n_rt_poses, p_quats, p_rt_poses, n_filenames, targets, matrixes = \
                    non_dataset["quats"], non_dataset["rt_poses"], non_dataset["pquats"], non_dataset["prt_poses"], \
                    non_dataset["filenames"], non_dataset["targets"], non_dataset["matrixes"]

                pkg_nums = pkg_num * np.ones(len(n_quats))
                pkg_nums = pkg_nums.tolist()

                if isSlice:
                    n_quats = n_quats[::slice_pad]
                    n_rt_poses = n_rt_poses[::slice_pad]
                    n_filenames = n_filenames[::slice_pad]
                    p_quats = p_quats[::slice_pad]
                    p_rt_poses = p_rt_poses[::slice_pad]
                    targets = targets[::slice_pad]
                    matrixes = matrixes[::slice_pad]
                    pkg_nums = pkg_nums[::slice_pad]

                n_all_quats.extend(n_quats)
                n_all_rtposes.extend(n_rt_poses)
                n_all_filenames.extend(n_filenames)
                p_all_quats.extend(p_quats)
                p_all_rtposes.extend(p_rt_poses)
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
        self.p_all_quats = p_all_quats  # [N, T, J, 4]
        self.p_all_rtposes = p_all_rtposes  # [N, T, 3]
        self.all_targets = all_targets  # [N T]
        self.all_matrixes = all_matrixes #[N, Ta, Tm]
        self.all_pkg_nums = all_pkg_nums  # [N]
        print("read motion data finished.")

        dtw_list = get_files(dtw_path, '.npz')
        all_dtw_matrixes = []
        all_dtw_dists = []
        for i in range(len(dtw_list)):
            if dtw_list[i].split("_")[-2] == phase:
                print(dtw_list[i])
                dtw_dataset = np.load(dtw_list[i], allow_pickle=True)[phase].item()
                dtw_matrixes = dtw_dataset["preds"]
                dtw_dists = dtw_dataset["dists"]


                if isSlice:
                    dtw_matrixes = dtw_matrixes[::slice_pad]
                    dtw_dists = dtw_dists[::slice_pad]

                all_dtw_matrixes.extend(dtw_matrixes)
                all_dtw_dists.extend(dtw_dists)
        self.all_dtw_matrixes = all_dtw_matrixes  # [N, Tm, Ta]
        self.all_dtw_dists = all_dtw_dists  # [N, Tm, Ta]
        print("the length of dtw set: {}.".format(len(self.all_dtw_matrixes)))

        self.chosen_joints = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26]  # Jo = 21

        self.new_parents = [0,1,2,3, 0,5,6,7, 0,9,10,11, 10,13,14,15, 10,17,18,19] #[20]
        self.new_offset = self.static["offsets"][self.chosen_joints[1:]] #[20,3]
        self.new_offset_len = np.linalg.norm(self.new_offset, axis=-1) #[20]

        self.n_all_local_pos = self.__get_all_local_pos(self.n_all_quats)  # [N, T, Jo*3]
        self.p_all_local_pos = self.__get_all_local_pos(self.p_all_quats)  # [N, T, Jo*3]

        self.n_all_glbs = self.__get_all_global_pos(self.n_all_quats, self.n_all_rtposes)  #[N,T,J,3]
        self.foot_contact = self.__get_all_foot_contact(self.n_all_glbs) #[N,T,4]
        self.p_all_glbs = self.__get_all_global_pos(self.p_all_quats, self.p_all_rtposes)  # [N,T,J,3]
        self.p_foot_contact = self.__get_all_foot_contact(self.p_all_glbs)  # [N,T,4]

        self.n_directions = self.__get_all_direction(self.n_all_local_pos, self.new_parents, self.new_offset_len) #[N,T,20,3]
        self.p_directions = self.__get_all_direction(self.p_all_local_pos, self.new_parents, self.new_offset_len) #[N,T,20,3]

        print("process motion data finished.")

        self.total_length = total_length

    def __get_all_foot_contact(self, glbs): #[N,T,J,3]
        foot_contacts = []
        for i in range(len(glbs)):
            ft = self.foot_contact_from_positions(glbs[i]) #[T,4]
            foot_contacts.append(ft)
        return foot_contacts #[N, T, 4]

    def foot_contact_from_positions(self, positions, fid_l=(3, 4), fid_r=(7, 8)):
        """
        positions: [T, J, 3], trimmed (only "chosen_joints")
        fid_l, fid_r: indices of feet joints (in "chosen_joints")
        """
        fid_l, fid_r = np.array(fid_l), np.array(fid_r)
        velfactor = np.array([0.05, 0.05])
        feet_contact = []
        for fid_index in [fid_l, fid_r]:
            foot_vel = (positions[1:, fid_index] - positions[:-1, fid_index]) ** 2  # [T - 1, 2, 3]
            foot_vel = np.sum(foot_vel, axis=-1)  # [T - 1, 2]
            foot_contact = (foot_vel < velfactor).astype(np.float)
            feet_contact.append(foot_contact)
        feet_contact = np.concatenate(feet_contact, axis=-1)  # [T - 1, 4]
        feet_contact = np.concatenate((feet_contact[0:1].copy(), feet_contact), axis=0)

        return feet_contact  # [T, 4]

    def __get_all_direction(self, positions, parents, off_len):
        directions = []
        for i in range(len(positions)):
            position = positions[i] #[T, Jo*3]
            T = position.shape[0]
            position = position.reshape((T, -1, 3)) #[T, 21, 3]
            direction = position[:,1:] - position[:, parents]  #[T, 20, 3]
            direction = direction / off_len[np.newaxis, :, np.newaxis] #normalization
            directions.append(direction)
        return directions



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
            glb[..., i, :] += glb[..., pi, :]
            transforms[..., i, :, :] = np.matmul(transforms[..., pi, :, :],
                                                 transforms[..., i, :, :])
        if trim:
            glb = glb[..., self.chosen_joints, :]
        return glb

    def __get_all_local_pos(self, quats):
        all_local_pos = []
        for i in range(self.length):
            # 根据新的rotation，求local position 并修剪成21个关节点
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

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        i = index % self.length

        real_length = len(self.n_directions[i])  # [T]
        nquat = self.n_all_quats[i].reshape((real_length,-1)).transpose()  #[108, T]
        n_direct = self.n_directions[i].reshape((real_length, -1)).transpose() #[60,T]
        nrtpos = self.n_all_rtposes[i].transpose()  # [3, T]
        pquat = self.p_all_quats[i].reshape((real_length, -1)).transpose()  # [108, T]
        p_direct = self.p_directions[i].reshape((real_length, -1)).transpose() #[60,T]
        prtpos = self.p_all_rtposes[i].transpose()  # [3, T]
        matrix = self.all_matrixes[i].transpose()  # [Tm, Ta]
        off_len = self.new_offset_len #[20]
        parents = self.new_parents #[20]

        dtw_matrix = self.all_dtw_matrixes[i]  # [Tm, Ta]
        dtw_dist = self.all_dtw_dists[i]  # [Tm,Ta]

        ft = self.foot_contact[i] #[T,4]
        pft = self.p_foot_contact[i] #[T, 4]


        if real_length < self.total_length:
            padding_mask = np.ones(self.total_length)
            padding_mask[:real_length] = np.zeros(real_length)

            new_nquat = np.zeros((nquat.shape[0], self.total_length))
            new_nquat[:,:real_length] = nquat.copy()
            new_pquat = np.zeros((pquat.shape[0], self.total_length))
            new_pquat[:, :real_length] = pquat.copy()
            new_ndirect = np.zeros((n_direct.shape[0], self.total_length))
            new_ndirect[:, :real_length] = n_direct.copy()
            new_pdirect = np.zeros((p_direct.shape[0], self.total_length))
            new_pdirect[:, :real_length] = p_direct.copy()
            new_nrtpos = np.zeros((3, self.total_length))
            new_nrtpos[:, :real_length] = nrtpos.copy()
            new_prtpos = np.zeros((3, self.total_length))
            new_prtpos[:, :real_length] = prtpos.copy()

            new_matrix = np.zeros((self.total_length, self.total_length))
            new_matrix[:real_length, :real_length] = matrix.copy()

            new_dtw_matrix = np.zeros((self.total_length, self.total_length))
            new_dtw_matrix[:real_length, :real_length] = dtw_matrix.copy()
            new_dtw_dist = np.zeros((self.total_length, self.total_length))
            new_dtw_dist[:real_length, :real_length] = dtw_dist.copy()

            new_ft = np.zeros((self.total_length, 4))
            new_ft[:real_length] = ft.copy()
            new_pft = np.zeros((self.total_length, 4))
            new_pft[:real_length] = pft.copy()
        else: #real_length >= self.total_length:
            padding_mask = np.zeros(self.total_length)

            new_nquat = nquat[:,:self.total_length].copy()
            new_pquat = pquat[:, :self.total_length].copy()

            new_ndirect = n_direct[:,:self.total_length].copy()
            new_pdirect = p_direct[:,:self.total_length].copy()
            new_nrtpos = nrtpos[:, :self.total_length].copy()
            new_prtpos = prtpos[:, :self.total_length].copy()

            new_matrix = matrix[:self.total_length, :self.total_length].copy()

            new_dtw_matrix = dtw_matrix[:self.total_length, :self.total_length].copy()
            new_dtw_dist = dtw_dist[:self.total_length, :self.total_length].copy()

            new_ft = ft[:self.total_length]
            new_pft = pft[:self.total_length]


        motion_len = real_length if real_length < self.total_length else self.total_length

        return {
            "pkg_num": self.all_pkg_nums[i],  # [1]
            "matrix": new_matrix,  # [Tm, Ta]

            "nquat": new_nquat, #[108, T]
            "ndirect": new_ndirect,  # [60, T]
            "nrtpos": new_nrtpos,  # [3, T]
            "pquat": new_pquat,  # [108, T]
            "pdirect": new_pdirect,  # [60, T]
            "prtpos": new_prtpos,  # [3, T]

            "padding_mask": padding_mask,  # [ T]
            "real_length": motion_len,  # [1]
            'filename': self.n_all_filenames[i],

            "dtw_matrix": new_dtw_matrix,
            "dtw_dist": new_dtw_dist,

            "off_len": off_len, #[20]
            "parents": np.array(parents), #[20]

            "ft": new_ft.transpose(), #[4,T]
            "pft": new_pft.transpose()  # [4,T]

        }

    def offset_tpose(self):
        offsets = self.static["offsets"] #[J,3]
        parents = self.static["parents"] #[J]
        lcl_tpose = np.zeros_like(offsets)#[J,3]
        for i in range(1, offsets.shape[0]):
            pi = parents[i]
            lcl_tpose[i] = lcl_tpose[pi] + offsets[i]
        return lcl_tpose #[J,3]


    def get_filename(self, index):
        i = index % self.length
        return self.n_all_filenames[i]

    def getStatic(self):
        return self.static

    def getLength(self):
        return self.length






