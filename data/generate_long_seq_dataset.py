import os
import argparse
import numpy as np
from utils.InverseKinematics import JacobianInverseKinematics
from utils.animation_data import forward_rotations
from utils.load_skeleton import Skel
from utils.BVH import load
from scipy import interpolate

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_kine_beat(glb, threshold_down = -0.5, threshold = 60):
    velocity = glb[1:] - glb[:-1] #[T-1, J, 3]

    # normalize
    velo_len = np.linalg.norm(velocity, axis=-1) #[T-1,J]
    velocity = velocity / velo_len[..., np.newaxis] # [T-1,J,3]
    # dot multiple
    velomul = velocity[1:] * velocity[:-1] # [T-2,J,3]
    velomul = np.sum(velomul, axis=-1) # [T-2,J]
    #minimize
    velomin = np.min(velomul, axis=-1) #[T-2]
    velomin = np.concatenate((velomin[-2:], velomin), axis=0) #[T]


    beat_list = []
    beat_list.append(0)
    for t in range(1, len(velomin)-1):
        if velomin[t] < threshold_down and t-beat_list[-1] > threshold:
            beat_list.append(t)
    if len(velomin)-1 - beat_list[-1] < threshold:
        beat_list.pop()
    beat_list.append(len(velomin) - 1)

    beat_seq = np.zeros_like(velomin)
    beat_list = sorted(set(beat_list), key=beat_list.index) #去重
    for b in beat_list:
        beat_seq[b] = 1

    return beat_list, beat_seq


def modi_angle_w_seq(motion, w):
    # motion:[T, J*3] w:ndarray
    # w:[T,J]
    frames = len(motion)  # T
    motion = motion.reshape(frames, -1, 3)  # [T, J, 3]

    p_num = [-1, 0, 1, 2, 3, 4,   0, 6, 7, 8, 9,  0, 11, 12, 13, 14, 15,   13, 17,18,19,20,   13,22,23,24,25]
    joint_num = 27

    limb_joints = [3, 4, 5,  8, 9, 10,   19, 20, 21,   24, 25, 26]
    spine_joints = [ 12,13, 15, 16]

    offset_len = []
    for j in range(joint_num):
        if p_num[j] != -1:
            offset = motion[..., j, :] - motion[..., p_num[j], :]  # [T,3]
        else:  # 0
            offset = motion[..., j, :]  # [T,3]
        offset_len.append(offset)
    offset_len = np.array(offset_len)
    offset_len = np.transpose(offset_len, (1, 0, 2))  # [T, J, 3]

    new_motion = []
    for i in range(frames):
        new_frame = []
        for j in range(joint_num):  # joints:0 - J-1
            if j in limb_joints:
                axis = [0, -1, 0]
                offset = offset_len[i, j]  # [3]
                limb_len = np.linalg.norm(offset, axis=0)  # [1]
                # direction = np.array(axis)[np.newaxis, :].repeat(frames, axis=0)  # (3,) -> [1, 3] -> [T, 3] -y axis
                direction = limb_len * np.array(axis)  # 1 * [ 3] = [3]
                new_offset = w[i,j] * offset + (1 - w[i,j]) * direction  # [3]
                new_offset = new_offset / np.linalg.norm(new_offset, axis=0)# [3] maintain limb length
                new_offset = limb_len * new_offset  # [3]
                new_position = new_frame[p_num[j]] + new_offset  # [3]

            elif j in spine_joints:
                axis = [0, 1, 0]
                offset = offset_len[i, j]  # [3]
                limb_len = np.linalg.norm(offset, axis=0)# [1]
                # direction = np.array(axis)[np.newaxis, :].repeat(frames, axis=0)  # (3,) -> [1, 3] -> [T, 3] y axis
                direction = limb_len * np.array(axis)  # [1] * [3] = [3]
                new_offset = w[i,j] * offset + (1 - w[i,j]) * direction  # [3]
                new_offset = new_offset / np.linalg.norm(new_offset, axis=0) # [3] maintain limb length
                new_offset = limb_len * new_offset  # [3]
                new_position = new_frame[p_num[j]] + new_offset  # [3]

            else:
                if p_num[j] != -1:  # 1, 5, 14, 19
                    new_position = new_frame[p_num[j]] + offset_len[i, j]  # [3]
                else:  # 0
                    new_position = offset_len[i, j]  # [3]
            new_frame.append(new_position) #[J, 3]
        new_motion.append(new_frame) #[T, J, 3]

    new_motion = np.array(new_motion)  # [T,J, 3]
    return new_motion


def ampliJointChange(rotation, rtpos, glb, skel, weight):
    rest, _, _ = skel.rest_bvh
    bvh = rest.copy()
    bvh.positions = bvh.positions.repeat(len(rtpos), axis=0)
    bvh.positions[:, 0, :] = rtpos
    bvh.rotations.qs = rotation

    glbpos = modi_angle_w_seq(glb, w=weight)

    targetmap = {}
    for j in range(glb.shape[1]):
        targetmap[j] = glbpos[:, j]

    ik = JacobianInverseKinematics(bvh, targetmap, iterations=20, damping=4.0, silent=True)
    ik()

    return bvh.rotations.qs, bvh.positions[:, 0, :]


def U_distribution(arraysize:tuple, alpha=1.0, beta=0.0):
    base_uniform = np.random.uniform(-3, 3, arraysize)  # [-3,3) normal distribution
    b = np.tanh(base_uniform)  # (-1,1) U-distribution

    rand_w = alpha * b + beta  # (beta-alpha, beta+alpha)
    return rand_w


def change_rhy_forsingle(quat, rtpos, beat_list, alpha, beta):  # [T, J, 4] [T, 3]
    new_beat_list = beat_list.copy()

    randpos = U_distribution((len(beat_list)-2,), alpha, beta)
    new_beat_list[1:-1] += randpos
    new_beat_list = np.sort(new_beat_list)
    while min(new_beat_list[1:] - new_beat_list[:-1]) <= 0 or new_beat_list[0]<0 or new_beat_list[-1]>=len(quat):
        new_beat_list = beat_list.copy()
        randpos = U_distribution((len(beat_list) - 2,), alpha, beta)
        new_beat_list[1:-1] += randpos
        new_beat_list = np.sort(new_beat_list)

    frames = len(quat)
    target = np.linspace(0, frames-1, frames)

    f = interpolate.interp1d(beat_list, new_beat_list, kind='linear')
    random_position = f(target)

    new_target = np.zeros(frames)
    newquat = np.zeros_like(quat)
    newrt = np.zeros_like(rtpos)

    for k in range(frames):
        orig_index = random_position[k]
        lower = int(np.floor(orig_index))
        upper = int(np.ceil(orig_index))

        if lower == upper:
            newquat[k] = quat[lower]
            newrt[k] = rtpos[lower]

            new_target[k] = target[lower]
        else:
            newquat[k] = quat[lower] * (upper - orig_index) + quat[upper] * (orig_index - lower)
            newrt[k] = rtpos[lower] * (upper - orig_index) + rtpos[upper] * (orig_index - lower)

            new_target[k] = target[lower] * (upper - orig_index) + target[upper] * (orig_index - lower)


    return newquat, newrt, new_target # [T, J, 4] [T, 3], [T]


def anti_target(target):
    target = target.reshape((-1))
    frames = len(target)

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

        if i >= frames-1: # last step
            ori_ori_target[j] = ori_target[j]
        else:
            ori_ori_target[j] = (1 / (target[i + 1] - target[i])) * (
                (ori_target[j] - target[i]) * blank[i + 1] + (target[i + 1] - ori_target[j]) * blank[i])
        j += 1

    return ori_ori_target


def matrix(target):
    target[-1] = np.floor(target[-1])
    mat = np.zeros((len(target), len(target)))

    for i in range(len(target)):
        lower = int(np.floor(target[i]))
        upper = int(np.ceil(target[i]))

        if lower == upper:
            mat[i][lower] = 1
        else:
            mat[i][lower] = upper - target[i]
            mat[i][upper] = target[i] - lower

    return mat  # [Ta, Tm]


def generate_dance_from_long_seq(split_path, dance_path, dance_target_name, beat_pad, ampli_alpha=1.1, ampli_beta=1.3, rhy_alpha=50, rhy_beta=0, frame_Rate=60, mode="train", pkg_length=250):

    with open(os.path.join(split_path + mode +".txt"), "r") as f:
        files = f.readlines()
    print("The number of files :{}.".format(len(files)))

    print('Start Dance: {}.'.format(mode))
    all_inputs = []
    static_info = {}
    for i in range(len(files)):
        # get file path
        filename = files[i][:-1] if files[i][-1] == "\n" else files[i]
        file = os.path.join(dance_path, filename+".bvh")

        anim, names, frameRate = load(file)
        rot_order = 'zyx'
        pos_order = 'xyz'

        quat = anim.rotations
        root_position = anim.positions[:,0,:]
        seq_len = len(quat)
        joints = quat.shape[1]
        print(" process:{}, bvh name:{}, seq_len:{}, the number of joints:{}".format(i, file, seq_len, joints))

        # static info
        if i == 0:
            static_info = {'rot_order': rot_order, 'pos_order': pos_order, 'offsets': anim.offsets,
                           'parents': anim.parents, 'names': names, 'frameRate': frameRate}  # the static information of bvh


        skel = Skel()
        skel.offset = anim.offsets
        skel.topology = anim.parents
        glb = forward_rotations(skel, quat.qs, root_position, trim=False)  # [T, J, 3] global position

        beat_list, _ = calc_kine_beat(glb)

        total_ampli_w = np.zeros((seq_len, joints))
        w_beat_count = int(seq_len / beat_pad)
        if w_beat_count < 4:
            w_beat_count = 4

        # U-distribution
        a_w = U_distribution((joints, w_beat_count), alpha=ampli_alpha, beta=ampli_beta)
        a_sign = np.sign(np.sum(np.sign(a_w-ampli_beta), axis=0)) #[w_b]
        a_w = a_sign[np.newaxis,...] * np.abs(a_w - ampli_beta) + ampli_beta

        frm_pos = np.linspace(0, seq_len - 1, w_beat_count)

        # find beat time point
        new_frmpos = []
        c_index = 0
        for fp in frm_pos:
            current_pos = int(np.round(fp))
            while c_index < len(beat_list) :
                if beat_list[c_index] >= current_pos:
                    if beat_list[c_index] == current_pos:
                        new_frmpos.append(beat_list[c_index])
                    elif c_index-1 > 0 and abs(beat_list[c_index-1] - fp) < abs(beat_list[c_index] - fp):
                        new_frmpos.append(beat_list[c_index-1])
                    else:
                        new_frmpos.append(beat_list[c_index])
                    break
                c_index += 1

        new_frmpos = sorted(set(new_frmpos), key=beat_list.index)
        if len(new_frmpos) < w_beat_count:
            new_frmpos = frm_pos.astype(np.int).tolist()

        for b in range(0, w_beat_count-1): #0, 3, 6, ...
            st_point = b
            fi_point = b+2 #b+4

            mats = np.zeros((2, 2))
            mats[0] = np.logspace(0, 1, 2, base=0)
            mats[1] = np.logspace(0, 1, 2, base=new_frmpos[fi_point-1]-new_frmpos[st_point])

            params = np.linalg.solve(mats, a_w[:, st_point:fi_point].transpose()) # [5,J]
            times = np.linspace(0, new_frmpos[fi_point-1]-new_frmpos[st_point], new_frmpos[fi_point-1] - new_frmpos[st_point] + 1 )
            a_ws = []
            for t in range(len(times)):
                tseq = np.logspace(0, 1, 2, base=times[t])
                a_ws.append(np.sum(np.multiply(np.repeat(tseq[np.newaxis, ...], joints, axis=0), params.transpose()), axis=-1))
            ampli_w = np.array(a_ws)  # [T',J]

            total_ampli_w[new_frmpos[st_point]:new_frmpos[fi_point-1]+1] = ampli_w.copy()

        # modify amplitude
        ampli_quat, ampli_rt = ampliJointChange(quat.qs, root_position, glb, skel, weight=total_ampli_w)

        # modify rhythm
        rhy_ampli_quat, rhy_ampli_rt, rhy_target = change_rhy_forsingle(ampli_quat, ampli_rt, new_frmpos, alpha=rhy_alpha, beta=rhy_beta) # [T, J, 4] [T, 3], [T]
        ori_rhy_target = anti_target(rhy_target)
        target_matrix = matrix(ori_rhy_target)

        storename = file.split("\\")[-1].split(".")[0]
        if storename == "":
            storename = file.split("/")[-1].split(".")[0]
        print(" storename:{}".format(storename))

        rhy_ampli_clip = {'quat': rhy_ampli_quat,
                        'rt_pos': rhy_ampli_rt,
                        'pquat': quat.qs,
                        'prt_pos': root_position,
                        'filename': storename,
                        "target": rhy_target,
                        "matrix": target_matrix}
        all_inputs.append(rhy_ampli_clip)
    print('Finish Dance.')
    print("the number of inputs:" + str(len(all_inputs)))


    data_dict = {}
    quats = [input["quat"] for input in all_inputs]
    rt_poses = [input["rt_pos"] for input in all_inputs]
    pquats = [input["pquat"] for input in all_inputs]
    prt_poses = [input["prt_pos"] for input in all_inputs]
    filenames = [input["filename"] for input in all_inputs]
    targets = [input["target"] for input in all_inputs]
    matrixes = [input["matrix"] for input in all_inputs]

    total_length = len(all_inputs)
    print("store:")
    for t in range(total_length // pkg_length + 1):
        print(" zipping {} to {}：".format(t * pkg_length, min((t + 1) * pkg_length, total_length) - 1))
        data_dict[mode] = {"quats": quats[t * pkg_length: min((t + 1) * pkg_length, total_length)],
                           "rt_poses": rt_poses[t * pkg_length: min((t + 1) * pkg_length, total_length)],
                           "pquats": pquats[t * pkg_length: min((t + 1) * pkg_length, total_length)],
                           "prt_poses": prt_poses[t * pkg_length: min((t + 1) * pkg_length, total_length)],
                           "filenames": filenames[t * pkg_length: min((t + 1) * pkg_length, total_length)],
                           "targets": targets[t * pkg_length: min((t + 1) * pkg_length, total_length)],
                           "matrixes": matrixes[t * pkg_length: min((t + 1) * pkg_length, total_length)]
                           }
        data_dict["static"] = static_info
        np.savez_compressed(dance_target_name + "_" + mode + "_" + str(t) + ".npz", **data_dict)
        print(" zipping {} to {} finished.".format(t * pkg_length, min((t + 1) * pkg_length, total_length) - 1))


    print("store finish")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=str, default="./AISTtable2/crossmodal_") #./table_range/crossmodal_
    parser.add_argument("--dance_path", type=str, default="./AIST++_bvh")
    parser.add_argument("--store_path", type=str, default="")
    parser.add_argument("--pkg_dir", type=str, default="./train_val_testset")
    parser.add_argument("--pkg_num", type=str, default="non_1")

    parser.add_argument("--beat_pad", type=int, default=180)

    parser.add_argument("--ampli_alpha", type=float, default=1.1)
    parser.add_argument("--ampli_beta", type=float, default=1.3)
    parser.add_argument("--rhy_alpha", type=float, default=50)
    parser.add_argument("--rhy_beta", type=float, default=0)

    parser.add_argument("--frameRate", type=int, default=60)
    parser.add_argument("--mode", type=str, default="test", help="[train|test|val]")

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()

    dance_target_dir = os.path.join(args.store_path, args.pkg_dir)
    mkdir(dance_target_dir)

    generate_dance_from_long_seq(args.split_path, args.dance_path, dance_target_dir+"/"+args.pkg_num,
                                 args.beat_pad, ampli_alpha=args.ampli_alpha, ampli_beta=args.ampli_beta, rhy_alpha=args.rhy_alpha, rhy_beta=args.rhy_beta,
                                 frame_Rate=args.frameRate, mode=args.mode)

