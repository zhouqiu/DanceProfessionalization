import os
import numpy as np
from utils.load_save_BVH import save
from utils.Quaternions_old import Quaternions
# from utils.Pivots import Pivots
import utils.BVH as BVH
from utils.InverseKinematics import JacobianInverseKinematics

def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r

def get_files(directory, suffix):
    # print(directory)
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith(suffix)]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_current_losses(epoch, losses, log_name):
    """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
    """
    message = 'epoch: %d ' % epoch
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


# def save_result(data_dict, quat, rt, savepath, static_dict, frameRate=60, suffix=''):
#     #static
#     rot_order = static_dict["rot_order"]
#     pos_order = static_dict["pos_order"]
#     offsets = static_dict["offsets"]
#     parents = static_dict["parents"]
#     names = static_dict["names"]
#     # print(len(names))
#     # frameRate = static_dict["frameRate"]
#     #dynamic
#     dancer = data_dict["filename"]
#     # emotion = data_dict["emotion"]
#     start = data_dict["start_pos"].detach().cpu().numpy()
#     #result
#     # 只改了这里一句
#     quat = quat.squeeze(0).detach().cpu().numpy().reshape(4*24, -1).transpose() # 60,  124
#     quat = quat.reshape(-1, 24, 4)  # 60, 31, 4
#     # print("quat shape")
#     # print(quat.shape)
#     rt = rt.squeeze(0).detach().cpu().numpy().reshape(3, -1).transpose()# 60,  3
#
#     f_path = os.path.join(savepath, dancer[0] + '_'  + str(start)[1:-1] + suffix + '.bvh')
#
#     save(quat, rt, rot_order, pos_order, offsets, parents, names, frameRate, f_path)

# def save_result_from_data(data_dict, quat, rt, savepath, static_dict, frameRate=60, suffix=''):
#     #static
#     rot_order = static_dict["rot_order"]
#     pos_order = static_dict["pos_order"]
#     offsets = static_dict["offsets"]
#     parents = static_dict["parents"]
#     names = static_dict["names"]
#     # print(len(names))
#     # frameRate = static_dict["frameRate"]
#     #dynamic
#     dancer = data_dict["filename"]
#     # emotion = data_dict["emotion"]
#     start = data_dict["start_pos"].detach().cpu().numpy()
#     #result
#     # 只改了这里一句
#
#     quat = quat.reshape(-1, 27, 4)  # T, 27, 4
#     rt = rt.reshape(-1, 3)
#     # print(quat.shape)
#     # print("quat shape")
#     # print(quat.shape)
#     # rt = rt.transpose()# 60,  3
#     # print(rt.shape)
#
#     f_path = os.path.join(savepath, dancer[0] + '_' + str(start)[1:-1] + suffix + '.bvh')
#
#     save(quat, rt, rot_order, pos_order, offsets, parents, names, frameRate, f_path)

def add_tpose(quat, rt): #[T,J,4] [T,3]
    J=quat.shape[1]
    tpose = np.zeros((J,4))
    tpose[:, 0] = np.ones(J)
    quat = np.concatenate((tpose[np.newaxis, ...], quat), axis=0) #[T+1, J, 4]
    rt = np.concatenate((rt[0][np.newaxis,...], rt), axis=0) #[T+1, 3]
    return quat, rt


def save_result_from_long(quat, rt, savepath, static_dict, filename, frameRate=60, suffix=''):
    #static
    rot_order = static_dict["rot_order"]
    pos_order = static_dict["pos_order"]
    offsets = static_dict["offsets"]
    parents = static_dict["parents"]
    names = static_dict["names"]
    # print(len(names))
    # frameRate = static_dict["frameRate"]
    # #dynamic
    # dancer = data_dict["filename"]
    # # emotion = data_dict["emotion"]
    # start = data_dict["start_pos"].detach().cpu().numpy()
    #result
    # 只改了这里一句

    quat = quat.reshape(-1, 27, 4)  # T, 27, 4
    rt = rt.reshape(-1, 3)

    #增加一帧Tpose
    quat, rt = add_tpose(quat, rt)

    # print(quat.shape)
    # print("quat shape")
    # print(quat.shape)
    # rt = rt.transpose()# 60,  3
    # print(rt.shape)

    f_path = os.path.join(savepath, filename + suffix + '.bvh')

    save(quat, rt, rot_order, pos_order, offsets, parents, names, frameRate, f_path)

# def save_result_from_data_for_Andreas(data_dict, quat, rt, savepath, static_dict, frameRate=60, suffix=''):
#     #static
#     rot_order = static_dict["rot_order"]
#     pos_order = static_dict["pos_order"]
#     offsets = static_dict["ori_offsets"]
#     parents = static_dict["ori_parents"]
#     names = static_dict["names"]
#     # print(len(names))
#     # frameRate = static_dict["frameRate"]
#     #dynamic
#     dancer = data_dict["filename"]
#     print(dancer)
#     # emotion = data_dict["emotion"]
#     start = data_dict["start_pos"].detach().cpu().numpy()
#     print(start)
#     #result
#     # 只改了这里一句
#     print(quat.shape)
#     quat = quat.reshape(-1, 24, 4)  # T, 24, 4
#     rt = rt.reshape(-1, 3)
#     print(rt.shape)
#
#     chosen_joints = [0, 1,2,3,4,  5, 6,7,8,  10,11, 12,13,  15,16,17,18,   20,21,22,23 ]
#     tar_joints = [0, 2,3,4,5, 7,8,9,10, 12,13,15,16,  18,19,20,22,   25,26,27,29]
#     new_quat = np.zeros((len(quat), 31, 4))
#     new_quat[:,:,0] = np.ones((len(quat), 31))
#     new_quat[:, tar_joints] = quat[:, chosen_joints].copy()
#     # print(quat.shape)
#     # print("quat shape")
#     # print(quat.shape)
#     # rt = rt.transpose()# 60,  3
#     # print(rt.shape)
#
#     f_path = os.path.join(savepath, dancer[0] + '_' + str(start)[1:-1] + suffix + '.bvh')
#
#     save(new_quat, rt, rot_order, pos_order, offsets, parents, names, frameRate, f_path)
#

def retarget(rtpos, glb_pos, template_quat):
    #target bvh
    rest, names, _ = BVH.load('./Andreas.bvh')

    targets = glb_pos #[T,J,3]

    #make output anim
    anim = rest.copy()
    pos = anim.offsets[np.newaxis,...].repeat(len(targets), axis=0)
    pos[:,0] = rtpos.copy() #[T,3]
    anim.positions = pos

    tar_chosen_joints = [0, 2, 3, 4, 5,  7, 8, 9, 10,  12, 13, 15, 16,  18, 19, 20, 21,  23, 24, 25, 26]  # [20]
    quat = anim.rotations.qs.repeat(len(targets), axis=0) #[T, 27]
    quat[:, tar_chosen_joints] = template_quat[:, tar_chosen_joints].copy()
    anim.rotations.qs = quat

    # make target map
    targetmap = {}
    for i in range(targets.shape[1]):
        targetmap[i] = targets[:, i]

    ik = JacobianInverseKinematics(anim, targetmap, iterations=20, damping=2.0, silent=True)
    ik()

    return anim.rotations.qs

def get_glb(direct, rtpos, off_len,parents):
    # 求ik_quat

    length = len(direct)
    direct = direct / np.linalg.norm(direct, axis=-1)[..., np.newaxis] #normalization
    direct = direct * off_len[np.newaxis, :, np.newaxis]  # [T, 20, 3]
    glb = np.zeros((length, len(parents) + 1, 3))  # [T, 21, 3]
    for j in range(len(parents)):
        c = j + 1
        p = parents[j]
        glb[:, c] = glb[:, p] + direct[:, j]
    glb += rtpos[:, np.newaxis, :]  # [T,21,3]
    match_joints = [0, 0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 10, 11, 12, 10, 13, 14, 15, 16, 10, 17, 18, 19,
                    20]  # [27]
    final_glb = glb[:, match_joints]
    return final_glb

def foot_contact_from_positions(positions, fid_l=(4, 5), fid_r=(9, 10)):#fid_l=(3, 4), fid_r=(7, 8)
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

def fix_on_floor(glb, rtpos, fid_l=(4, 5), fid_r=(9, 10)):
    print("fix_on_floor")
    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    # print(np.min(foot_heights))
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print(floor_height)
    rtpos[:, 1] -= floor_height
    return rtpos

def fix_on_floor_for_aist(glb, rtpos, fid_l=(4, 5), fid_r=(9, 10)):
    print("fix_on_floor_for_aist")
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    # print(np.min(foot_heights))
    # floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print(floor_height)
    rtpos[:, 1] -= foot_heights
    return rtpos



def remove_fs(template_quat, glb,rtpos, foot, fid_l=(4, 5), fid_r=(9, 10), interp_length=5, force_on_floor=True):
    # (anim, names, ftime), glb = nrot2anim(anim, frametime=1/15)
    # print("ftime=" + str(ftime))
    # target bvh
    # template_quat = np.concatenate((template_quat[0][np.newaxis,...], template_quat),axis=0) #[T+1,J,4]
    # rtpos = np.concatenate((rtpos[0][np.newaxis,...], rtpos),axis=0) #[T+1,3]

    rest, names, _ = BVH.load('./Andreas.bvh')

    targets = glb  # [T,J,3]
    newT = len(targets)

    # make output anim
    anim = rest.copy()
    pos = anim.offsets[np.newaxis, ...].repeat(newT, axis=0)
    pos[:, 0] = rtpos.copy()  # [T+1,3]
    anim.positions = pos

    tar_chosen_joints = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26]  # [20]
    quat = anim.rotations.qs.repeat(newT, axis=0)  # [T+1, 27]
    quat[:, tar_chosen_joints] = template_quat[:, tar_chosen_joints].copy()
    anim.rotations.qs = quat


    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    # print(np.min(foot_heights))
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    # print(floor_height)
    glb[:, :, 1] -= floor_height
    anim.positions[:, 0, 1] -= floor_height

    for i, fidx in enumerate(fid):
        fixed = foot[:, i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            # print(fixed[s - 1:t + 2])

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()


    from scipy.signal import savgol_filter

    # glb = savgol_filter(glb, 9,3)
    # glb_tpose = lcl_tpose + rtpos[0][np.newaxis,:] #[J,3]
    # glb = np.concatenate((glb_tpose[np.newaxis,...], glb), axis=0) #[T+1,J,3]

    targetmap = {}
    for j in range(glb.shape[1]):
        #平滑
        temp_x = savgol_filter(glb[:, j, 0], 9, 3)
        temp_y = savgol_filter(glb[:, j, 1], 9, 3)
        temp_z = savgol_filter(glb[:, j, 2], 9, 3)
        temp = np.concatenate((temp_x[:, np.newaxis],
                               temp_y[:, np.newaxis],
                               temp_z[:, np.newaxis]), axis=-1) #[T, 3]
        targetmap[j] = temp
        # targetmap[j] = glb[:, j]
    # print("targetmap:")
    # print(len(targetmap)) #24
    # print(targetmap[0].shape) #[60, 3]




    ik = JacobianInverseKinematics(anim, targetmap, iterations=50, damping=4.0,
                                   silent=False)
    ik()

    return anim.rotations.qs, glb[:,0]



