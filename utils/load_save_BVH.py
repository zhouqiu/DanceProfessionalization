import numpy as np
import sys
import math
from utils.Quaternions_old import Quaternions

def load(filename, frameRate=30):
    f = open(filename, "r")
    lines = f.readlines()
    count = len(lines) # how many lines of this file
    # print(count)

    # static information
    offsets = [] # offsets: joint_num * 3
    parents = np.array([], dtype=int) # parent id: joint_num * 1
    names = [] # joint name: joint_num * 1
    orders = [] # channels' order of joint rotation : joint_num * 1

    # dynamic information
    joint_num = 0 # the number of joints
    frameTime = 0.0
    frames = 0
    root_positions = np.array([]) # root_positions: frames * 3
    rotations = np.array([])#.reshape((0, 0, 3)) #rotations: frames * joint_num * 3
    fCount = 0
    isEnd = False

    stack = Stack()
    stack.push(-1)
    for i in range(count): # i start from 0
        line = lines[i]
        if line.find('HIERARCHY') >= 0 or line.find('MOTION') >= 0:
            continue
        elif line.find('ROOT') >= 0 or line.find('JOINT') >= 0:
            origin = line.split()
            names.append(origin[1])
            joint_num += 1
            parents = np.append(parents, stack.gettop())
        elif line.find('{') >= 0:
            stack.push(joint_num - 1)
        elif line.find('}') >= 0:
            if isEnd:
                isEnd = False
            stack.pop()
        elif line.find('End Site') >= 0:
            isEnd = True
        elif line.find('CHANNELS') >= 0:
            channels = line.split()
            if channels[1] == '3':
                order = channels[2][0].lower() + channels[3][0].lower() + channels[4][0].lower()
                orders.append(order)
            elif channels[1] == '6':
                pos_order = channels[2][0].lower() + channels[3][0].lower() + channels[4][0].lower()
                order = channels[5][0].lower() + channels[6][0].lower() + channels[7][0].lower()
                orders.append(order)
            else:
                print("Too much channels!")
                sys.exit(-1)
        elif line.find('OFFSET') >= 0:
            origin = line.split()
            if not isEnd:
                offsets.append([list(map(float, origin[1:]))])
        elif line.find('Frames') >= 0:
            origin = line.split(':')
            frames = int(origin[1])
            # print(joint_num)
            rotations = np.zeros((frames, joint_num, 3))
            root_positions = np.zeros((frames, 3))
            offsets = np.array(offsets).reshape(joint_num, 3)
        elif line.find('Frame Time') >= 0:
            origin = line.split(':')
            frameTime = float(origin[1])
        else:
            data = line.strip().split()
            if data is not None:
                data_array = np.array(list(map(float, data)))
                root_positions[fCount] = data_array[0:3]
                rotations[fCount] = data_array[3:].reshape(joint_num, 3)
                fCount += 1

    # print('-------------------------finish-------------------------')
    # print('frameTime:' + str(frameTime))
    # # print('frames:' + str(frames))
    # print('pos order:' + pos_order)
    # print('order:' + orders[0])
    # # print('name shape:' + str(len(names)))
    # print('parents id:')
    # print(parents)
    # print('root_positions:' + str(root_positions.shape))
    # print(root_positions)
    # print('rotations:' + str(rotations.shape))
    # print(rotations[0])
    # print(rotations[-1])
    # print('offsets:' + str(offsets.shape))
    # print(offsets)
    # # print('joint number:' + str(joint_num))

    f.close()

    # down sample
    originFR = round(1/frameTime)
    # print(originFR)
    # print(frameRate)
    assert originFR >= frameRate
    s = originFR // frameRate
    # print(s_dataset)
    rotations = rotations[::s]
    rot_order = orders[0]
    root_positions = root_positions[::s]

    #quaternion


    return (rotations, rot_order, root_positions, pos_order, offsets, parents, names, frameRate)


class Stack(object):

    def __init__(self):
        self.stack = []

    def push(self, data):
        self.stack.append(data)

    def pop(self):
        return self.stack.pop()

    def gettop(self):
        return self.stack[-1]

    def is_empty(self):
        return self.stack == []

    def size(self):
        return len(self.items)

def euler2quaternion(euler, order):
    eulerArray = euler / 2
    sA = np.sin(eulerArray * np.pi / 180)
    cA = np.cos(eulerArray * np.pi / 180)

    if order == 'zyx':
        m = [2, 1, 0]
    elif order == 'zxy':
        m = [2, 1, 0]
    else:
        print('Unknown order.')
        sys.exit(-1)

 # 大坑：如果data的顺序是zxy，则求quaternion时每一项中相乘的顺序应是yxz,也就是倒序
    w = cA[:, :, m[0]] * cA[:, :, m[1]] * cA[:, :, m[2]] + sA[:, :, m[0]] * sA[:, :, m[1]] * sA[:, :, m[2]]
    x = sA[:, :, m[0]] * cA[:, :, m[1]] * cA[:, :, m[2]] - cA[:, :, m[0]] * sA[:, :, m[1]] * sA[:, :, m[2]]
    y = cA[:, :, m[0]] * sA[:, :, m[1]] * cA[:, :, m[2]] + sA[:, :, m[0]] * cA[:, :, m[1]] * sA[:, :, m[2]]
    z = cA[:, :, m[0]] * cA[:, :, m[1]] * sA[:, :, m[2]] - sA[:, :, m[0]] * sA[:, :, m[1]] * cA[:, :, m[2]]

    quat = np.concatenate((w[:, :, np.newaxis], x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=2)
    return quat

def quat_normalize(quat):#[..., 4]
    result = np.sum(quat ** 2.0, axis=-1) ** 0.5 #150, 31
    result = result[..., np.newaxis]  # 150 31 1
    result = np.transpose(result, (1,0,2)) #31,150,1

    num = [1,5,6,10,11,21,22,23,28,29,30]
    # print(result)
    quat = np.transpose(quat, (1, 0, 2))# 31,150,4
    total = quat[0][np.newaxis, ...]
    for i in range(1, len(quat)):
        if i in num:
            total = np.concatenate((total, quat[i][np.newaxis, ...]), axis=0)
        else:
            temp=quat[i][np.newaxis, ...]/result[i][np.newaxis, ...]
            total = np.concatenate((total, temp), axis=0)

    total = np.transpose(quat, (1, 0, 2))
    return total

    # quat = quat / result[...,np.newaxis]
    # return quat
    # print(quat)

def quaternion2euler(quat, order='xyz'):
    # quaternion 2 euler
    quat = quat_normalize(quat)
    Y = np.degrees(np.arctan2(2 * (quat[:, :, 0] * quat[:, :, 1] + quat[:, :, 2] * quat[:, :, 3]),
                   1 - 2 * (np.square(quat[:, :, 1]) + np.square(quat[:, :, 2]))))
    X = np.degrees(
        np.arcsin(2 * (quat[:, :, 0] * quat[:, :, 2] - quat[:, :, 3] * quat[:, :, 1])))
    Z = np.degrees(
        np.arctan2(2 * (quat[:, :, 0] * quat[:, :, 3] + quat[:, :, 1] * quat[:, :, 2]),
                   1 - 2 * (np.square(quat[:, :, 2]) + np.square(quat[:, :, 3]))))

    if order == 'zyx':
        euler = np.concatenate((Z[:, :, np.newaxis], X[:, :, np.newaxis], Y[:, :, np.newaxis]), axis=2)
    elif order == 'zxy':
        euler = np.concatenate((Z[:, :, np.newaxis], X[:, :, np.newaxis], Y[:, :, np.newaxis]), axis=2)
    else:
        euler = np.concatenate((X[:, :, np.newaxis], Y[:, :, np.newaxis], Z[:, :, np.newaxis]), axis=2)

    return euler

def save(quat, rt_pos, rot_order, pos_order, offsets, parents, names, frameRate, file_path):
    """
    :param quat: float array [frames, joint_num, 4]
    :param rt_pos: float array [frames, 3]
    :param rot_order: str
    :param pos_order: str
    :param offsets: float array [joint_num, 3]
    :param parents: int array [joint_num, 1]
    :param names: str array [joint_num, 1]
    :param frameRate: int
    :param start_poses: int
    :param dancers: str
    :return: None. Generate a bvh file.
    """
    # print(quat.shape[1])
    # print(len(names))
    # print(len(quat))
    # print(len(rt_pos))
    assert quat.shape[0] == rt_pos.shape[0] # frames是否相等
    assert quat.shape[1] == offsets.shape[0] and quat.shape[1] == parents.shape[0] \
           and quat.shape[1] == len(names)# joint_num是否相等

    frames = quat.shape[0]
    joint_num = quat.shape[1]
    # rotations = quaternion2euler(quat, order=rot_order) #[frames, joint_num, 3]
    # print(quat[0])
    rotations = np.degrees(Quaternions(quat).euler()) #order=rot_order[::-1]
    rotations = rotations[..., ::-1]

    # rt_rot = rotations[:,0,:]
    # # print(rt_rot.shape) #(150, 3)
    # # print(rt_rot[0])
    # temp = rt_rot[:, -1].copy()
    # # print(temp[0])
    # rt_rot[:,-1] = rt_rot[:,-2]
    # # print(rt_rot[0,-1])
    # # print(temp[0])
    # rt_rot[:, -2] = temp
    # # print(rt_rot[0, -2])
    # # print(rt_rot[0])
    # rotations[:,0,:] = rt_rot
    # print(rotations[0])

    # print(quat[0][0])
    # print(rotations[0][0])

    #制作每个节点的孩子节点的列表,这一步写对了
    # print(parents)
    children_dict = {}
    for i in range(joint_num):
        children = np.where(parents==i)[0]
        children_dict[str(i)] = children
    # for k, v in children_dict.items():
    #     print(k + '\'s_dataset children is:', end='')
    #     print(v)

    # coding=UTF-8
    # filename = dancers + '_' + str(start_poses) + '.bvh'
    # file_path = os.path.join(store_path, filename)
    # print(file_path)
    with open(file_path, 'w') as file_obj:
        # file_obj.write("Add two words\n")
        write_root(file_obj, 'zyx', pos_order, offsets, children_dict, names)
        write_motion(file_obj, rt_pos, rotations, frames, frameRate, joint_num)


def write_root(file_obj, rot_order, pos_order, offsets, children_dict, names):
    file_obj.write("HIERARCHY\n")
    file_obj.write("ROOT %s\n" % names[0])
    file_obj.write("{\n")
    file_obj.write("\tOFFSET %.4f %.4f %.4f\n" % (offsets[0][0], offsets[0][1], offsets[0][2]))
    file_obj.write("\tCHANNELS 6 %sposition %sposition %sposition %srotation %srotation %srotation\n"
                   % (pos_order[0].upper(), pos_order[1].upper(), pos_order[2].upper(),
                      rot_order[0].upper(), rot_order[1].upper(), rot_order[2].upper()))

    tag = '\t'
    children = children_dict["0"]
    for joint in children:
        # print(joint)
        write_joint(file_obj, rot_order, offsets, children_dict, names, tag, joint)

    file_obj.write("}\n")

def write_joint(file_obj, rot_order, offsets, children_dict, names, tag, joint):
    file_obj.write(tag + "JOINT %s\n" % names[joint])
    file_obj.write(tag + "{\n")
    file_obj.write(tag + "\tOFFSET %.4f %.4f %.4f\n" % (offsets[joint][0], offsets[joint][1], offsets[joint][2]))
    file_obj.write(tag + "\tCHANNELS 3 %srotation %srotation %srotation\n" %
                   (rot_order[0].upper(), rot_order[1].upper(), rot_order[2].upper()))

    new_tag = tag + "\t"
    children = children_dict[str(joint)]
    if len(children) != 0:
        for j in children:
            # print(j)
            write_joint(file_obj, rot_order, offsets, children_dict, names, new_tag, j)
    else:
        # print("this is end site")
        file_obj.write(tag + "\tEnd Site\n")
        file_obj.write(tag + "\t{\n")
        file_obj.write(tag + "\t\tOFFSET 0 0 0\n")
        file_obj.write(tag + "\t}\n")

    file_obj.write(tag + "}\n")

def write_motion(file_obj, rt_pos, rotations, frames, frameRate, joint_num):
    file_obj.write("MOTION\n")
    file_obj.write("Frames: %d\n" % frames)
    file_obj.write("Frame Time: %.8f\n" % (1/frameRate))

    for i in range(frames):
        rt = rt_pos[i]
        rot = rotations[i]
        message = "%.5f %.5f %.5f " % (rt[0], rt[1], rt[2])
        for j in range(joint_num):
            message += "%.5f %.5f %.5f " % (rot[j][0], rot[j][1], rot[j][2])
        message += "\n"
        file_obj.write(message)

if __name__ == '__main__':

    quat = np.array([ 0.74189869, -0.03518094, -0.66791437,  0.0473185 ])
    quat = quat.reshape((1,1,4))
    result = quaternion2euler(quat, order='zyx')
    print(result)

    filepath = '../motion/Pro/Maritsa_Elia_Excited_v1.bvh'
    rotations, rot_order, root_positions, pos_order, offsets, parents, names, frameRate = load(filepath)
    quat = euler2quaternion(rotations, rot_order)

    store_path = './'
    save(quat, root_positions, rot_order, pos_order, offsets, parents, names, frameRate, 0, 'Maritsa', store_path)


