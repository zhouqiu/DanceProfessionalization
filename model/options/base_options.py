import argparse
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):

        self.parser.add_argument("--gpu_ids", type=str, default="0,1")
        self.parser.add_argument("--net_root", type=str, default='./all_nets') # root dictionary of all nets
        self.parser.add_argument("--net_path", type=str, default='dance')  # dictionary of current net
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--iters", type=int, default=400)
        self.parser.add_argument("--store_freq", type=int, default=20)

        self.parser.add_argument("--num_workers", type=int, default=4)

        self.parser.add_argument("--non_path", type=str, default="../data/train_val_testset", help='')
        self.parser.add_argument("--music_path", type=str, default="../data/mp3", help='')
        self.parser.add_argument("--dtw_path", type=str, default="", help='')
        self.parser.add_argument("--total_length", type=int, default=3000)
        self.parser.add_argument("--mtype", type=str, default="origin", help="[origin|samegenre|diff]")

        self.parser.add_argument("--lr_gen", type=float, default=0.0001)
        self.parser.add_argument("--kernel_size", type=int, default=32)

        # dtw net
        self.parser.add_argument("--isnorm", action='store_true', help="whether use instance norm in network.")
        self.parser.add_argument("--isConv", action='store_true', help="whether use instance norm in network.")
        self.parser.add_argument("--isTrans", action='store_true', help="whether use instance norm in network.")
        self.parser.add_argument("--useTripletloss", action='store_true')
        self.parser.add_argument("--corr_w", type=float, default=1.0)

        # autoencoder net
        self.parser.add_argument("--isFinetune", action='store_true', help="whether Finetune.")
        self.parser.add_argument("--rec_w", type=float, default=1.0)
        self.parser.add_argument("--velo_w", type=float, default=0.0)




    def parse(self):
        self.initialize()
        opt = self.parser.parse_args()

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


if __name__ == '__main__':
    opt = BaseOptions().parse()







