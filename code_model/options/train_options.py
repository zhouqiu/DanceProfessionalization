from options.base_options import BaseOptions

class TrainOptions(BaseOptions):

    def __init__(self):
        BaseOptions.__init__(self)

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--phase", type=str, default='train', help='[train|test]')
        self.parser.add_argument("--dataset_mode", type=str, default='train', help='[train|test]')
        self.parser.add_argument("--isSlice", action='store_true')
        self.parser.add_argument("--slice_pad", type=int, default=30)

        self.parser.add_argument("--continueTrain", action='store_true')
        self.parser.add_argument("--model_epoch", type=int, default=0, help="epoch of continue train")


if __name__ == '__main__':
    opt = TrainOptions().parse()
