from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--phase", type=str, default='test', help='[train|test]')
        self.parser.add_argument("--dataset_mode", type=str, default='test', help='[train|test]')
        self.parser.add_argument("--isSlice", action='store_true')
        self.parser.add_argument("--slice_pad", type=int, default=30)

        self.parser.add_argument("--continueTrain", action='store_true', help="default false.")
        self.parser.add_argument("--model_epoch", type=int, default=400)

        self.parser.add_argument("--result_root", type=str, default="./all_results")
        self.parser.add_argument("--result_path", type=str, default="dance")

        self.parser.add_argument('--save_pkg_num', type=int, default=4, help='which pkg motion to save')










