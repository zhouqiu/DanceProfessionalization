import argparse
import sys
# sys.path.append('./')


class Options:
    def __init__(self, argv):

        if "--" not in argv:
            self.argv = []  # as if no args are passed
        else:
            self.argv = argv[argv.index("--") + 1:]

        usage_text = (
                "Run blender in background mode with this script:"
                "  blender --background --python [main_python_file] -- [options]"
        )

        self.parser = argparse.ArgumentParser(description=usage_text)
        self.initialize()
        self.args = self.parser.parse_args(self.argv)

    def initialize(self):
        self.parser.add_argument('--bvh_path1', type=str, default='./example_bvh/example1.bvh', help='path of input bvh file1')
        self.parser.add_argument('--bvh_path2', type=str, default='./example_bvh/example2.bvh', help='path of input bvh file2')

        self.parser.add_argument('--save_path', type=str, default='./results/', help='path of output video file')
        self.parser.add_argument('--music_path', type=str, default="./examples/mBR0_0.mp3", help='path of music file')
        self.parser.add_argument('--render_engine', type=str, default='eevee',
                                 help='name of preferable render engine: cycles, eevee')
        self.parser.add_argument('--render', action='store_true', default=False, help='render an output video')
        # rendering parameters
        self.parser.add_argument('--resX', type=int, default=1920, help='x resolution')
        self.parser.add_argument('--resY', type=int, default=1080, help='y resolution')

        self.parser.add_argument('--frame_start', type=int, default=0, help='the index of the first rendered frame')
        self.parser.add_argument('--frame_step', type=int, default=60, help='the step of rendered frame,only for generate pict')
        self.parser.add_argument('--frame_end', type=int, default=180, help='the index of the last rendered frame')

        self.parser.add_argument('--color', type=str, help='read colormap')
        self.parser.add_argument('--color2', type=str, help='read colormap')

        self.parser.add_argument('--camera_position', type=str, help='[near|far]')

    def parse(self):
            return self.args
