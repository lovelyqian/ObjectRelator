from argparse import ArgumentParser
from vos_benchmark.benchmark import benchmark
"""
Data paths
"""
parser = ArgumentParser()
parser.add_argument('-g', '--gt', help='Path to a folder containing folders of ground-truth masks')
parser.add_argument('-m',
                    '--mask',
                    help='Path to a folder containing folders of masks to be evaluated')
parser.add_argument('-n',
                    '--num_processes',
                    default=16,
                    type=int,
                    help='Number of concurrent processes')
parser.add_argument(
    '-s',
    '--strict',
    help='Make sure every video in the ground-truth has a corresponding video in the prediction',
    action='store_true')

# https://github.com/davisvideochallenge/davis2017-evaluation/blob/d34fdef71ce3cb24c1a167d860b707e575b3034c/davis2017/evaluation.py#L85
parser.add_argument(
    '-d',
    '--do_not_skip_first_and_last_frame',
    help=
    'By default, we skip the first and the last frame in evaluation following DAVIS semi-supervised evaluation.'
    'They should not be skipped in unsupervised evaluation.',
    action='store_true')
args = parser.parse_args()

benchmark([args.gt], [args.mask],
          args.strict,
          args.num_processes,
          verbose=True,
          skip_first_and_last=not args.do_not_skip_first_and_last_frame)
