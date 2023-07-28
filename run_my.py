import os, sys, time

from src.argparse import parse_cl_args
from src.distprocs import DistProcs


def main():
    # use_double = True
    use_double = False

    start_t = time.time()
    print('start running main...')
    args = parse_cl_args()

    if use_double:
        args.sim = 'test'
        args.n = 1
        args.tsc = 'dqn'
        args.load = False
        args.nogui = True
        args.mode = 'test'
    else:
        # args.sim = 'single'
        args.sim = 'double'
        args.n = 1
        args.tsc = 'dqn'
        args.load = False
        args.nogui = False
        args.mode = 'train'

    print(args)

    distprocs = DistProcs(args, args.tsc, args.mode)
    distprocs.run()
    print(args)
    print('...finish running main')
    print('run time ' + str((time.time() - start_t) / 60))


if __name__ == '__main__':
    main()
