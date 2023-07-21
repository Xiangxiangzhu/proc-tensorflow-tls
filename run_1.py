import os, sys, time

from src.argparse import parse_cl_args
from src.distprocs import DistProcs


def main():
    start_t = time.time()
    print('start running main...')
    args = parse_cl_args()

    args.sim = 'double'
    args.tsc = 'dqn'
    args.lr = 0.00005
    args.lre = 0.0000001
    args.nreplay = 15000
    args.nsteps = 2
    args.target_freq = 128
    args.updates = 15000
    args.batch = 32
    args.save = False
    args.nogui = True
    args.n_hidden = 3
    args.mode = 'train'
    args.gmin = 5

    print(args)

    distprocs = DistProcs(args, args.tsc, args.mode)
    distprocs.run()
    print(args)
    print('...finish running main')
    print('run time ' + str((time.time() - start_t) / 60))


if __name__ == '__main__':
    main()
