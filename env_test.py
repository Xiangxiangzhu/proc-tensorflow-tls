from src.networkdata import NetworkData
from src.sumosim import SumoSim
from src.argparse import parse_cl_args

args = parse_cl_args()

args.sim = 'single'
args.n = 1
args.tsc = 'dqn'
args.load = True
args.nogui = False
args.mode = 'test'

print(args)


def get_sim(sim_str):
    if sim_str == 'lust':
        cfg_fp = 'networks/lust/scenario/dua.actuated.sumocfg'
        net_fp = 'networks/lust/scenario/lust.net.xml'
    elif sim_str == 'single':
        cfg_fp = 'networks/single.sumocfg'
        net_fp = 'networks/single.net.xml'
    elif sim_str == 'double':
        cfg_fp = 'networks/double.sumocfg'
        net_fp = 'networks/double.net.xml'
    return cfg_fp, net_fp


if args.sim:
    args.cfg_fp, args.net_fp = get_sim(args.sim)

nd = NetworkData(args.net_fp)
netdata = nd.get_net_data()

sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc, False, netdata, args, -1)
sim.gen_sim()
netdata = sim.update_netdata()
sim.close()
