import sys, os, subprocess, time
from multiprocessing import *

from tensorflow.python.framework.ops import disable_eager_execution

from src.simproc import SimProc
from src.learnerproc import LearnerProc
from src.networkdata import NetworkData
from src.sumosim import SumoSim

import numpy as np
import xml.etree.ElementTree as ET


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
    elif sim_str == 'test':
        cfg_fp = 'networks/test.sumocfg'
        net_fp = 'networks/acosta_buslanes.net.xml'
    return cfg_fp, net_fp


class DistProcs:
    def __init__(self, args, tsc, mode):
        self.args = args
        rl_tsc = ['ddpg', 'dqn']
        traditional_tsc = ['websters', 'maxpressure', 'sotl', 'uniform']

        # depending on tsc alg, different hyper param checks
        if tsc in rl_tsc:
            # disable_eager_execution()
            # need actors and at least one learner
            if mode == 'train':
                # ensure we have at least one learner
                if args.l < 1:
                    args.l = 1
            elif mode == 'test':
                # no learners necessary for testing
                if args.l > 0:
                    args.l = 0
        elif tsc in traditional_tsc:
            # traditional tsc doesn't require learners
            if args.l > 0:
                args.l = 0
        else:
            print('Input argument tsc ' + str(tsc) + ' not found, please provide valid tsc.')
            return

        # ensure at least one sim, otherwise no point in running program
        if args.n < 0:
            args.n = 1

        # if sim arg provided, use to get cfg and netfp
        # otherwise, continue with args default
        if args.sim:
            args.cfg_fp, args.net_fp = get_sim(args.sim)

        args.nreplay = int(args.nreplay / args.nsteps)

        barrier = Barrier(args.n + args.l)
        print("!!!!!!!###### barrier is ", args.n + args.l)

        nd = NetworkData(args.net_fp)
        netdata = nd.get_net_data()
        print("### all inters are ", netdata['inter'].keys())

        # create a dummy sim to get tsc data for creating nn
        # print('creating dummy sim for netdata...')
        sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc, True, netdata, args, -1)
        sim.gen_sim(no_addition=True)
        netdata = sim.update_netdata()
        self.tsc_temp = sim.tsc
        sim.close()

        # mark: change_additional file here!!!
        # print('...finished with dummy sim')
        tsc_ids = netdata['inter'].keys()
        tsc_names = [n for n in netdata['inter']]

        self.generate_additional_file(tsc_names)
        self.modify_additional_template()

        sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc, True, netdata, args, -1)
        sim.gen_sim()
        netdata = sim.update_netdata()
        sim.close()

        # create mp dict for sharing
        # reinforcement learning stats
        # Todo: comment back!!!
        # rl_stats = self.create_mp_stats_dict(tsc_ids)
        # exp_replays = self.create_mp_exp_replay(tsc_ids)

        rl_stats = self.create_mp_stats_dict_1(tsc_ids)
        exp_replays = self.create_mp_exp_replay_1(tsc_ids)

        eps_rates = self.get_exploration_rates(args.eps, args.n, args.mode, args.sim)
        print("eps_rates is ", eps_rates)
        offsets = self.get_start_offsets(args.mode, args.sim_len, args.offset, args.n)
        print("offsets is ", offsets)

        # create sumo sim procs to generate experiences
        sim_procs = [SimProc(i, args, barrier, netdata, rl_stats, exp_replays, eps_rates[i], offsets[i]) for i in
                     range(args.n)]

        # create learner procs which are assigned tsc/rl agents
        # to compute neural net updates for
        if args.l > 0:
            learner_agents = self.assign_learner_agents(tsc_ids, args.l)
            print('===========LEARNER AGENTS')
            for l in learner_agents:
                print('============== ' + str(l))
            learner_procs = [LearnerProc(i, args, barrier, netdata, learner_agents[i], rl_stats, exp_replays) for i in
                             range(args.l)]
        else:
            learner_procs = []

        self.procs = sim_procs + learner_procs

    def modify_additional_template(self):
        all_inter_phase = {}
        for inter in self.tsc_temp:
            all_inter_phase[inter] = self.tsc_temp[inter].phase_inter
        # 解析XML文件
        tree = ET.parse("networks/tls_addition.xml")
        root = tree.getroot()

        for road in all_inter_phase:
            # 查找需要修改的元素
            tl_logic = root.find("./tlLogic[@id='{}']".format(road))

            for idx, phase_item in enumerate(all_inter_phase[road]):
                print("id is ", idx)
                print("phase item is ", phase_item)
                # 修改第一个state的值
                phase = tl_logic.find("./phase[{}]".format(idx + 1))
                phase.set('state', phase_item)

        # 保存修改后的XML文件
        tree.write('networks/tls_addition.xml')

    def generate_additional_file(self, tsc_names):
        # 解析XML文件
        tls_dir = "networks/tls_addition_template.xml"
        tree = ET.parse(tls_dir)
        root = tree.getroot()

        # 修改第一个tsc的id
        tl_logic = root.find("./tlLogic[@id='initial_id']")
        tl_logic.set('id', tsc_names[0])

        if len(tsc_names) > 1:
            # 找到tlLogic元素
            tl_logic = root.find('.//tlLogic')

            # 复制并修改tlLogic元素
            for i in range(len(tsc_names) - 1):
                print("add tlLogic")
                new_tl_logic = ET.Element('tlLogic')
                new_tl_logic.set('id', tsc_names[i + 1])
                new_tl_logic.set('programID', tl_logic.get('programID'))
                new_tl_logic.set('offset', tl_logic.get('offset'))
                new_tl_logic.set('type', tl_logic.get('type'))

                # 复制phase子元素
                for phase in tl_logic.findall('phase'):
                    new_phase = ET.Element('phase')
                    new_phase.set('duration', phase.get('duration'))
                    new_phase.set('state', phase.get('state'))
                    new_tl_logic.append(new_phase)

                # 将新的tlLogic元素添加到根元素下
                root.append(new_tl_logic)

        # 将修改后的XML写入文件
        tree.write('networks/tls_addition.xml')

    def run(self):
        print('Starting up all processes...')
        ###start everything   
        for p in self.procs:
            p.start()

        ###join when finished
        for p in self.procs:
            p.join()

        print('...finishing all processes')

    # inter
    def create_mp_stats_dict(self, tsc_ids):
        ###use this mp shared dict for data between procs
        manager = Manager()
        rl_stats = manager.dict({})
        for i in tsc_ids:
            rl_stats[i] = manager.dict({})
            rl_stats[i]['n_exp'] = 0
            rl_stats[i]['updates'] = 0
            rl_stats[i]['max_r'] = 1.0
            rl_stats[i]['online'] = None
            rl_stats[i]['target'] = None
            rl_stats['n_sims'] = 0
            rl_stats['total_sims'] = 104
            rl_stats['delay'] = manager.list()
            rl_stats['queue'] = manager.list()
            rl_stats['throughput'] = manager.list()

        return rl_stats

    def create_mp_stats_dict_1(self, tsc_ids):
        ###use this mp shared dict for data between procs
        # manager = Manager()
        rl_stats = {}
        for i in tsc_ids:
            rl_stats[i] = {}
            rl_stats[i]['n_exp'] = 0
            rl_stats[i]['updates'] = 0
            rl_stats[i]['max_r'] = 1.0
            rl_stats[i]['online'] = None
            rl_stats[i]['target'] = None
            rl_stats['n_sims'] = 0
            rl_stats['total_sims'] = 104
            rl_stats['delay'] = []
            rl_stats['queue'] = []
            rl_stats['throughput'] = []

        return rl_stats

    # inter
    def create_mp_exp_replay(self, tsc_ids):
        ###create shared memory for experience replay 
        # (governs agents appending and learners accessing and deleting)
        manager = Manager()
        return manager.dict({tsc: manager.list() for tsc in tsc_ids})

    def create_mp_exp_replay_1(self, tsc_ids):
        ###create shared memory for experience replay
        # (governs agents appending and learners accessing and deleting)
        # manager = Manager()
        return {tsc: [] for tsc in tsc_ids}

    # inter
    def assign_learner_agents(self, agents, n_learners):
        learner_agents = [[] for _ in range(n_learners)]
        for agent, i in zip(agents, range(len(agents))):
            learner_agents[i % n_learners].append(agent)
        ##list of lists, each sublit is the agents a learner is responsible for
        return learner_agents

    # inter
    def get_exploration_rates(self, eps, n_actors, mode, net):
        if mode == 'test':
            return [eps for _ in range(n_actors)]
        elif mode == 'train':
            if net == 'lust':
                # for lust we restrict the exploration rates
                e = [1.0, 0.5, eps]
                erates = []
                for i in range(n_actors):
                    erates.append(e[i % len(e)])
                return erates
            else:
                return np.linspace(1.0, eps, num=n_actors)

    # inter
    def get_start_offsets(self, mode, simlen, offset, n_actors):
        if mode == 'test':
            return [0] * n_actors
        elif mode == 'train':
            return np.linspace(0, simlen * offset, num=n_actors)
