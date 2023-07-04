import sys, os, time
from multiprocessing import *
import tensorflow as tf

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

from src.sumosim import SumoSim
from src.nn_factory import gen_neural_networks
from src.picklefuncs import save_data
from src.helper_funcs import check_and_make_dir, get_time_now, write_to_log


class SimProc(Process):
    def __init__(self, idx, args, barrier, netdata, rl_stats, exp_replays, eps, offset):
        Process.__init__(self)
        self.idx = idx
        self.args = args
        self.barrier = barrier
        self.netdata = netdata
        self.sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc, args.nogui, netdata, args, idx)
        self.rl_stats = rl_stats
        self.exp_replays = exp_replays
        self.eps = eps

        self.offset = offset
        self.initial = True

    def run(self):
        learner = False
        if self.args.load == True and self.args.mode == 'test':
            load = True
        else:
            load = False

        neural_networks = gen_neural_networks(self.args,
                                              self.netdata,
                                              self.args.tsc,
                                              self.netdata['inter'].keys(),
                                              learner,
                                              load,
                                              self.args.n_hidden)

        print('sim proc ' + str(self.idx) + ' waiting at barrier ---------')
        write_to_log(' ACTOR #' + str(self.idx) + ' WAITING AT SYNC WEIGHTS BARRIER...')
        print('!!!!' + ' ACTOR #' + str(self.idx) + ' WAITING AT SYNC WEIGHTS BARRIER...')
        self.barrier.wait()
        write_to_log(' ACTOR #' + str(self.idx) + '  BROKEN SYNC BARRIER...')
        print('!!!!' + ' ACTOR #' + str(self.idx) + '  BROKEN SYNC BARRIER...')
        if self.args.l > 0 and self.args.mode == 'train':
            neural_networks = self.sync_nn_weights(neural_networks)
        # barrier
        # grab weights from learner or load from file
        # barrier

        if self.args.mode == 'train':
            while not self.finished_updates():
                self.run_sim(neural_networks)
                if (self.eps == 1.0 or self.eps < 0.02):
                    self.write_to_csv(self.sim.sim_stats())
                # self.write_travel_times()
                self.sim.close()

        elif self.args.mode == 'test':
            print(str(self.idx) + ' test  waiting at offset ------------- ' + str(self.offset))
            print(str(self.idx) + ' test broken offset =================== ' + str(self.offset))
            self.initial = False
            # just run one sim for stats
            self.run_sim(neural_networks)
            if (self.eps == 1.0 or self.eps < 0.02) and self.args.mode == 'test':
                self.write_to_csv(self.sim.sim_stats())
                with open(str(self.eps) + '.csv', 'a+') as f:
                    f.write('-----------------\n')
            self.write_sim_tsc_metrics()
            # self.write_travel_times()
            self.sim.close()
        print('------------------\nFinished on sim process ' + str(self.idx) + ' Closing\n---------------')

    # inter
    def run_sim(self, neural_networks):
        start_t = time.time()
        self.sim.gen_sim()

        if self.initial is True:
            # if the initial sim, run until the offset time reached
            self.initial = False
            self.sim.run_offset(self.offset)
            print(str(self.idx) + ' train  waiting at offset ------------- ' + str(self.offset) + ' at ' + str(
                get_time_now()))
            write_to_log(
                ' ACTOR #' + str(self.idx) + ' FINISHED RUNNING OFFSET ' + str(self.offset) + ' to time ' + str(
                    self.sim.t) + ' , WAITING FOR OTHER OFFSETS...')
            self.barrier.wait()
            print(str(self.idx) + ' train  broken offset =================== ' + str(self.offset) + ' at ' + str(
                get_time_now()))
            write_to_log(' ACTOR #' + str(self.idx) + '  BROKEN OFFSET BARRIER...')

        self.sim.create_tsc(self.rl_stats, self.exp_replays, self.eps, neural_networks)
        write_to_log('ACTOR #' + str(self.idx) + '  START RUN SIM...')
        self.sim.run()
        print('sim finished in ' + str(time.time() - start_t) + ' on proc ' + str(self.idx))
        write_to_log('ACTOR #' + str(self.idx) + '  FINISHED SIM...')

    # inter
    def write_sim_tsc_metrics(self):
        # get data dict of all tsc in sim
        # where each tsc has dict of all metrics
        tsc_metrics = self.sim.get_tsc_metrics()
        # create file name and path for writing metrics data
        # now = datetime.datetime.now()
        # fname = str(self.idx)+'_'+str(now).replace(" ","-")
        fname = get_time_now()
        # write all metrics to correct path
        # path = 'metrics/'+str(self.args.tsc)
        path = 'metrics/' + str(self.args.tsc)
        # save delay and queue
        for tsc in tsc_metrics:
            for m in tsc_metrics[tsc]:
                mpath = path + '/' + str(m) + '/' + str(tsc) + '/'
                check_and_make_dir(mpath)
                save_data(mpath + fname + '_' + str(self.eps) + '_.p', tsc_metrics[tsc][m])

        # save travel time
        travel_times = self.sim.get_travel_times()
        path += '/traveltime/'
        check_and_make_dir(path)
        save_data(path + fname + '.p', travel_times)

    # inter
    def write_to_csv(self, data):
        with open(str(self.eps) + '.csv', 'a+') as f:
            f.write(','.join(data) + '\n')

    # inter; determine whether to stop update
    def finished_updates(self):
        for tsc in self.netdata['inter'].keys():
            print(tsc + '  exp replay size ' + str(len(self.exp_replays[tsc])))
            print(tsc + '  updates ' + str(self.rl_stats[tsc]['updates']))
            if self.rl_stats[tsc]['updates'] < self.args.updates:
                return False
        return True

    # inter
    def sync_nn_weights(self, neural_networks):
        for nn in neural_networks:
            weights = self.rl_stats[nn]['online']
            if self.args.tsc == 'ddpg':
                # sync actor weights
                neural_networks[nn]['actor'].set_weights(weights, 'online')
            elif self.args.tsc == 'dqn':
                neural_networks[nn].set_weights(weights, 'online')
            else:
                # raise not found exceptions
                assert 0, 'Supplied RL traffic signal controller ' + str(self.args.tsc) + ' does not exist.'
        return neural_networks
