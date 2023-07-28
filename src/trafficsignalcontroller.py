import os, sys, copy
import math

import numpy as np

from collections import deque
from operator import itemgetter
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

from src.trafficmetrics import TrafficMetrics


class TrafficSignalController:
    """Abstract base class for all traffic signal controller.

    Build your own traffic signal controller by implementing the follow methods.
    """

    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t):
        self.conn = conn
        self.id = tsc_id
        self.netdata = netdata

        # reordering tsl index from north in a clockwise direction
        self.reshape_index()

        self.red_t = red_t
        self.yellow_t = yellow_t
        self.green_phases = self.get_tl_green_phases()
        self.phase_time = 0
        self.all_red = len((self.green_phases[0])) * 'r'
        self.phase = self.all_red
        self.phase_lanes = self.phase_lanes(self.green_phases)

        # if not use general state one can use incoming veh or all lane veh as state
        self.use_full_lane_state = True

        # create subscription for this traffic signal junction to gather
        # vehicle information efficiently
        self.conn.junction.subscribeContext(tsc_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 150,
                                            [traci.constants.VAR_SIGNALS,
                                             traci.constants.VAR_LANEPOSITION,
                                             traci.constants.VAR_SPEED,
                                             traci.constants.VAR_LANE_ID])

        # todo:get_direction
        # self.conn.junction.getLaneChangeState()
        # self.conn.junction.getLaneChangeStatePretty()

        # get all incoming lanes to intersection
        self.incoming_lanes = set()
        for p in self.phase_lanes:
            for l in self.phase_lanes[p]:
                self.incoming_lanes.add(l)

        self.incoming_lanes = sorted(list(self.incoming_lanes))

        self.all_lanes = set()
        self.all_road_to_lanes = {}
        self.all_roads = set()
        for r_in in self.netdata["inter"][self.id]['incoming']:
            self.all_roads.add(r_in)
        for r_out in self.netdata["inter"][self.id]['outgoing']:
            self.all_roads.add(r_out)

        self.all_roads = sorted(list(self.all_roads))
        for road_ in self.all_roads:
            temp_lane = self.netdata['edge'][road_]['lanes']
            for l_ in temp_lane:
                self.all_lanes.add(l_)
            # add road to lanes dict
            self.all_road_to_lanes[road_] = sorted(temp_lane)
        self.all_lanes = sorted(self.all_lanes)
        self.all_lane_capacity = np.array(
            [float(self.netdata['lane'][lane]['length']) / 7.5 for lane in self.all_lanes])

        # lane capacity is the lane length divided by the average vehicle length+stopped headway
        self.lane_capacity = np.array(
            [float(self.netdata['lane'][lane]['length']) / 7.5 for lane in self.incoming_lanes])

        # for collecting various traffic metrics at the intersection
        # can be extended in trafficmetric.py class to collect new metrics
        if mode == 'train':
            self.metric_args = ['delay']
        if mode == 'test':
            self.metric_args = ['queue', 'delay']
        self.trafficmetrics = TrafficMetrics(tsc_id, self.incoming_lanes, netdata, self.metric_args, mode)

        self.ep_rewards = []

    # lane sorting rules
    @staticmethod
    def custom_sort_rule(char):
        order = {'s': 2, 'r': 1, 'l': 3}
        return order.get(char, 0)

    @staticmethod
    def rearrange_string(string, index):
        rearranged_string = ''.join(string[i] for i in index)
        return rearranged_string

    def run(self):
        data, all_lane_data = self.get_subscription_data()
        self.trafficmetrics.update(data)
        self.update(data, all_lane_data)
        self.increment_controller()

    def get_metrics(self):
        for m in self.metric_args:
            metric = self.trafficmetrics.get_metric(m)

    def get_traffic_metrics_history(self):
        return {m: self.trafficmetrics.get_history(m) for m in self.metric_args}

    def increment_controller(self):
        if self.phase_time == 0:
            ###get new phase and duration
            next_phase = self.next_phase()
            print("yyyyy next phase is ", next_phase)
            acting_phase = self.reorder_phase(next_phase)
            # change phase of a given tl
            self.conn.trafficlight.setRedYellowGreenState(self.id, acting_phase)

            # change allowed speed for any edge or lane
            self.conn.edge.setMaxSpeed("gneE1", 66.6)
            self.conn.lane.setMaxSpeed("gneE1_0", 77.7)

            # close given lanes or edges
            self.conn.lane.setDisallowed("gneE1_0", ["all"])
            self.conn.edge.setDisallowed("-gneE17", ["all"])

            # reopen given lanes or edges
            self.conn.lane.setAllowed("gneE1_0", ["all"])
            self.conn.edge.setAllowed("-gneE17", ["all"])

            # disable given types of vehcles for any lane or edge
            self.conn.lane.setDisallowed("gneE1_1", ["private"])


            # self.conn.edge.setDisallowed("gneE1_2", ["all"])
            # self.conn.edge.setDisallowed("-gneE1", ["all"])




            self.phase = next_phase
            self.phase_time = self.next_phase_duration()
        self.phase_time -= 1

    # inter
    def reshape_index(self):
        print("##### reshaped lane tsc index #####")
        lane_set = set(self.conn.trafficlight.getControlledLanes(self.id))
        incoming_roads = [self.netdata['lane'][lane]['edge'] for lane in lane_set]
        # incoming_roads = [r for r in self.netdata['inter'][self.id]['incoming']]

        # reorder incoming roads
        heading = {}
        for road in incoming_roads:
            road_from = self.netdata['edge'][road]['coord'][0]
            road_to = self.netdata['edge'][road]['coord'][1]
            angle_x = road_from[0] - road_to[0]
            angle_y = road_from[1] - road_to[1]
            theta = math.atan2(angle_y, angle_x)
            degrees = math.degrees(theta)
            adjusted_degrees = (90 - degrees) % 360
            print("road is ", road)
            print("theta is ", adjusted_degrees)
            heading[road] = adjusted_degrees

        sorted_heading = sorted(heading.items(), key=itemgetter(1))
        sorted_dict = {k: v for k, v in sorted_heading}

        # create tsc index Dataframe : merged_df
        df1 = pd.DataFrame(list(self.netdata["inter"][self.id]["tlsindex"].items()), columns=['tsl_index', 'lane_name'])
        df2 = pd.DataFrame(list(self.netdata["inter"][self.id]["tlsindexdir"].items()),
                           columns=['tsl_index', 'lane_dir'])

        # concatenate DataFrame
        merged_df = pd.concat([df1, df2['lane_dir']], axis=1)

        merged_df['reordered_index'] = [-1 for _ in range(len(merged_df))]

        # reorder tsc index
        temp_id = 0
        for road in sorted_dict:
            lane_reverse = False
            lane_list = self.netdata["edge"][road]["lanes"]
            if "r" in self.netdata["lane"][lane_list[0]]['movement'] or "l" in self.netdata["lane"][lane_list[-1]][
                'movement']:
                lane_reverse = False
            elif "l" in self.netdata["lane"][lane_list[0]]['movement'] or "r" in self.netdata["lane"][lane_list[-1]][
                'movement']:
                lane_reverse = True
            if lane_reverse:
                lane_list.reverse()
            for lane_ in lane_list:
                #####
                temp_dir = self.netdata["lane"][lane_]['movement']
                sorted_lane_dir = ''.join(sorted(temp_dir, key=self.custom_sort_rule))
                for d in sorted_lane_dir:
                    matching_keys = int(merged_df.loc[(merged_df['lane_name'] == lane_) & (
                            merged_df['lane_dir'] == d), 'tsl_index'].values[0])
                    merged_df.loc[merged_df['tsl_index'] == matching_keys, 'reordered_index'] = temp_id
                    temp_id += 1

        # generate phase for this inter
        road_phases = {}
        for road in sorted_dict:
            movement_temp = ""
            lane = self.netdata["edge"][road]['lanes']
            for l in lane:
                movement_temp += self.netdata['lane'][l]["movement"]
            n_r = movement_temp.count("r")
            n_s = movement_temp.count("s")
            n_l = movement_temp.count("l")

            l_t_phase = n_r * "G" + n_s * "r" + n_l * "G"
            s_t_phase = n_r * "G" + n_s * "G" + n_l * "r"
            l_s_phase = n_r * "G" + n_s * "G" + n_l * "G"
            stop_phase = n_r * "G" + n_s * "r" + n_l * "r"

            road_phases[road] = [l_t_phase, s_t_phase, l_s_phase, stop_phase]

        phase_1, phase_2, phase_3, phase_4, phase_5, phase_6, phase_7, phase_8 = "", "", "", "", "", "", "", ""
        for idx, road in enumerate(sorted_dict):
            if len(sorted_dict) == 4:
                if idx == 0 or idx == 2:
                    phase_1 += road_phases[road][0]
                    phase_2 += road_phases[road][1]
                    phase_3 += road_phases[road][3]
                    phase_4 += road_phases[road][3]
                    if idx == 0:
                        phase_5 += road_phases[road][2]
                        phase_7 += road_phases[road][3]
                    if idx == 2:
                        phase_7 += road_phases[road][2]
                        phase_5 += road_phases[road][3]
                    phase_6 += road_phases[road][3]
                    phase_8 += road_phases[road][3]
                else:
                    phase_1 += road_phases[road][3]
                    phase_2 += road_phases[road][3]
                    phase_3 += road_phases[road][0]
                    phase_4 += road_phases[road][1]
                    if idx == 1:
                        phase_6 += road_phases[road][2]
                        phase_8 += road_phases[road][3]
                    if idx == 3:
                        phase_8 += road_phases[road][2]
                        phase_6 += road_phases[road][3]
                    phase_5 += road_phases[road][3]
                    phase_7 += road_phases[road][3]
            elif len(sorted_dict) == 3:
                phase_1 += road_phases[road][3]
                phase_2 += road_phases[road][3]
                phase_3 += road_phases[road][3]
                phase_4 += road_phases[road][3]
                phase_5 += road_phases[road][3]
                if idx == 0:
                    phase_6 += road_phases[road][2]
                    phase_7 += road_phases[road][3]
                    phase_8 += road_phases[road][3]
                if idx == 1:
                    phase_6 += road_phases[road][3]
                    phase_7 += road_phases[road][2]
                    phase_8 += road_phases[road][3]
                if idx == 2:
                    phase_6 += road_phases[road][3]
                    phase_7 += road_phases[road][3]
                    phase_8 += road_phases[road][2]
            else:
                phase_1 += road_phases[road][2]
                phase_2 += road_phases[road][2]
                phase_3 += road_phases[road][2]
                phase_4 += road_phases[road][2]
                phase_5 += road_phases[road][2]
                phase_6 += road_phases[road][2]
                phase_7 += road_phases[road][2]
                phase_8 += road_phases[road][2]

        self.phase_inter = [phase_1, phase_2, phase_3, phase_4, phase_5, phase_6, phase_7, phase_8]

        df_sorted = merged_df.sort_values('reordered_index')
        self.reordered_tsl_index = df_sorted

    # inter
    def reorder_phase(self, phase):
        initial_index = self.reordered_tsl_index['tsl_index'].tolist()
        reordered_index = self.reordered_tsl_index['reordered_index'].tolist()
        assert len(initial_index) == len(reordered_index) == len(phase), "plz check tsc phase number !!!"
        rearranged_phase = self.rearrange_string(phase, initial_index)

        return rearranged_phase

    def get_intermediate_phases(self, phase, next_phase):
        if phase == next_phase or phase == self.all_red:
            return []
        else:
            yellow_phase = ''.join([p if p == 'r' else 'y' for p in phase])
            return [yellow_phase, self.all_red]

    def next_phase(self):
        raise NotImplementedError("Subclasses should implement this!")

    def next_phase_duration(self):
        raise NotImplementedError("Subclasses should implement this!")

    def update(self, data):
        """Implement this function to perform any
           traffic signal class specific control/updates 
        """
        raise NotImplementedError("Subclasses should implement this!")

    def get_subscription_data(self):
        # use SUMO subscription to retrieve vehicle info in batches
        # around the traffic signal controller
        tl_data = self.conn.junction.getContextSubscriptionResults(self.id)

        # create empty incoming lanes for use else where
        lane_vehicles = {l: {} for l in self.incoming_lanes}
        all_lane_vehicles = {l: {} for l in self.all_lanes}
        if tl_data is not None:
            for v in tl_data:
                lane = tl_data[v][traci.constants.VAR_LANE_ID]
                if lane not in lane_vehicles:
                    lane_vehicles[lane] = {}
                if lane not in all_lane_vehicles:
                    all_lane_vehicles[lane] = {}
                lane_vehicles[lane][v] = tl_data[v]
                all_lane_vehicles[lane][v] = tl_data[v]
        return lane_vehicles, all_lane_vehicles

    def get_tl_green_phases(self):
        logics = self.conn.trafficlight.getCompleteRedYellowGreenDefinition(self.id)
        logic = None
        for logic_ in logics:
            if logic_.programID == "my_config_tls":
                logic = logic_
        if logic is None:
            logic = logics[0]
            print("!!! configured tls is not loaded !!!")
        # yyy = self.conn.trafficlight.getControlledLinks("gneJ2")

        # get only the green phases
        green_phases = [p.state for p in logic.getPhases()
                        if 'y' not in p.state
                        and ('G' in p.state or 'g' in p.state)]

        # sort to ensure parity between sims (for RL actions)
        return sorted(green_phases)

    def phase_lanes(self, actions):
        phase_lanes = {a: [] for a in actions}
        for a in actions:
            green_lanes = set()
            red_lanes = set()
            for s in range(len(a)):
                if a[s] == 'g' or a[s] == 'G':
                    green_lanes.add(self.netdata['inter'][self.id]['tlsindex'][s])
                elif a[s] == 'r':
                    red_lanes.add(self.netdata['inter'][self.id]['tlsindex'][s])

            ###some movements are on the same lane, removes duplicate lanes
            pure_green = [l for l in green_lanes if l not in red_lanes]
            if len(pure_green) == 0:
                phase_lanes[a] = list(set(green_lanes))
            else:
                phase_lanes[a] = list(set(pure_green))
        return phase_lanes

    # helper functions for rl controllers
    def input_to_one_hot(self, phases):
        identity = np.identity(len(phases))
        one_hots = {phases[i]: identity[i, :] for i in range(len(phases))}
        return one_hots

    def int_to_input(self, phases):
        return {p: phases[p] for p in range(len(phases))}

    def get_state(self):
        # the state is the normalized density of all incoming lanes
        return np.concatenate(
            [self.general_state(self.get_normalized_density), self.general_state(self.get_normalized_queue)])

    def general_state(self, func):
        assert func in [self.get_normalized_density, self.get_normalized_queue]

        incoming_roads = self.netdata['inter'][self.id]['incoming']
        outgoing_roads = self.netdata['inter'][self.id]['outgoing']

        # generate normed veh density data of given lanes in a form of [r, s, l]
        def get_general_state(roads_):
            state_norm_dict = {}
            for r_ in roads_:
                r_turn_veh, s_turn_veh, l_turn_veh = [0], [0], [0]

                lane_ = self.all_road_to_lanes[r_]
                veh_info = func(lane_)
                for l_, v_ in zip(lane_, veh_info):
                    if "r" in self.netdata['lane'][l_]["movement"]:
                        r_turn_veh.append(v_ / len(self.netdata['lane'][l_]["movement"]))
                    if "s" in self.netdata['lane'][l_]["movement"]:
                        s_turn_veh.append(v_ / len(self.netdata['lane'][l_]["movement"]))
                    if "l" in self.netdata['lane'][l_]["movement"]:
                        l_turn_veh.append(v_ / len(self.netdata['lane'][l_]["movement"]))
                lane_state = [max(r_turn_veh), max(s_turn_veh), max(l_turn_veh)]
                state_norm_dict[r_] = lane_state
            return state_norm_dict

        # veh data for incoming lanes and outgoing lanes
        incoming_norm = get_general_state(incoming_roads)
        outgoing_norm = get_general_state(outgoing_roads)

        # attach incoming and outgoing veh data
        state_norm = []
        for lane in incoming_norm:
            state_norm += incoming_norm[lane]
        for lane in outgoing_norm:
            state_norm += outgoing_norm[lane]

        return np.array(state_norm)

    def get_normalized_density(self, lane_input=None):
        # number of vehicles in each incoming lane divided by the lane's capacity

        if lane_input is None:
            incoming_lane_density = np.array(
                [len(self.data[lane]) for lane in self.incoming_lanes]) / self.lane_capacity
            all_lane_density = np.array(
                [len(self.all_lane_data[lane]) for lane in self.all_lanes]) / self.all_lane_capacity
        else:
            input_lane_index = [self.all_lanes.index(l) for l in lane_input]

            return [len(self.all_lane_data[self.all_lanes[lane_id]]) / self.all_lane_capacity[lane_id] for lane_id in
                    input_lane_index]

        if self.use_full_lane_state:
            return all_lane_density
        else:
            return incoming_lane_density

    def get_normalized_queue(self, lane_input=None):
        lane_queues = []
        all_lane_queues = []
        for lane in self.all_lanes:
            q = 0
            for v in self.all_lane_data[lane]:
                if self.all_lane_data[lane][v][traci.constants.VAR_SPEED] < 0.3:
                    q += 1
            if lane_input is None:
                if lane in self.incoming_lanes:
                    lane_queues.append(q)
            else:
                if lane in lane_input:
                    lane_queues.append(q)
            all_lane_queues.append(q)

        if lane_input is not None:

            input_lane_index = [self.all_lanes.index(l) for l in lane_input]
            lane_queue = []

            for que, idx in zip(lane_queues, input_lane_index):
                lane_queue.append(que / self.all_lane_capacity[idx])

            return lane_queue
        if self.use_full_lane_state:
            return np.array(all_lane_queues) / self.all_lane_capacity
        else:
            return np.array(lane_queues) / self.lane_capacity

    def empty_intersection(self):
        for lane in self.incoming_lanes:
            if len(self.data[lane]) > 0:
                return False
        return True

    def get_reward(self):
        # return negative delay as reward
        delay = int(self.trafficmetrics.get_metric('delay'))
        if delay == 0:
            r = 0
        else:
            r = -delay

        self.ep_rewards.append(r)
        return r

    def empty_dtse(n_lanes, dist, cell_size):
        return np.zeros((n_lanes, int(dist / cell_size) + 3))

    def phase_dtse(phase_lanes, lane_to_int, dtse):
        phase_dtse = {}
        for phase in phase_lanes:
            copy_dtse = np.copy(dtse)
            for lane in phase_lanes[phase]:
                copy_dtse[lane_to_int[lane], :] = 1.0
            phase_dtse[phase] = copy_dtse
        return phase_dtse

    def get_dtse(self):
        dtse = np.copy(self._dtse)
        for lane, i in zip(self.incoming_lanes, range(len(self.incoming_lanes))):
            for v in self.data[lane]:
                pos = self.data[lane][v][traci.constants.VAR_LANEPOSITION]
                dtse[i, pos:pos + 1] = 1.0

        return dtse
