from collections import OrderedDict, defaultdict
from sumolib.net import readNet

from args import Args


class Config:
    def __init__(self):

        # path
        self.path_network   = Args.path_network
        self.path_detectors = Args.path_detectors

        # intersection ids
        self.node_ids = Args.node_ids

        # detector length
        self.e2_length = Args.e2_length

        # setup
        self.infos = self.set_intersections()


    def set_phases(self):
        '''Define the phases for the traffic lights and returns a dictionary'''
        phases = {}
        phases['green']  = ('grrgGrgrrgGr', 'grrgrGgrrgrG', 'gGrgrrgGrgrr', 'grGgrrgrGgrr')
        phases['yellow'] = ('grrgyrgrrgyr', 'grrgrygrrgry', 'gyrgrrgyrgrr', 'grygrrgrygrr')
        return phases


    def set_neighbors(self):
        '''Define the neighbors for the nodes and return a dictionary'''
        node2neighbors = {}
        node2neighbors['A0'] = ('B0', )
        node2neighbors['B0'] = ('A0', 'C0')
        node2neighbors['C0'] = ('B0', )
        return node2neighbors


    def set_detectors(self):
        '''
        Set up the detectors for the nodes
        https://sumo.dlr.de/docs/Tools/Sumolib.html
        https://sumo.dlr.de/pydoc/sumolib.net.html
        
        Returns a dictionary mapping node ids to detector ids
        '''

        net = readNet(self.path_network)
        xml = ['<additional>\n']
        node2detectors = defaultdict(list)
        seen = set()
        for node_id in self.node_ids:
            node = net.getNode(node_id)
            for c in sorted(node.getConnections(), key=lambda x:x.getJunctionIndex()):
                lane = c.getFromLane()
                lane_id = lane.getID()

                if lane_id in seen:
                    continue

                pos = lane.getLength() - self.e2_length - 1
                xml.append(f'  <laneAreaDetector file="NUL" freq="86400" id="{lane_id}" lanes="{lane_id}" pos="{pos}" endPos="-1"/>\n')
                node2detectors[node_id].append(lane_id)
                seen.add(lane_id)
                
        xml.append('</additional>')
        xml = ''.join(xml)

        if self.path_detectors is not None:
            with open(self.path_detectors, 'w') as f:
                f.write(xml)

        return node2detectors


    def set_intersections(self):
        '''Set up the intersections for the nodes and return a dictionary'''
        infos = OrderedDict()
        phases = self.set_phases()
        node2neighbors = self.set_neighbors()
        node2detectors = self.set_detectors()
        for node_id in self.node_ids:
            infos[node_id] = {}
            infos[node_id]['node_id'] = node_id
            infos[node_id]['phases'] = phases
            infos[node_id]['neighbors'] = node2neighbors[node_id]
            infos[node_id]['detectors'] = node2detectors[node_id]
        return infos
