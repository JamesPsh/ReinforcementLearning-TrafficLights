import os
import sys
import numpy as np

from sumolib import checkBinary

from args import Args


class Env:
    '''This class is responsible for handling the traffic light optimization environment.'''
    def __init__(self, infos, path_config, name=0):
        '''
        Initializes the environment with given information,
        configuration path and initial state path.
        '''
        self.path_config = path_config
        self.infos = infos
        self.node_ids = Args.node_ids
        assert list(self.node_ids) == list(self.infos)
        self.path_init_state = f'data/init_state_{name}'
        assert not os.path.exists(self.path_init_state)
        self.gui = Args.gui

        self.e2_length      = Args.e2_length
        self.yellow_length  = Args.yellow_length
        self.step_length    = Args.step_length
        self.rest_length    = self.step_length - self.yellow_length
        self.episode_length = Args.episode_length
        self.episode_step   = Args.episode_step

        self.name = name


    def _start(self):
        '''
        Starts the simulation.
        https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html#importing_traci_in_a_script
        '''
        import traci
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        sumo_binary = checkBinary('sumo-gui' if self.gui else 'sumo')
        cmd = [sumo_binary, '-c', self.path_config]
        traci.start(cmd, label=self.name)
        self.conn = traci.getConnection(self.name)


    def close(self):
        '''Closes the simulation.'''
        self.conn.close()
        if os.path.exists(self.path_init_state):
            os.remove(self.path_init_state)


    def _save(self):
        '''
        Saves the current state of the simulation.
        https://sumo.dlr.de/docs/TraCI/Change_Simulation_State.html
        '''
        self.conn.simulation.saveState(self.path_init_state)


    def _load(self):
        '''
        Loads the saved state of the simulation.
        https://sumo.dlr.de/docs/TraCI/Change_Simulation_State.html
        '''
        self.conn.simulation.loadState(self.path_init_state)


    def _step(self, step_length):
        '''Advances the simulation by a given number of steps.'''
        for _ in range(step_length):
            self.conn.simulationStep()


    def _get_occupancy(self, detector_id):
        '''
        Returns the occupancy of a given detector.
        https://sumo.dlr.de/docs/TraCI/Lane_Area_Detector_Value_Retrieval.html
        '''
        return self.conn.lanearea.getLastStepOccupancy(detector_id) / 100


    def _get_queue(self, detector_id):
        '''
        Returns the queue length at a given detector.
        https://sumo.dlr.de/docs/TraCI/Lane_Area_Detector_Value_Retrieval.html
        '''
        return self.conn.lanearea.getJamLengthMeters(detector_id) / self.e2_length


    def _get_values(self):
        '''Retrieves the occupancy, queue and phase for each node.'''
        node2values = {}
        for node_id, info in self.infos.items():
            values = {}
            values['occupancy'] = [self._get_occupancy(d) for d in info['detectors']]
            values['queue'] = [self._get_queue(d) for d in info['detectors']]
            values['phase'] = [self.prev_phase_indexes[node_id]]
            node2values[node_id] = values

        states, rewards = [], []
        global_reward = 0
        for node_id, info in self.infos.items():
            occupancy = [node2values[node_id]['occupancy']]
            queue = [node2values[node_id]['queue']]
            phase = [node2values[node_id]['phase']]
            reward = -np.sum(node2values[node_id]['queue'])
            global_reward -= np.sum(node2values[node_id]['queue'])
            for nnode_id in info['neighbors']:
                occupancy.append(node2values[nnode_id]['occupancy'])
                queue.append(node2values[nnode_id]['queue'])
                phase.append(node2values[nnode_id]['phase'])
                reward -= np.sum(node2values[nnode_id]['queue']) * 0.5

            states.append(np.concatenate(occupancy + queue + phase, dtype=np.float32))
            rewards.append(reward)
            
        return states, rewards, global_reward


    def _set_yellows(self, actions):
        '''Sets the yellow phase for nodes based on the given actions.'''
        for node_id, action in zip(self.node_ids, actions):

            if action == 0:
                continue

            info = self.infos[node_id]
            prev_idx = self.prev_phase_indexes[node_id]
            idx = (prev_idx + 1) % len(info['phases']['green'])
            self.prev_phase_indexes[node_id] = idx

            y = info['phases']['yellow'][prev_idx]
            self._set_phase(node_id, y)


    def _set_greens(self):
        '''Sets the green phase for each node.'''
        for node_id, info in self.infos.items():
            idx = self.prev_phase_indexes[node_id]
            g = info['phases']['green'][idx]
            self._set_phase(node_id, g)


    def _set_phase(self, node_id, phase):
        '''Sets the phase for a specific node.'''
        if self.prev_phases.get(node_id) != phase:
            self.conn.trafficlight.setRedYellowGreenState(node_id, phase)
            self.prev_phases[node_id] = phase


    def step(self, actions):
        '''Performs a step in the simulation based on the given actions.'''
        
        self._set_yellows(actions)
        self._step(self.yellow_length)
        self._set_greens()
        self._step(self.rest_length)
        self.curr_step += 1

        states, rewards, global_reward = self._get_values()
        done = self.curr_step >= self.episode_step
        return states, rewards, done, global_reward


    def reset(self):
        '''Resets the simulation to its initial state.'''

        # Initialize simulation
        if not os.path.exists(self.path_init_state):
            self._start()
            self._save()
        self._load()

        self.curr_step = 0
        self.prev_phases = {n:'none' for n in self.node_ids}
        self.prev_phase_indexes = {n:0 for n in self.node_ids}

        self._set_greens()
        self._step(1)
        states, _, _ = self._get_values()
        return states
