class Args:

    '''Args class defines key parameters for training and simulation.'''

    # num workers: Number of workers for parallel processing
    num_workers    = 4

    # path
    path_network   = 'data/sumo.net.xml'
    path_detectors = 'data/detectors.add.xml'
    path_configs   = ['data/sumo.sumocfg'] * num_workers
    path_weights   = 'data/model'

    # intersection ids
    node_ids = ('A0', 'B0', 'C0')
    
    # Simulation settings
    gui            = False
    e2_length      = 150
    yellow_length  = 5
    step_length    = 10
    episode_length = 7360
    episode_step   = episode_length // step_length
    num_episode    = 1000

    # Reinforcement learning hyperparameters
    learning_rate  = 0.0005
    gamma          = 0.95
    gae_lambda     = 0.9
    ratio_clipping = 0.05
    entropy_coef   = 0.01
    k_epoch        = 3
    batch_size     = 32
    action_dim     = 2


assert Args.episode_length % Args.step_length == 0
assert Args.episode_step % Args.batch_size == 0
