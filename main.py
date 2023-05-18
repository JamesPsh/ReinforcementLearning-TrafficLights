import os
import ray
import numpy as np
import pandas as pd
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam

from args import Args
from config import Config
from tools import RewardScaler, standardize
from env import Env

# Set the seed for numpy and tensorflow to ensure reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# args
learning_rate  = Args.learning_rate
gamma          = Args.gamma
gae_lambda     = Args.gae_lambda
ratio_clipping = Args.ratio_clipping
entropy_coef   = Args.entropy_coef

k_epoch        = Args.k_epoch
batch_size     = Args.batch_size
num_episode    = Args.num_episode
action_dim     = Args.action_dim
path_configs   = Args.path_configs
path_weights   = Args.path_weights


def init_models(infos):
    '''Initialize PPO models and optimizers for each traffic node.'''
    models, optimizers = [], []
    for node_id, info in infos.items():
        d0 = len(info['detectors'])
        discrete_dim = 1
        max_num_phase = len(info['phases']['green'])

        for neighbor_id in info['neighbors']:
            d0 += len(infos[neighbor_id]['detectors'])
            discrete_dim += 1
            max_num_phase = max(max_num_phase, len(infos[neighbor_id]['phases']['green']))

        d1 = d0 * 2
        model = PPOModel(action_dim, d0, d1, discrete_dim, max_num_phase, node_id)
        state_dim = d1 + discrete_dim
        model.build(input_shape=(None, state_dim))
        opt = Adam(learning_rate=learning_rate)

        # Save the model and optimizer
        models.append(model)
        optimizers.append(opt)
    return models, optimizers


class PPOModel(tf.keras.Model):
    def __init__(self, action_dim, d0, d1, discrete_dim, emb_input_dim, node_id):
        super(PPOModel, self).__init__()
        self.d0 = d0
        self.d1 = d1
        self.discrete_dim = discrete_dim
        self.rnn_units = 64

        self.fc00 = Dense(64, 'relu')
        self.fc01 = Dense(64, 'relu')
        self.fc02 = Dense(64, 'relu')
        self.embs = [Embedding(emb_input_dim, 32) for _ in range(self.discrete_dim)]

        self.rnn = LSTM(self.rnn_units, return_sequences=True, return_state=True)

        self.fc10 = Dense(64, 'relu')
        self.fc11 = Dense(64, 'relu')
        self.policy = Dense(action_dim, 'softmax')
        self.v_value = Dense(1)

        self.node_id = node_id


    def call(self, state, hidden=None, return_type='both'):
        x0 = state[:, :self.d0]  # occupancy
        x1 = state[:, self.d0:self.d1]  # queue
        x2 = state[:, -self.discrete_dim:]  # phase

        if hidden is None:
            hidden = [tf.zeros((1, self.rnn_units)), tf.zeros((1, self.rnn_units))]

        x0 = self.fc00(x0)
        x1 = self.fc01(x1)
        x2 = [emb(x2[:, i]) for i, emb in enumerate(self.embs)]
        x2 = tf.concat(x2, axis=-1)
        x2 = self.fc02(x2)

        x = tf.concat([x0, x1, x2], axis=1)
        x = tf.expand_dims(x, axis=0)
        x, h, c = self.rnn(x, initial_state=hidden)
        x = tf.squeeze(x, axis=0)

        x = self.fc10(x)
        x = self.fc11(x)
        
        if return_type == 'both':
            return self.policy(x), self.v_value(x), [h, c]

        elif return_type == 'policy':
            return self.policy(x), [h, c]

        elif return_type == 'v_value':
            return self.v_value(x)

        raise ValueError(f"Invalid return_type. Expected 'both', 'policy', or 'v_value', but got: {return_type}")


class Worker:
    def __init__(self, infos, path_config, name):
        '''Initialize a worker with a given configuration.'''
        self.env = Env(infos, path_config, name)
        self.models, _ = init_models(infos)
        self.scaler = RewardScaler(len(self.env.node_ids), gamma)


    def get_init_hidden(self, model):
        units = model.rnn_units
        hidden = [tf.zeros((1, units), tf.float32), tf.zeros((1, units), tf.float32)]
        return hidden


    def get_actions(self, states, hiddens, deterministic=False):
        '''Calculate actions and policies for given states.'''
        actions, policies, new_hiddens = [], [], []
        for i, model in enumerate(self.models):

            s = states[i].reshape((1, -1))
            h = hiddens[i]

            p, new_h = model(s, h, 'policy')
            p = p.numpy()[0]
            if deterministic:
                a = np.argmax(p)
            else:
                a = np.random.choice(np.arange(len(p)), p=p)
            actions.append(a)
            policies.append(p[a])
            new_hiddens.append(new_h)
        return actions, policies, new_hiddens


    def get_trajectory_segment(self):
        '''Get a segment of the trajectory.'''
        segment = defaultdict(list)
        s = self.states
        h = self.prev_hiddens_0 = self.hiddens
        for i in range(batch_size):
            a, p, next_h = self.get_actions(s, h)
            next_s, r, done, global_reward = self.env.step(a)

            r = np.array(r, dtype=np.float32)
            r = self.scaler.step(r)

            segment['s'].append(s)
            segment['a'].append(a)
            segment['r'].append(r)
            segment['next_s'].append(next_s)
            segment['old_pi_a'].append(p)
            done_mask = [0 if done else 1] * len(a)
            segment['done_masks'].append(done_mask)

            if i == 0:
                self.prev_hiddens_1 = next_h

            s = next_s
            h = next_h
            self.global_reward += global_reward

            if done:
                if len(segment['s']) != batch_size:
                    raise ValueError("The number of states is not equal to the batch size.")
                break

        self.make_batch(segment)
        self.segment = segment
        self.states = s
        self.hiddens = h
        return self.global_reward, done


    def make_batch(self, segment):
        '''Converts segment values into batch format.'''
        for k, value in segment.items():
            d_type = np.int32 if k == 'a' else np.float32
            segment[k] = [np.vstack(v).astype(d_type) for v in zip(*value)]


    def get_gradients(self):
        '''Calculate the gradients for the models.'''
        segment = self.segment
        grads = []
        for i, model in enumerate(self.models):
            s, a, r, next_s = segment['s'][i], segment['a'][i], segment['r'][i], segment['next_s'][i]
            old_pi_a = segment['old_pi_a'][i]
            done_mask = segment['done_masks'][i]
            h0, h1 = self.prev_hiddens_0[i], self.prev_hiddens_1[i]
            with tf.GradientTape() as tape:

                # Forward passes
                v = model(s, h0, 'v_value')
                next_v = model(next_s, h1, 'v_value')

                # Compute TD target and delta
                td_target = r + gamma * next_v.numpy() * done_mask
                delta = td_target - v.numpy()

                # Compute GAE advantage
                advs = []
                adv = 0.0
                for delta_t in delta[::-1]:
                    adv = gamma * gae_lambda * adv + delta_t[0]
                    advs.append([adv])
                advs.reverse()
                advs = np.array(advs, dtype=np.float32)

                # standardize only for advs, not v_targets
                # https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/actor_critic.py#LL261C45-L261C87
                advs = standardize(advs)

                pi, _ = model(s, h0, 'policy')
                pi_a = tf.gather(pi, a, axis=1, batch_dims=1)
                log_pi_a = tf.math.log(tf.clip_by_value(pi_a, 1e-10, 1.0))
                log_old_pi_a = tf.math.log(tf.clip_by_value(old_pi_a, 1e-10, 1.0))
                ratio = tf.exp(log_pi_a - log_old_pi_a)

                # Actor loss
                surrogate_loss = tf.minimum(
                    ratio * advs,
                    tf.clip_by_value(ratio, 1 - ratio_clipping, 1 + ratio_clipping) * advs
                )
                actor_loss = -tf.reduce_mean(surrogate_loss)

                # Value loss
                value_loss = tf.reduce_mean((v - td_target) ** 2)

                # Entropy loss
                log_pi = tf.math.log(tf.clip_by_value(pi, 1e-10, 1.0))
                entropy = -tf.reduce_sum(pi * log_pi, axis=-1)
                entropy_loss = -tf.reduce_mean(entropy)
                
                # Total loss
                loss = actor_loss + value_loss + entropy_loss * entropy_coef

            grad = tape.gradient(loss, model.trainable_variables)
            grads.append(grad)
        return grads


    def set_weights(self, model_weights):
        '''Set model weights.'''
        try:
            for i, model in enumerate(self.models):
                model.set_weights(model_weights[i])
        except ValueError as ve:
            print(f"ValueError occurred while setting weights: {ve}")
            raise ve
        except Exception as e:
            print(f"Unexpected error occurred while setting weights: {e}")
            raise e


    def load_weights(self):
        """Load model weights."""
        for i, model in enumerate(self.models):
            node_id = model.node_id
            try:
                model.load_weights(f'{path_weights}_{name}_{node_id}.h5')
            except FileNotFoundError:
                print(f"No weights file found for node {node_id}, model will be initialized with random weights.")


    def close(self):
        '''Closes the environment.'''
        self.env.close()


    def reset(self):
        '''Resets the environment and other parameters.'''
        self.segment = None
        self.hiddens = [self.get_init_hidden(m) for m in self.models]
        self.prev_hiddens_0 = None
        self.prev_hiddens_1 = None
        self.global_reward = 0
        self.states = self.env.reset()


class Learner:
    def __init__(self, infos):
        '''Initialize a Learner with given information.'''
        self.models, self.optimizers = init_models(infos)


    def apply_gradients(self, gradients_list):
        '''Apply averaged gradients to models.'''
        gradients_list = [g for g in zip(*gradients_list)]
        for model, opt, gradients in zip(self.models, self.optimizers, gradients_list):
            averaged_gradients = []
            for grads in zip(*gradients):
                grads = tf.stack(grads)
                averaged_gradients.append(tf.reduce_mean(grads, axis=0))
            opt.apply_gradients(zip(averaged_gradients, model.trainable_variables))


    def get_weights(self):
        '''Get model weights.'''
        return [model.get_weights() for model in self.models]


    def save_weights(self):
        '''Save model weights.'''
        for i, model in enumerate(self.models):
            node_id = model.node_id
            try:
                model.save_weights(f'{path_weights}_{name}_{node_id}.h5')
            except Exception as e:
                print(f"Could not save weights for node {node_id}. Error: {e}")


# ray worker
Worker_remote = ray.remote(Worker)
def initialize_workers(infos, path_configs, name):
    '''Initialize workers.'''
    num_workers = min(len(path_configs), os.cpu_count())
    selected_path_configs = path_configs[:num_workers]
    ray.init(num_cpus=num_workers)
    print(f'num_workers: {num_workers}')

    workers = []
    for i, path_config in enumerate(selected_path_configs):
        try:
            worker = Worker_remote.remote(infos, path_config, f'{name}_{i}')
            workers.append(worker)
        except Exception as e:
            print(f"Error initializing worker {i}: {e}")
    return workers


def set_worker_weights(workers, model_weights):
    '''Set weights for workers.'''
    futures = [worker.set_weights.remote(model_weights) for worker in workers]
    ray.wait(futures, num_returns=len(workers))


def train():

    infos = Config().infos
    learner = Learner(infos)
    workers = initialize_workers(infos, path_configs, name)

    model_weights = ray.put(learner.get_weights())
    set_worker_weights(workers, model_weights)

    df_rewards = []
    is_first_run = True
    for n_epi in range(num_episode):
        futures = [worker.reset.remote() for worker in workers]
        ray.wait(futures, num_returns=len(workers))

        while True:

            # get_trajectory_segment
            futures = [worker.get_trajectory_segment.remote() for worker in workers]
            ray.wait(futures, num_returns=len(workers))
            rewards, dones = [v for v in zip(*ray.get(futures))]

            if not is_first_run:
                for _ in range(k_epoch):

                    # get gradients
                    futures = [worker.get_gradients.remote() for worker in workers]
                    ray.wait(futures, num_returns=len(workers))
                    gradients_list = ray.get(futures)
                    
                    # update
                    learner.apply_gradients(gradients_list)
                    
                    # set weights 
                    model_weights = ray.put(learner.get_weights())
                    set_worker_weights(workers, model_weights)
            else:
                is_first_run = False

            if dones[0]:
                break
            
        # Save the weights
        learner.save_weights()

        df_rewards.append({n:r for n, r in zip(Args.node_ids, rewards)})
        print(f'n_epi: {n_epi}, rewards: {rewards}')
        try:
            pd.DataFrame(df_rewards).to_csv(path_reward)
        except Exception as e:
            print(f'Error saving rewards: {e}')

    futures = [worker.close.remote() for worker in workers]
    ray.wait(futures, num_returns=len(workers))
    ray.shutdown()
    print('train done!!!')


def test():

    infos = Config().infos
    agent = Worker(infos, path_configs[0], name)
    agent.load_weights()
    agent.env.gui = True
    deterministic = False

    agent.reset()
    s = agent.states
    h = agent.hiddens
    while True:
        a, p, h = agent.get_actions(s, h, deterministic)
        s, r, done, global_reward = agent.env.step(a)
        print(f'p: {np.round(p, 2)}, global_reward: {np.round(global_reward, 2)}')
        if done:
            break

    agent.close()
    print('test done!!!')


import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', dest='name', type=str)
    parser.add_argument('--mode', '-m', dest='mode', type=str)
    args = parser.parse_args()
    name = args.name
    path_reward = f'data/rewards_{name}.csv'

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
