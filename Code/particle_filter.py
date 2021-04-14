import numpy as np
from scipy.stats import multivariate_normal
# from Building import Building
# from Path import Path
# from lstm import predict
from tqdm import tqdm
import glob
import pdb
from matplotlib import pyplot as plt
import pickle

class ParticleFilter:
    # TODO - keep track of parents
    def __init__(self, site_path, n_particles=1000, sample_particles=20, initial_multiply=10, wifi_weight = 1, min_p = .2):
        self.n_particles = n_particles
        self.sample_particles = sample_particles
        self.initial_multiply = initial_multiply
        # self.building = Building(site_path)

        self.movement_keys = ['acce', 'acce_uncali', 'gyro', 'gyro_uncali', 'magn', 'magn_uncali', 'ahrs']
        self.wifi_weight = wifi_weight
        self.min_p = min_p

    def estimate(self, target_path, waypoint_times=[], plot=False):
        # params to adjust
        # OVERALL PARAMS
            # n_particles -> higher the better
            # sample_particles -> higher the better
        # WIFI PARAMS
            # in building, how much to filter? How many do we take?
            # min_p - need to be careful about unseen data
            # dist vs k, and what values
            # dist_weighting?
        # MOVEMENT PARAMS
            # max time to move before update?
            # any multiplier on sigma?
            # adjust p at all?
        # FINAL
            # take the MLE? or the mode somehow? kernel density?

        self.initialize_particles()
        self.path = Path(target_path)
        if len(waypoint_times) == 0:
            waypoint_times = self.path['waypoint'][:,0]
        self.waypoint_times = waypoint_times

        diff = np.diff(waypoint_times)

        # LSTM trained to do up to every 2 seconds, make sure we run LSTM at least every 2 seconds
        max_ms = 2000
        num_between = diff // max_ms
        start_time = waypoint_times[0]
        eval_times = [waypoint_times[0]]
        # keep track of which ones to actually evaluate
        waypoint_inds = [1.0]
        for t0, t1, s in zip(waypoint_times[:-1], waypoint_times[1:], num_between):
            s = int(s)
            times = np.linspace(t0, t1, s+2)[1:]
            inds = np.zeros(s+1)
            inds[-1] = 1
            waypoint_inds.extend(inds)
            eval_times.extend(times)
        eval_times = np.array(eval_times)
        # make sure to look at the first one
        # waypoint_inds[0] = 1
        waypoint_inds = np.array(waypoint_inds)

        # get movement vars
        movement_data = self.path.get_many_between_times(self.movement_keys, eval_times)
        wifi_data = self.path.get_many_between_times(['wifi'], eval_times)[0]

        # TODO - pass movement data through LSTM, get estimates
        # instances = len(movement_data)
        sequences = []
        for j in range(len(eval_times)-1):
            all_data = []
            for i in range(len(self.movement_keys)):
                try:
                    all_data.append(movement_data[i][j])
                except:
                    print('Something went wrong...')
                    pdb.set_trace()
                    pass
            all_data = np.array(all_data)
            all_data = np.swapaxes(all_data, 0, 1)
            # take out time
            all_data = all_data[:,:,1:]
            # flatten
            all_data = all_data.reshape(all_data.shape[0], -1)
            sequences.append(all_data)
        # mus, sigmas = predict(sequences)
        mus, sigmas = predict(sequences)
        # remove all movement info
        # mus = 0 * mus
        # sigmas = 0 * sigmas
        # sigmas[:,1,1] = 30
        # sigmas[:,0,0] = 30
        # TODO - experiment with sigma size?
        # sigmas *= 1.2

        if not np.allclose(eval_times[waypoint_inds == 1], waypoint_times):
            print('Waypoints times off...')
            pdb.set_trace()
            pass
        found_wifi = False
        # TODO - for each diff
        for i in tqdm(range(len(eval_times)-1)):
            if plot:
                true_X, true_Y = self.path.waypoint_interp(eval_times)[i]

                Z, X, Y = self.particles[:,0], self.particles[:,1], self.particles[:,2]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X, Y, Z, alpha=.1)
                z_unique = np.unique(Z)
                n_z = len(z_unique)
                ax.scatter(n_z*[true_X], n_z*[true_Y], z_unique, alpha=1)
                plt.show()
            t0, t1 = eval_times[i], eval_times[i+1]
            # move particles, update p
            change = self.move(mus[i], sigmas[i])
            # TODO - pickup here
            wifi_times, wifi_vecs = self.building.create_wifi_vectors(wifi_data[i])
            for t, v in zip(wifi_times, wifi_vecs):
                found_wifi = True
                lam = (t1 - t) / (t1 - t0)
                wifi_position = self.particles[:,:-1]
                wifi_position[:,1:] = wifi_position[:,1:] - change * lam
                # look backward, multiply by likelihood
                log_p_wifi = self.building.log_p_wifi(v, wifi_position, dist=10, dist_weight=True, min_p = self.min_p)

                # multiply by wifi_weight
                log_p_wifi = self.wifi_weight * log_p_wifi
                self.update_log_p(log_p_wifi)
            # sample particles down
            # if no wifi found, keep size
            if not found_wifi:
                self.sample_down(self.n_particles * self.initial_multiply)
            else:
                self.sample_down()

        # TODO - pick most likely final point at end
        i = self.particles[:,-1].argmax()
        path = [self.particles[i]]
        for parents, particles in zip(reversed(self.parent_history), reversed(self.history[:-1])):
            i = parents[i]
            path.append(particles[i])
        path = np.array(path)[::-1]
        waypoint_hat = path[np.argwhere(waypoint_inds).reshape(-1)][:,:-1]

        if len(self.path['waypoint']) > 0:
            diff = waypoint_hat[:,1:] - self.path['waypoint'][:,1:]
            print(diff)
            loss = np.sqrt((diff**2).sum(axis=1)).mean()
            print(loss)

        # TODO - trace back
        # TODO - return waypoints
        return waypoint_hat

    def initialize_particles(self):
        ''' Draw particles randomly, assign equal probabilities'''
        multiplier = self.initial_multiply
        waypoints = sample_waypoints(self.n_particles * multiplier)
        # waypoints = self.building.sample_waypoints(self.n_particles * multiplier)
        # initialize log_p = 1
        log_p = np.zeros((self.n_particles * multiplier, 1))
        particles = np.hstack([waypoints, log_p])
        # set particles
        self.particles = particles
        # keep track of particles parents at previous time step
        self.parents = []
        # each time step is a step of particles
        self.history = [self.particles]
        self.parent_history = []

    def update_log_p(self, log_p):
        ''' Updates particle probabilities'''
        self.particles[:,-1] += log_p
        # mean shift
        self.particles[:,-1] += -self.particles[:,-1].mean()
        return

    def sample_down(self, n=None):
        if n is None:
            n = self.n_particles
        p = np.exp(self.particles[:, -1])
        # TODO - handle?
        # p[p == np.nan] = p.min()
        p *= 1 / p.sum()
        indices = np.random.choice(len(self.particles), n, p = p)
        self.particles = self.particles[indices]
        self.parent_history.append(self.parents[indices])
        self.parents = []
        self.history.append(self.particles)
        return



    def move(self, mu, sigma):
        ''' For each particle, move randomly according to gaussian noise
        Update probabilites according to log_likelihood '''
        # calculate sample_particles for each n_particle
        n_particles = len(self.particles)
        size = (n_particles, self.sample_particles)
        # get samples
        change = np.random.multivariate_normal(mu, sigma, size=size)
        # calculate change
        log_p = np.log(multivariate_normal.pdf(change, mean=mu, cov=sigma))

        particles = self.particles
        # TODO - reshape particles, repeat across dim 2 by sample_particles?
        particles = np.expand_dims(particles, 1)
        parents = np.expand_dims(np.arange(len(particles)), 1)
        # repeat
        particles = np.repeat(particles, self.sample_particles, 1)
        parents = np.repeat(parents, self.sample_particles, 1)

        particles = particles.reshape(-1, 4)
        parents = parents.reshape(-1)
        change = change.reshape(n_particles * self.sample_particles, 2)
        log_p = log_p.reshape(n_particles * self.sample_particles)

        particles[:,1:3] = particles[:,1:3] + change
        # bayes update
        particles[:,-1] = particles[:,-1] + log_p

        self.particles = particles
        self.parents = parents
        # TODO - filter to
        self.filter_to_reasonable()
        return change

    def filter_to_reasonable(self):
        ''' Filters particles to only near places where people actually walked, limit to floor plan '''
        return

    # def update_wifi(self, wifi_vector):
    #     position = self.particles[:,1:3]
    #     # get bernoulli likelihood
    #     p = get_wifi_likelihood(wifi_vector, position)
    #     self.particles[:,-1] = self.particles[:,-1] * p

    # def resample(self):
    #     particles = np.random.choice(self.particles, self.particles[:,-1])

if __name__ == '__main__':
    # building_path = 'small_data/train/5dc8cea7659e181adb076a3f'
    # target_path = 'small_data/train/5dc8cea7659e181adb076a3f/F6/5dcd3e45a4dbe7000630b00d.txt'
    # DRIFT WITH WIFI
    # building_path = 'small_data/train/5da138754db8ce0c98bca82f'
    # target_path = 'small_data/train/5da138754db8ce0c98bca82f/F4/5dd21b8b878f3300066c8013.txt'

    # DRIFT
    # building_path = 'small_data/train/5da958dd46f8266d0737457b'
    # target_path = 'small_data/train/5da958dd46f8266d0737457b/F1/5db126c8e62491000652bb22.txt'


    # RANDOM
    # building_path = np.random.choice(glob.glob('small_data/train/*'))
    # target_path = np.random.choice(glob.glob(building_path + '/*/*.txt'))

    with open('/Users/chelsey/Documents/ACME Labs/Project_Vol_3/hurricane-prediction/Data/atlantic_series.pickle','rb') as f:
        atlantic = pickle.load(f)
    new_atlantic = []
    for i, data in enumerate(atlantic):
        new_atlantic.append(data[1:])
    with open('/Users/chelsey/Documents/ACME Labs/Project_Vol_3/hurricane-prediction/Data/pacific_series.pickle','rb') as f:
        pacific = pickle.load(f)

    # building_path = glob.glob('small_data/train/*')[1]
    # building_path = 'data/train/5cd56b79e2acfd2d33b5b74e'

    n_particles = 100
    sample_particles = 20
    # initial_multiply = 10
    initial_multiply = 10
    wifi_weight = 1
    min_p = .45
    pf = ParticleFilter(
            new_atlantic[:-10],
            n_particles=n_particles,
            sample_particles=sample_particles,
            initial_multiply=initial_multiply,
            wifi_weight = wifi_weight,
            min_p = min_p,
    )
    # pf = ParticleFilter(building_path)
    # target_path = glob.glob(building_path + '/*/*.txt')[200]
    P = [.001, .01, .1, .2, .3, .4, .45, .49]
    for min_p in P:
        pf.min_p = min_p
        print(min_p)
        pf.estimate(new_atlantic[-10:], plot=False)
    print(building_path)
    print(target_path)
