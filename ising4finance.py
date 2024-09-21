import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from functools import partial 
from time import time

class metropolis_ising:
    def __init__(self, arr_len=20, temp = 1., 
                 field_str=0., block_rad=1, 
                 free_energy_update=5, alpha =4,
                 rand_seed=None):
        """
        Initializes a periodically connected 2D square array of spins whose shape is (arr_len,arr_len).
        
        Parameters
        -----
        
        """
        self.arr_len   = arr_len
        self.num_spins = arr_len*arr_len
        self.temp      = temp
        self.field_str = field_str
        self.epoch_num = 0
        self.rand_seed = rand_seed
        self.alpha= alpha

        # self.J = np.ones((arr_len, arr_len))
        self.C = np.ones((arr_len, arr_len))
        
        self.neighbor_shift_list = [[-1, 0],[0, -1], [0, 1], [1, 0]]
        self.num_neighbors = len(self.neighbor_shift_list)
        
        #To store spin arrays are regular intervals
        self.spin_arr                 = None
        self.spin_arr_memory          = []
        self.spin_arr_epoch_memory    = []
        self.spin_arr_memory_depth    = 25
        
        #Helpful for monitoring state of the system during updates
        self.block_rad                = block_rad
        self.free_energy_update       = free_energy_update
        self.free_energy_epoch_memory = []
        self.flip_energy_entropy      = []
        self.spin_entropy             = []
        self.average_spin             = []
        self.spin_block_entropy       = []
        self.helmholtz_energy         = []
        self.magnetization            = []

        self.fraction_positive_memory = []
        
    def create_random_periodic_spins_on_grid(self,r=0.5):
        """
        Initialize with equal probability for +1 and -1 spins 
        Approximates the high temperature limit.
        
        Parameters
        ------
        seed: int (optional)
            You can specify a random seed for initializing the random spin array
        """
        if r < 0 or r > 1:
            raise ValueError("The ratio r should be between 0 and 1.")
        if type(self.rand_seed) is int:
            np.random.seed(self.rand_seed)
        spin_arr      = np.random.choice([-1, 1], size=self.num_spins, p=[r,1-r])
        self.spin_arr = spin_arr.reshape(self.arr_len, self.arr_len).astype(np.int8).copy()
        np.random.seed()
        
    def create_uniform_periodic_spins_on_grid(self):
        """
        Initialize all spins as +1. 
        This is one of two possible ground states (lowest energy state).
        """
        self.spin_arr = np.ones((self.arr_len, self.arr_len), dtype=np.int8)
    
    def compute_energy_of_spin(self, ind):
        """
        Computes the energy for a spin at index ind:
            E = - S_i * [\sum_j J*S_j - \alpha* C_i(t)*M(t)]
            where s_j are the four spins that are north, south, east, west of s_i at position ind.
            
        Note that we have assumed Boltzmann constant kB=1, and magnetic coupling strength J=1. 
        
        Parameters
        -----
        ind : integer 2-tuple
            [row, column] location of spin whose energy we are trying to compute
        """
        curr_spin      = self.spin_arr[ind[0], ind[1]]
        energy_of_spin = self.field_str
        
        for shifts in self.neighbor_shift_list:
            r = (shifts[0]+ind[0])%self.arr_len 
            c = (shifts[1]+ind[1])%self.arr_len 
            energy_of_spin    += self.spin_arr[r, c]
        energy_of_spin -= self.alpha * self.C[ind[0], ind[1]] * self.calculate_magnetization()
        energy_of_spin *= -curr_spin
        return energy_of_spin
        
    def calculate_magnetization(self):
        """
        Calculate the magnetization of the system.
        
        Magnetization is defined as the sum of all spins divided by the total number of spins.
        """
        total_spins = np.sum(self.spin_arr)
        magnetization = total_spins / self.num_spins
        return magnetization

    def compute_cost_for_spin_flip(self, ind):
        """
        Computes the (energy cost divided by temperature) for spin flip using
        Energy for spin s_i is:
            E = -ext_field*s_i - s_i* (\sum_j s_j)
            where s_j are the four spins that are north, south, east, west of s_i at position ind.
            
        Note that we have assumed Boltzmann constant kB=1, and magnetic coupling strength J=1. 
        
        Parameters
        -----
        ind : integer 2-tuple
            [row, column] location of spin that we are trying to flip
        """
        curr_spin           = self.spin_arr[ind[0], ind[1]]
        energy_of_spin_flip = self.field_str
        
        for shifts in self.neighbor_shift_list:
            r = (shifts[0]+ind[0])%self.arr_len 
            c = (shifts[1]+ind[1])%self.arr_len 
            energy_of_spin_flip    += self.spin_arr[r, c]
        energy_of_spin_flip -= self.alpha * self.C[ind[0], ind[1]] * self.calculate_magnetization()
        energy_of_spin_flip *= (2.*curr_spin)/self.temp    
        
        return energy_of_spin_flip
    
    def attempt_spin_flip(self, ind, cost_thres):
        """
        Attempt to flip a spin at ind = [row,col] using Boltzmann distribution as acceptance function.
        
        Parameters
        -----
        ind : integer 2-tuple
            [row, column] location of spin that we are trying to flip
        """
        energy_of_spin_flip = self.compute_cost_for_spin_flip(ind)
        
        #Spin flip only if energy for doing so is negative (E_flipped < E_noflip)
        #else if spin_flip energy>0 only flip if below cost-threshold
        #Instead of probability, we use log-probabilities to prevent over/underflow
        log_prob_of_flip = - np.log(1+ np.exp(energy_of_spin_flip))
        #cost_thres       = np.log(np.random.rand())
        
        if (energy_of_spin_flip < 0.):
            self.spin_arr[ind[0], ind[1]] *= -1
        elif (log_prob_of_flip > cost_thres):
            self.spin_arr[ind[0], ind[1]] *= -1
    
    def evolve(self, num_epochs=100, store_freq=None, verbose=False):
        """
        Evolve the dynamics of the 2D Ising spins using MCMC algorithm.
        For each epoch, we sequentially attempt to flip each spin. 
        
        You can uncomment one of the lines below flip random spins non-sequentially. 
        Both sequential and non-sequential spin flip recipes should converge to equilibrium states that are within the same thermodynamic ensemble. 
        
        Parameters
        -----
        num_epochs: integer
            Number of epochs where we will evolve the spin configurations using the MCMC algorithm.
        
        store_freq: integer
            How often (measured in epochs) will store a snapshot of the spin configuration
            
        verbose: bool
            Whether to print additional information about the MCMC evolution. 
        """
        t0 = time()
        for e in range(num_epochs):
            #Seed "random temperature fluctuations" if we need to 
            #repeat a trajectory
            if type(self.rand_seed) is int:
                np.random.seed(self.rand_seed+self.epoch_num)
                cost_thresh_array = np.random.rand(self.num_spins)
                np.random.seed()
            else: 
                cost_thresh_array = np.random.rand(self.num_spins)
            cost_thresh_array = np.log(cost_thresh_array)
            

            for pos,ct in zip(range(self.num_spins), cost_thresh_array):
                pos = np.random.randint(0, self.num_spins)
                flip_candidate = np.unravel_index(pos, (self.arr_len, self.arr_len))
                self.attempt_spin_flip(flip_candidate, ct)
                if self.spin_arr[flip_candidate[0], flip_candidate[1]] * self.C[flip_candidate[0], flip_candidate[1]] * self.calculate_magnetization() < 0:
                    self.C[flip_candidate[0], flip_candidate[1]] *= -1 
            self.epoch_num += 1
            fraction_positive = np.mean(self.C == 1)

            #Determine if free energy monitors should be computed+stored during this epoch.
            if (self.epoch_num%self.free_energy_update) == 0:
                self.free_energy_epoch_memory.append(self.epoch_num)
                #self.spin_entropy.append(self.compute_spin_entropy())
                self.average_spin.append(self.compute_average_spin())
                self.spin_block_entropy.append(self.compute_spin_block_entropy())
                self.helmholtz_energy.append(self.compute_helmholtz_energy_per_spin())
                self.magnetization.append(self.calculate_magnetization())
                self.fraction_positive_memory.append(fraction_positive)
                
            #Determine if a snapshot of the spin_arr should be saved in this epoch 
            if store_freq is not None:
                if (e%store_freq) == 0:
                    self.store_spin_arr()
                    self.store_epoch_num()
        if verbose:
            print(f'Time per epoch: {(time()-t0)/num_epochs:0.3f}s')
            
    def compute_flip_energy_map(self):
        """
        Computes the energy cost to flip each spin.
        """
        flip_cost = np.zeros(self.num_spins)
        for pos in range(self.num_spins):
            flip_candidate = np.unravel_index(pos, (self.arr_len, self.arr_len))
            flip_cost[pos] = self.compute_cost_for_spin_flip(flip_candidate)
        return flip_cost.reshape(self.arr_len, self.arr_len)    
    
    def compute_flip_energy_entropy(self):
        """
        Computes the "entropy" associated with flipping each spin.
        """
        flip_energy_map = self.compute_flip_energy_map()
        c               = Counter(flip_energy_map.flatten())
        probs           = list(c.values())
        tot_prob        = np.sum(probs)
        h               = 0.
        for p in probs:
            prob = p/tot_prob
            h    += -(prob) * np.log2(prob)
        return h
                
    def compute_average_spin(self):
        """
        Return the average magnetization of the current spin configuration
        """
        return self.spin_arr.mean()
    
    def compute_spin_entropy(self):
        """
        Computes the average entropy of single spins.
        Deprecated! Should use extract_spin_blocks instead.
        """
        down_spin_frac = (self.spin_arr == -1).sum() / self.num_spins
        up_spin_frac   = (self.spin_arr == 1).sum() / self.num_spins
        return np.sum([-p*np.log2(p) for p in [down_spin_frac, up_spin_frac] if p > 0.5/self.num_spins])
        
    def extract_spin_blocks(self):
        """
        Extracts all square sub-blocks of (self.block_rad+1)**2 spins from the spin_arr.
        The spin-pattern of each spin-block is stored as strings e.g., "-111-1-1-1".
        
        Note: converting to binary 0bXXX words before doing a counter is slower.
        """
        block_range  = range(-self.block_rad, self.block_rad+1)
        block_shifts = [[x,y] for x in block_range for y in block_range]
        
        #Representing spin blocks as strings
        all_words = []
        for pos in range(self.num_spins):
            ind  = np.unravel_index(pos, (self.arr_len, self.arr_len))
            word = []
            for n,shifts in enumerate(block_shifts):
                r = (shifts[0]+ind[0])%self.arr_len 
                c = (shifts[1]+ind[1])%self.arr_len
                word.append(str(self.spin_arr[r, c]))
            all_words.append(''.join(word))
        return all_words
    
    def compute_spin_block_entropy(self):
        """
        Computes the entropy of the spin blocks harvested from spin_arr.
        This entropy estimates the number of possible spin-block patterns that might occur 
        based on ensemble of spin-blocks from spin_arr. 
        
        The output divides this entropy by the number of spins within each spin-block
        to give us average entropy per spin. 
        
        Note that when self.block_rad = 0, this routine essentially returns only the single spin entropy.
        """
        spin_blocks_as_words = self.extract_spin_blocks()
        c        = Counter(spin_blocks_as_words)
        probs    = list(c.values())
        tot_prob = np.sum(probs)
        h        = 0.
        
        for p in probs:
            prob = p/tot_prob
            h    += -(prob) * np.log2(prob)
        return h/((2*self.block_rad+1)**2)
        
    def compute_mean_energy(self):
        """
        Computes the total spin energy of the spin_arr, then divides by the number of spins in spin_arr.
        Approximates the internal energy U of the spin_arr.
        """
        tot_energy = 0.
        for pos in range(self.num_spins):
            flip_candidate = np.unravel_index(pos, (self.arr_len, self.arr_len))
            tot_energy     += self.compute_energy_of_spin(flip_candidate)
        return tot_energy/self.num_spins
    
    def compute_helmholtz_energy_per_spin(self):
        """
        Computes the Helmholtz energy of the spin_arr:
            F = U - T S, 
            where U is the internal energy
            T is the average temperature of the spin_arr
            S is the entropy of the spin_arr computed using spin-blocks as features in counting entropy
        """
        return self.compute_mean_energy() - self.compute_spin_block_entropy()*self.temp

    def store_spin_arr(self):
        """
        Append the spin array to a running list of spin-arrays that have a maximum depth.
        We implement a first-in-first-out (FIFO) queue. 
        """
        if len(self.spin_arr_memory) < self.spin_arr_memory_depth:
            self.spin_arr_memory.append(self.spin_arr.copy())
        else:
            self.spin_arr_memory.pop(0)
            self.spin_arr_memory.append(self.spin_arr.copy())
            
    def store_epoch_num(self):
        """
        We implement a first-in-first-out (FIFO) queue.
        """
        if len(self.spin_arr_epoch_memory) < self.spin_arr_memory_depth:
            self.spin_arr_epoch_memory.append(self.epoch_num)
        else:
            self.spin_arr_epoch_memory.pop(0)
            self.spin_arr_epoch_memory.append(self.epoch_num)
            
    def plot_spins(self, num_rows=5, num_cols=5, figsize=(10,10)):
        """
        Plot the stored spin configurations.
        """
        fig, axes   = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
        shared_args = {"xytext":(-1,-1), "va":'top', "ha":"left", "textcoords":'offset points'}
        for ax,arr,lb in zip(axes.ravel(), self.spin_arr_memory, self.spin_arr_epoch_memory):
            ax.imshow(arr, vmin=-1, vmax=1, cmap=plt.cm.bone_r)
            ax.annotate(f'Epoch {lb:d}', xy=(1,0), color='black',
                        bbox=dict(boxstyle="round", fc="white", ec="black"), **shared_args)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
    def plot_summary(self):
        """
        """
        total_energy = [helmholtz_energy + self.temp * spin_block_entropy for helmholtz_energy, spin_block_entropy in zip(self.helmholtz_energy, self.spin_block_entropy)]

        fig, axes = plt.subplots(1,3, figsize=(18,5))
        axes[0].plot(self.free_energy_epoch_memory, self.spin_block_entropy, label='spin block entropy')
        axes[0].legend()
        axes[0].set_xlabel('epoch number')
        axes[0].set_ylabel('metric')

        axes[1].plot(self.free_energy_epoch_memory, self.helmholtz_energy, label='helmholtz energy per spin')
        axes[1].legend()
        axes[1].set_xlabel('epoch number')

        axes[2].plot(self.free_energy_epoch_memory, total_energy, label='Total Energy (U)')
        axes[2].set_xlabel('Epoch Number')
        axes[2].set_ylabel('Energy')
        axes[2].legend()

        
    def ising_mean_field_transcendental_objective(self, temp, mean_mag):
        """
        Returns the objective function (transcedental equation) to minimize, 
        which gives us the mean field magnetization in a 2D Ising system 
        in the absence of an external field. 
        
        Parameters
        ------
        temp: floating point
            temperature of the spin system
            
        mean_mag: floating point
            mean magnetization of the spin system

        """
        return np.fabs(mean_mag - np.tanh(self.num_neighbors*mean_mag/temp))

    def compute_mean_mag(self, temp, x0=0.05):
        """
        Solve the transcendental equation that gives us the 
        mean field magnetization for the Ising system at a particular temperature
        ------
        Parameters:
        x0 : floating point
            Initial guess for magnetization when iteratively solving the transcendental equation
        """
        ising_mean_t = partial(self.ising_mean_field_transcendental_objective, temp)
        res          = minimize(ising_mean_t, x0) 
        return np.abs(res.x)
    
    def compute_mean_energy_per_epoch(self):
        """
        Computes the mean energy per spin at each epoch.
        """
        mean_energy_per_epoch = []
        for spin_arr_snapshot in self.spin_arr_memory:
            tot_energy = 0.
            for pos in range(self.num_spins):
                flip_candidate = np.unravel_index(pos, (self.arr_len, self.arr_len))
                tot_energy += self.compute_energy_of_spin(flip_candidate)
            mean_energy_per_epoch.append(tot_energy / self.num_spins)
        return mean_energy_per_epoch

    def plot_mean_energy_per_epoch(self):
        """
        Plots the mean energy per spin as a function of epochs.
        """
        mean_energy_per_epoch = self.compute_mean_energy_per_epoch()
        plt.plot(self.free_energy_epoch_memory, mean_energy_per_epoch)
        plt.xlabel('Epoch Number')
        plt.ylabel('Mean Energy per Spin')
        plt.title('Mean Energy per Spin vs. Epochs')
        plt.show()



    def internal_energy(self) :
        return [helmholtz_energy + self.temp * spin_block_entropy for helmholtz_energy, spin_block_entropy in zip(self.helmholtz_energy, self.spin_block_entropy)]
    
    def plot_magnetization_per_epoch(self):
        """
        Plots the total magnetization as a function of epochs.
        """
        plt.plot(self.free_energy_epoch_memory, self.magnetization)
        plt.xlabel('Epoch Number')
        plt.ylabel('Magnetization')
        plt.title('Magnetization vs. Epochs')
        plt.show()

    def get_return_per_epoch(self):
        """
        Outputs the return per epoch as a lsit: ret(t) = ln(M(t)) - ln(M(t-1)).
        """
        # Compute the difference in magnetization between consecutive epochs
        returns = np.diff(self.magnetization) /2 
        
        return returns

    def plot_return_per_epoch(self):
        """
        Plots the return per epoch: ret(t) = ln(M(t)) - ln(M(t-1)).
        """
        returns = np.diff(self.magnetization) /2 
        mean_return = np.mean(returns)
        
        # Plot the return per epoch
        plt.plot(self.free_energy_epoch_memory[1:], returns)
        plt.axhline(y=mean_return, color='r', linestyle='--', label='Mean Return')
        plt.xlabel('Epoch Number')
        plt.ylabel('Return')
        plt.title('Return per Epoch')
        plt.show()

    def plot_fraction_positive(self):
        """
        Plots the fraction of agents with a strategy C_i = +1 over epochs.
        """
        mean_fraction = np.mean(self.fraction_positive_memory[1:])
        plt.plot(self.free_energy_epoch_memory, self.fraction_positive_memory)
        plt.axhline(y=mean_fraction, color='r', linestyle='--', label='Mean')
        plt.xlabel('Epoch Number')
        plt.ylabel('Fraction of Agents with C_i = +1')
        plt.title('Fraction of Agents with C_i = +1 over Epochs')
        plt.show()

    def calculate_volatility_per_epoch(self):
        """
        Calculates the volatility per epoch based on the time series of magnetization.

        Parameters:
        - magnetization: list or array, time series of magnetization values

        Returns:
        - volatility_per_epoch: list, volatility values for each epoch
        """
        # Initialize an empty list to store volatility values
        volatility_per_epoch = []

        # Compute volatility for each epoch
        for t in range(1, len(self.magnetization)):
            # Compute the squared difference between magnetization at time t and t-1
            squared_diff = (self.magnetization[t] - self.magnetization[t-1])**2

            # Append squared difference to the list
            volatility_per_epoch.append(np.sqrt(squared_diff))

        return volatility_per_epoch

    def plot_volatility_per_epoch(self):
        """
        Plots the volatility per epoch based on the time series of magnetization.

        Parameters:
        - magnetization: list or array, time series of magnetization values
        """
        # Calculate volatility per epoch
        volatility_per_epoch = self.calculate_volatility_per_epoch()

        # Create a range of epochs for the x-axis
        epochs = range(1, len(volatility_per_epoch) + 1)
        
        mean_volatility = np.mean(volatility_per_epoch)
        # Plot the volatility per epoch
        plt.plot(epochs, volatility_per_epoch, linestyle='-')
        plt.axhline(y=mean_volatility, color='r', linestyle='--', label='Mean')
        # Add labels and title
        plt.xlabel('Epoch Number')
        plt.ylabel('Volatility')
        plt.title('Volatility per Epoch')
        
        # Show grid
        plt.grid(True)
        
        # Show the plot
        plt.show()

    def calculate_volatility(self):
        returns = self.get_return_per_epoch()
        volatilities = []
        for t in range(1, len(returns)):
            # Calculate volatility for each time step t using all past data points
            window_returns = returns[:t]
            volatility = np.std(window_returns, ddof=1)  # Use unbiased estimator (N-1) for standard deviation
            volatilities.append(volatility)
        return volatilities

    def plot_volatility(self):
        returns = self.get_return_per_epoch()
        volatilities = self.calculate_volatility()
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(returns)), volatilities, label='Volatility')
        plt.title('Time-varying Volatility')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.legend()
        plt.show()


    def plot_autocorrelation(self):
        """
        Plots the autocorrelation of the returns.
        """
        # Compute autocorrelation of returns
        returns = np.diff(self.magnetization) /2 
        autocorrelation = np.correlate(returns, returns, mode='full')
        autocorrelation /= np.max(autocorrelation)  # Normalize autocorrelation

        # Plot autocorrelation
        plt.plot(autocorrelation)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation of Returns')
        plt.show()