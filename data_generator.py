import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import ising4finance as i4f
from multiprocessing import Pool
import h5py


def generate_data_for_regime(num_values_per_regime, Tc=3.7):
    regimes = ['below', 'a little below', 'around', 'above', 'way above']
    temperatures_per_regime = {}
    
    def generate_random_temperatures(num_values, regime):
        if regime == 'below':
            min_temp, max_temp = 0.1, 0.5 * Tc
        elif regime == 'a little below':
            min_temp, max_temp = 0.5 * Tc, 0.9 * Tc
        elif regime == 'around':
            min_temp, max_temp = 0.9 * Tc, 1.1 * Tc
        elif regime == 'above':
            min_temp, max_temp = 1.1 * Tc, 2.0 * Tc
        elif regime == 'way above':
            min_temp, max_temp = 2.0 * Tc, 5.0 * Tc
        else:
            raise ValueError("Invalid regime specified")
        
        return np.random.uniform(min_temp, max_temp, num_values)
    
    for regime in regimes:
        temperatures_per_regime[regime] = generate_random_temperatures(num_values_per_regime, regime)
    
    return temperatures_per_regime

def run_simulation(temp):
    ising_data = i4f.metropolis_ising(arr_len=40, temp=temp, block_rad=1, free_energy_update=2)
    ising_data.create_random_periodic_spins_on_grid()
    ising_data.evolve(num_epochs=500, store_freq=4, verbose=True)
    returns = ising_data.get_return_per_epoch()
    burn_in_length = int(len(returns) * 0.25)
    returns = returns[burn_in_length:]  # Apply burn-in
    return (temp, returns)

# Function to append data to HDF5 file
def append_data_to_hdf5(data, filename):
    with h5py.File(filename, 'a') as f:
        for temp, returns in data:
            # Check if dataset exists, if not, create it
            if str(temp) not in f:
                dset = f.create_dataset(str(temp), data=returns, maxshape=(None,))
            else:
                dset = f[str(temp)]
                prev_size = len(dset)
                new_size = prev_size + len(returns)
                dset.resize((new_size,))
                dset[prev_size:new_size] = returns


# def run_ising_model(temp,n_epochs=500):
#     """
#     Returns the mean energy of the ising model after burn-in period (averaged value over the last 20% of epochs )
#     """
#     ising_obj = i4f.metropolis_ising(arr_len=40, temp=temp, block_rad=1, 
#                                        free_energy_update=5, rand_seed=None)  # Use a different random seed for each run
#     ising_obj.create_random_periodic_spins_on_grid()
#     ising_obj.evolve(num_epochs=n_epochs, store_freq=4, verbose=True)
#     epoch_energies = internal_energy(ising_obj)
#     burn_in_index = int(0.8 * len(epoch_energies))
#     mean_energy = np.mean(epoch_energies[burn_in_index:])
#     return mean_energy

# def internal_energy(ising_model) :
#     return [helmholtz_energy + ising_model.temp * spin_block_entropy for helmholtz_energy, spin_block_entropy in zip(ising_model.helmholtz_energy, ising_model.spin_block_entropy)]
