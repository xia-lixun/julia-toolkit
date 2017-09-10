import h5py

# Create random data
import numpy as np
data_matrix = np.random.uniform(-1, 1, size=(10, 3))

# Write data to HDF5
data_file = h5py.File('example.h5', 'w')
data_file.create_dataset('group_name', data=data_matrix)
data_file.close()

# read data
f = h5py.File('example.h5', 'r')
dm = f.get('group_name')
dm.value