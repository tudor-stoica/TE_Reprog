import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class classLibrary:
    def __init__(self):
        """Utility functions for data preparation and manipulation."""

    # Prepare a dataset by processing files in a directory, applying sliding windows, and standardizing the data.
    def data_conca(self, file_dirs, time_len, unlist, num_class, num_dataset):
        num_dataset = num_dataset
        unlist = set(unlist)  # Convert unlist to a set for faster lookups
        data_set = np.zeros(shape=(1, time_len, 51))  # Initialize an empty dataset

        # Standardize the dataset using the first file as the reference
        standardizer = StandardScaler()
        first_file = os.path.join(file_dirs, os.listdir(file_dirs)[0])  # Assuming the first file exists
        fault0 = pd.read_csv(first_file).to_numpy()
        standardizer.fit(fault0[1:num_dataset, :])

        # Process all files in the directory
        num = 0
        for root, dirs, files in os.walk(file_dirs):
            for file in files:
                
                # Limit to number of 
                if num >= num_class:
                    break

                # Skip files for excluded classes
                if num in unlist:
                    num += 1
                    continue

                file_dir = os.path.join(file_dirs, file)
                data = pd.read_csv(file_dir).to_numpy()

                # Standardize the data
                data = standardizer.transform(data)
                data = np.concatenate(
                    (data[:num_dataset, 0:45], data[:num_dataset, 46:49], data[:num_dataset, 50:52]), axis=1
                )

                # Add labels to the data
                labels = np.full((len(data), 1), num)
                data = np.concatenate((data, labels), axis=1)

                # Apply a sliding window
                data_temp = np.zeros((len(data) - time_len + 1, time_len, data.shape[1]))
                for i in range(len(data) - time_len + 1):
                    for j in range(time_len):
                        data_temp[i, j] = data[i + j]

                # Add the processed data to the dataset
                data_set = np.concatenate((data_set, data_temp), axis=0)
                num += 1

        # Remove the initial empty entry
        data_set = data_set[1:, :, :]
        return data_set

# PyTorch Dataset for handling 3D arrays of time-series data.
class DealDataset(Dataset):

    # Initialize the dataset with data and labels.
    def __init__(self, xy, dim, transforms=None):
        xy = np.array(xy)
        if dim == 4:
            xy_x = xy[:, :, 0:-1]
            xy_x = np.reshape(xy_x, (xy_x.shape[0], 1, xy_x.shape[1], xy_x.shape[2]))
        elif dim == 3:
            xy_x = xy[:, 0:-1]
            xy_x = np.reshape(xy_x, (xy_x.shape[0], 1, xy_x.shape[1]))
        else:
            xy_x = xy[:, 0:-1]

        self.x_data = torch.from_numpy(xy_x).float()
        if dim == 4:
            self.y_data = torch.from_numpy(xy[:, 0, -1])
        else:
            self.y_data = torch.from_numpy(xy[:, -1])

        self.len = xy.shape[0]
        self.transform = transforms

    # Retrieve a single data point and its label.
    def __getitem__(self, index):
        x, y = self.x_data[index], self.y_data[index]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    # Get the length of the dataset.
    def __len__(self):
        return self.len