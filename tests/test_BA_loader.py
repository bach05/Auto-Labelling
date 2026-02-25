import os
import pytest
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
from utils.datasets import CoupledCameraDataset
from torch.utils.data import DataLoader
from utils.io_utils import load_config

import torch

def test_dataloader():
    """Test the DataLoader with the CoupledCameraDataset."""
    config = load_config('configs/BA_loader_test.yaml')  # Path to the config file
    dataset_config = config.get('dataset', {})

    source_file = dataset_config.get('source_file')
    cameras = dataset_config.get('cameras', [])
    groups = [int(g) for g in dataset_config.get('groups', [])]
    start_datetime = dataset_config.get('start_datetime')
    end_datetime = dataset_config.get('end_datetime')

    # Initialize the dataset
    dataset = CoupledCameraDataset(
        coupling_json=source_file,
        groups=groups,
        cameras=cameras,
        date_from=start_datetime,
        date_to=end_datetime,
        transform=None,  # Optionally define any transformation here
        enable_pbar=True,
    )

    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate over the dataloader
    for i, (data, meta) in enumerate(dataloader):
        print(f"Batch {i + 1}")
        print(f"Data keys: {data.keys()}")
        print(f"Meta keys: {meta.keys()}")

        # Ensure the data is returned as a tensor
        assert isinstance(data['basler'][0], torch.Tensor), "Data is not a tensor."

        # Optionally stop after a few iterations for debugging
        if i >= 5:
            break


if __name__ == "__main__":
    # Run all tests
    pytest.main()