import os
import os.path
import hashlib
import errno
from torch.utils.model_zoo import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
import glob as glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()

def normalize_array(x):
    # x: input numpy array with shape (W, H)
    x_min = np.amin(x)
    x_max = np.amax(x)
    y = (x - x_min) / (x_max - x_min)
    return y, x_min, x_max

def unnormalize_array(y, x_min, x_max):
    return y * (x_max - x_min) + x_min

def data_blurring(data_sample, us_size):
    # data_sample: torch tensor, size: [128, 128, 3]
    # us_size: image upscale size: 64
    # return: blurred torch tensor, size: [64, 64, 3]

    ds_size = 16
    resample_method = Image.NEAREST

    x_array, x_min, x_max = normalize_array(data_sample.numpy())
    im = Image.fromarray((x_array*255).astype(np.uint8))
    im_ds = im.resize((ds_size, ds_size))
    im_us = im_ds.resize((us_size, us_size), resample=resample_method)
    x_array_blur = np.asarray(im_us)

    # inverse-transform to the original value range
    x_array_blur = x_array_blur.astype(np.float32)/255.0
    x_array_blur = unnormalize_array(x_array_blur, x_min, x_max)

    return torch.from_numpy(x_array_blur)

def data_preprocessing(target, Image_Size):
    # Convert 128 x 128 target data to 64 x 64
    # reduce the resolution of target data to create 64 x 64 img data
    # target: torch tensor, size: [BS, 3, 128, 128]
    # Image_Size: 64 by default
    # img: torch tensor, size: [BS, 3, Image_Size, Image_Size]
    # output_target: torch tensor, size: [BS, 3, Image_Size, Image_Size]

    img = torch.zeros(target.size(0), target.size(1), Image_Size, Image_Size) # size: [32, 3, 64, 64]
    for idx in range(target.size(0)):
        x = target[idx]
        x = torch.permute(x, [1,2,0]) # x size: [128, 128, 3]
        x = data_blurring(x, Image_Size) # x size: [64, 64, 3]
        img[idx] =  torch.permute(x, [2,0,1])

    down_scale_transform = transforms.Resize(Image_Size)
    output_target = down_scale_transform(target)

    return img, output_target

# Create a customized dataset class for fno dataset

import torch
from torch.utils.data import Dataset

class FNO_Dataset(Dataset):
    """
    Fourier Neural Operator (FNO) Dataset class for fluid simulation data.
    
    This dataset class loads pre-processed fluid simulation data suitable for training
    Fourier Neural Operator models, which are commonly used for solving PDEs and modeling
    complex physical systems like fluid dynamics. The class handles data loading and
    preprocessing, particularly the permutation of dimensions to match the expected input
    format for deep learning models (batch, channels, height, width).
    
    The dataset is designed to work with PyTorch's DataLoader for efficient batch processing
    during training and evaluation of diffusion models for fluid super-resolution.
    
    Attributes:
        data (torch.Tensor): The loaded tensor data containing fluid simulation snapshots,
                           with dimensions rearranged to [batch, channels, height, width].
    """
    def __init__(self, data_dir):
        """
        Initialize the FNO Dataset by loading data from the specified directory.
        
        Args:
            data_dir (str): Path to the pre-processed fluid simulation data file (*.pt or *.pth).
                           This file should contain a PyTorch tensor with simulation data.
        
        Note:
            The input tensor is expected to have shape [n_samples, height, width, channels],
            which is then permuted to [n_samples, channels, height, width] to match PyTorch's
            conventional BCHW format for image-like data processing.
        """
        self.data = torch.load(data_dir)
        # Permute dimensions from [batch, height, width, channels] to [batch, channels, height, width]
        # This is necessary for compatibility with PyTorch's convolutional layers which expect BCHW format
        self.data = torch.permute(self.data, [0,3,1,2])

    def __len__(self):
        """
        Return the total number of fluid simulation samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a specific fluid simulation sample by its index.
        
        This method is called by PyTorch's DataLoader to fetch individual samples
        during training or evaluation.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            torch.Tensor: A single fluid simulation snapshot with shape [channels, height, width].
                         Typically represents vorticity or other fluid properties at a specific timestep.
        """
        return self.data[idx]


class KMFlowDataset(Dataset):
    """
    Kolmogorov Flow Dataset class for dynamic fluid simulation sequences.
    
    This dataset is specifically designed for Kolmogorov flow simulations, which are
    standard test cases in computational fluid dynamics representing 2D periodic flows
    driven by a sinusoidal forcing. Unlike the simpler FNO_Dataset, this dataset:
    
    1. Handles temporal sequences (multiple timesteps)
    2. Manages file-based data loading with caching for efficiency
    3. Performs normalization and resolution adjustments
    4. Supports train/test splitting
    
    The dataset is structured to return triplets of consecutive flow fields (3 timesteps),
    which is useful for training models that need to learn temporal dynamics in fluid flows.
    
    This advanced dataset class is particularly important for physics-informed diffusion models
    as it provides the temporal context needed for physical consistency in flow evolution.
    
    Attributes:
        train_fname_lst (list): List of training data file paths.
        test_fname_lst (list): List of testing data file paths.
        fname_lst (list): Currently active list (either training or testing).
        inner_steps (int): Number of steps within each outer time interval.
        outer_steps (int): Number of major time intervals in the simulation.
        resolution (int): Spatial resolution for the data (grid size).
        max_cache_len (int): Maximum number of samples to keep in memory cache.
        cache (dict): Memory cache storing loaded data to avoid repeated disk access.
        scaler (StandardScaler): Normalizes data to have zero mean and unit variance.
    """
    def __init__(self, data_dir, resolution=256, max_cache_len=3200,
                 inner_steps=32, outer_steps=10, train_ratio=0.9, test=False,
                 stat_path=None):
        """
        Initialize the Kolmogorov Flow Dataset with the specified parameters.
        
        This function sets up the dataset by:
        1. Finding all simulation files in the specified directory
        2. Creating train/test splits based on the provided ratio
        3. Setting up caching parameters for performance
        4. Either loading or computing normalization statistics
        
        Args:
            data_dir (str): Directory containing the simulation data files.
                           Files should be organized in subdirectories named 'seed*'.
            resolution (int): Target spatial resolution for the flow fields. Default: 256.
            max_cache_len (int): Maximum number of samples to keep in memory cache. Default: 3200.
            inner_steps (int): Number of simulation steps within each outer time interval. Default: 32.
            outer_steps (int): Number of major time intervals in the simulation. Default: 10.
            train_ratio (float): Fraction of data to use for training (remainder for testing). Default: 0.9.
            test (bool): If True, use test data; otherwise, use training data. Default: False.
            stat_path (str, optional): Path to pre-computed normalization statistics (mean, std).
                                     If None, statistics will be computed from the data. Default: None.
        
        Note:
            The dataset structure expects files organized as:
            data_dir/seed*/sol_t{outer_step}_step{inner_step}.npy
            
            Each file contains a 2D numpy array representing the flow field at that timestep.
        """
        # Find all seed directories in the data_dir
        fname_lst = glob.glob(data_dir + '/seed*')
        
        # Set random seed for reproducibility in train/test splitting
        np.random.seed(1)
        
        # Calculate how many samples to use for training based on train_ratio
        num_of_training_samples = int(train_ratio*len(fname_lst))
        
        # Shuffle the file list for random train/test split
        np.random.shuffle(fname_lst)
        
        # Split the file list into training and testing sets
        self.train_fname_lst = fname_lst[:num_of_training_samples]
        self.test_fname_lst = fname_lst[num_of_training_samples:]

        # Select either training or testing file list based on the 'test' flag
        if not test:
            self.fname_lst = self.train_fname_lst[:]
        else:
            self.fname_lst = self.test_fname_lst[:]

        # Store simulation parameters
        self.inner_steps = inner_steps  # Number of steps within each outer time interval
        self.outer_steps = outer_steps  # Number of major time intervals
        self.resolution = resolution    # Target spatial resolution
        self.max_cache_len = max_cache_len  # Maximum cache size to manage memory usage

        if stat_path is not None:
            self.stat_path = stat_path
            self.stat = np.load(stat_path)
            self.scaler = StandardScaler()
            self.scaler.mean_ = self.stat['mean']
            self.scaler.scale_ = self.stat['scale']
        else:
            self.prepare_data()
        self.cache = {}

    def __len__(self):
        """
        Calculate the total number of samples in the dataset.
        
        The total size is determined by the number of seed directories multiplied by
        the number of valid time combinations (considering all pairs of three consecutive 
        timesteps across all inner and outer steps).
        
        The -2 accounts for the fact that we need three consecutive timesteps for each sample,
        so the last two timesteps of the sequence cannot be starting points.
        
        Returns:
            int: Total number of triplet samples available in the dataset.
        """
        return len(self.fname_lst) * (self.inner_steps * self.outer_steps - 2)

    def prepare_data(self):
        """
        Calculate and store normalization statistics for the entire dataset.
        
        This method:
        1. Initializes a StandardScaler for data normalization
        2. Iterates through a subset of the data (sampling every 4th frame)
        3. Incrementally fits the scaler to compute mean and standard deviation
        4. Reports the final statistics
        
        The partial_fit method allows for memory-efficient calculation of statistics
        without loading the entire dataset into memory at once.
        
        This pre-computation of statistics is essential for consistent normalization
        across the dataset, which improves training stability and convergence.
        """
        # Initialize StandardScaler for normalization
        self.scaler = StandardScaler()
        
        # Process all files in the selected file list (train or test)
        for data_dir in tqdm(self.fname_lst):
            # Iterate through outer time steps (major intervals)
            for i in range(self.outer_steps):
                # Sample every 4th inner step to reduce computational load
                for j in range(0, self.inner_steps, 4):
                    # Construct the filename for this timestep
                    fname = os.path.join(data_dir, f'sol_t{i}_step{j}.npy')
                    
                    # Load the data in memory-map mode (doesn't load entire file into memory)
                    # and subsample spatially every 4th point to reduce memory usage
                    data = np.load(fname, mmap_mode='r')[::4, ::4]
                    
                    # Reshape to column vector for StandardScaler
                    data = data.reshape(-1, 1)
                    
                    # Update the scaler statistics incrementally
                    self.scaler.partial_fit(data)
                    
                    # Release memory
                    del data

        # Print the resulting statistics for verification
        print(f'Data statistics, mean: {self.scaler.mean_}, standard deviation: {self.scaler.scale_}')

    def preprocess_data(self, data):
        """
        Preprocess raw simulation data by subsampling and normalizing.
        
        This function performs two key operations:
        1. Downsamples the data to the target resolution through strided indexing
        2. Normalizes the data using pre-computed mean and standard deviation
        
        Args:
            data (numpy.ndarray): Raw simulation data of arbitrary resolution
            
        Returns:
            numpy.ndarray: Processed data of shape [resolution, resolution]
                          with zero mean and unit variance
                          
        The normalization is essential for stable training of deep learning models
        and helps the diffusion model work in a standardized data space.
        """
        # Get the original size of the data
        s = data.shape[0]
        
        # Calculate subsampling factor to reach target resolution
        sub = int(s // self.resolution)
        
        # Apply strided subsampling to downsample the data
        data = data[::sub, ::sub]

        # Apply standardization (zero mean, unit variance)
        # First reshape to vector, normalize, then reshape back to 2D grid
        data = self.scaler.transform(data.reshape(-1, 1)).reshape((self.resolution, self.resolution))
        
        return data

    def save_data_stats(self, out_dir):
        """
        Save the calculated data statistics (mean and standard deviation) to a file.
        
        This allows for consistent normalization between training and inference,
        as well as when resuming training from checkpoints.
        
        Args:
            out_dir (str): Path where the statistics will be saved as a .npz file
        """
        # Save mean and scale (std deviation) to NPZ file format
        np.savez(out_dir, mean=self.scaler.mean_, scale=self.scaler.scale_)

    def __getitem__(self, idx):
        """
        Retrieve a triplet of consecutive flow field snapshots by index.
        
        This method implements complex logic to:
        1. Map the linear index to specific simulation and time indices
        2. Handle edge cases at the boundaries between inner and outer steps
        3. Load and preprocess the required data with efficient caching
        
        The triplets of consecutive snapshots are essential for the physics-informed
        diffusion model as they provide the temporal context needed to compute accurate
        physical derivatives (especially time derivatives) for the vorticity equation.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            numpy.ndarray: Array of shape [3, resolution, resolution] containing
                          three consecutive normalized flow field snapshots
                          
        Cache Implementation:
            - Uses a dictionary to store recently accessed samples
            - Limits memory usage with max_cache_len
            - Implements random replacement when cache is full
        """
        # Calculate which simulation (seed) and frame within that simulation to access
        seed = idx // (self.inner_steps * self.outer_steps - 2)
        frame_idx = idx % (self.inner_steps * self.outer_steps - 2)
        
        # Handle special case: end of inner steps sequence (need to move to next outer step)
        if frame_idx % self.inner_steps == 31:
            inner_step = frame_idx % self.inner_steps
            outer_step = frame_idx // self.inner_steps
            next_outer_step = outer_step + 1
            next_next_outer_step = next_outer_step
            next_inner_step = 0
            next_next_inner_step = 1
        # Handle special case: second-to-last step (spans across outer steps boundary)
        elif frame_idx % self.inner_steps == 30:
            inner_step = frame_idx % self.inner_steps
            outer_step = frame_idx // self.inner_steps
            next_outer_step = outer_step
            next_next_outer_step = next_outer_step + 1
            next_inner_step = inner_step + 1
            next_next_inner_step = 0
        # Standard case: three consecutive steps within same outer step
        else:
            inner_step = frame_idx % self.inner_steps
            outer_step = frame_idx // self.inner_steps
            next_outer_step = outer_step
            next_next_outer_step = next_outer_step
            next_inner_step = inner_step + 1
            next_next_inner_step = next_inner_step + 1

        # Create unique identifier for this sample (for caching)
        id = f'seed{seed}_t{outer_step}_step{inner_step}'

        # Return cached result if available to avoid repeated disk I/O
        if id in self.cache.keys():
            return self.cache[id]
        else:
            # Get the directory for this simulation
            data_dir = self.fname_lst[seed]
            
            # Load and preprocess first frame (t)
            fname0 = os.path.join(data_dir, f'sol_t{outer_step}_step{inner_step}.npy')
            frame0 = np.load(fname0, mmap_mode='r')
            frame0 = self.preprocess_data(frame0)

            # Load and preprocess second frame (t+1)
            fname1 = os.path.join(data_dir, f'sol_t{next_outer_step}_step{next_inner_step}.npy')
            frame1 = np.load(fname1, mmap_mode='r')
            frame1 = self.preprocess_data(frame1)

            # Load and preprocess third frame (t+2)
            fname2 = os.path.join(data_dir, f'sol_t{next_next_outer_step}_step{next_next_inner_step}.npy')
            frame2 = np.load(fname2, mmap_mode='r')
            frame2 = self.preprocess_data(frame2)

            # Stack frames along a new axis to create a 3D tensor [3, resolution, resolution]
            frame = np.concatenate((frame0[None, ...], frame1[None, ...], frame2[None, ...]), axis=0)
            
            # Store in cache for future access
            self.cache[id] = frame

            # If cache exceeds size limit, remove a random entry to prevent memory overflow
            if len(self.cache) > self.max_cache_len:
                self.cache.pop(np.random.choice(self.cache.keys()))
                
            return frame


class KMFlowTensorDataset(Dataset):
    """
    Memory-optimized dataset for pre-loaded Kolmogorov Flow tensor data.
    
    This dataset class is an alternative to KMFlowDataset, designed for scenarios
    where the entire dataset can be efficiently loaded into memory at once from
    a single file (typically a .npy array). It offers several advantages:
    
    1. Faster data access during training (avoids file I/O overhead)
    2. Simplified file management (single file vs. directory structure)
    3. Memory efficiency through intelligent caching and indexing
    
    Like KMFlowDataset, it provides triplets of consecutive timesteps for each sample,
    which is essential for computing time derivatives in the physics-informed model.
    
    This dataset is particularly useful for smaller simulations or when working with
    pre-processed data collections that have already been consolidated.
    
    Attributes:
        all_data (numpy.ndarray): Full data tensor containing all simulation samples
        train_idx_lst (list): Indices of simulations used for training
        test_idx_lst (list): Indices of simulations used for testing
        idx_lst (list): Currently active list (either training or testing)
        time_step_lst (numpy.ndarray): List of available timestep indices
        cache (dict): Memory cache to avoid redundant computation
        max_cache_len (int): Maximum cache size to limit memory usage
        stat (dict): Statistics for data normalization (mean, std)
    """
    def __init__(self, data_path,
                 train_ratio=0.9, test=False,
                 stat_path=None,
                 max_cache_len=4000):
        """
        Initialize the KMFlowTensorDataset with pre-loaded tensor data.
        
        Args:
            data_path (str): Path to the consolidated .npy data file containing
                           all simulation data with shape [n_simulations, n_timesteps, height, width]
            train_ratio (float): Fraction of simulations to use for training. Default: 0.9
            test (bool): If True, use test data; otherwise, use training data. Default: False
            stat_path (str, optional): Path to pre-computed normalization statistics.
                                     If None, statistics will be computed from the data. Default: None
            max_cache_len (int): Maximum number of samples to cache in memory. Default: 4000
        """
        # Set fixed random seed for reproducible train/test splits
        np.random.seed(1)
        
        # Load the entire dataset into memory
        self.all_data = np.load(data_path)
        print('Data set shape: ', self.all_data.shape)
        
        # Generate indices for all simulations
        idxs = np.arange(self.all_data.shape[0])
        
        # Calculate split point between training and testing sets
        num_of_training_seeds = int(train_ratio*len(idxs))
        
        # Split indices into training and testing sets
        # Note: no shuffling applied as commented out below
        # np.random.shuffle(idxs)
        self.train_idx_lst = idxs[:num_of_training_seeds]
        self.test_idx_lst = idxs[num_of_training_seeds:]
        
        # Create array of all possible timestep indices (except last two)
        # We need three consecutive frames, so last two can't be starting points
        self.time_step_lst = np.arange(self.all_data.shape[1]-2)
        
        # Select either training or testing indices based on the 'test' flag
        if not test:
            self.idx_lst = self.train_idx_lst[:]
        else:
            self.idx_lst = self.test_idx_lst[:]
            
        # Initialize cache dictionary and set maximum cache size
        self.cache = {}
        self.max_cache_len = max_cache_len

        # Either load existing statistics or compute them from the data
        if stat_path is not None:
            self.stat_path = stat_path
            self.stat = np.load(stat_path)
        else:
            self.stat = {}
            self.prepare_data()

    def __len__(self):
        """
        Calculate the total number of samples in the dataset.
        
        The total count is the product of the number of simulations (seeds)
        and the number of possible starting timesteps in each simulation.
        
        Returns:
            int: Total number of triplet samples available in the dataset.
        """
        return len(self.idx_lst) * len(self.time_step_lst)

    def prepare_data(self):
        """
        Calculate normalization statistics (mean and standard deviation) for the dataset.
        
        Unlike KMFlowDataset, this can directly compute statistics from the loaded tensor
        without iterating through files, making it more efficient.
        
        The statistics are used to standardize the data (zero mean, unit variance),
        which is essential for stable training of the diffusion model.
        """
        # Calculate mean across all training data (reshape to flatten spatial dimensions)
        self.stat['mean'] = np.mean(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        # Calculate standard deviation across all training data
        self.stat['scale'] = np.std(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        
        data_mean = self.stat['mean']
        data_scale = self.stat['scale']
        # Print statistics for verification
        print(f'Data statistics, mean: {data_mean}, scale: {data_scale}')

    def preprocess_data(self, data):
        """
        Normalize input data using pre-computed statistics.
        
        Args:
            data (numpy.ndarray): Raw data to normalize
            
        Returns:
            numpy.ndarray: Normalized data as float32 (for memory efficiency)
            
        The normalization transforms the data to have zero mean and unit variance,
        which is standard practice for neural network inputs to improve training stability.
        """
        # Apply standardization: (x - mean) / std
        data = (data - self.stat['mean']) / (self.stat['scale'])
        # Convert to float32 to reduce memory usage while maintaining precision
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        """
        Save the dataset's normalization statistics to a file.
        
        Args:
            out_dir (str): Path where the statistics will be saved as a .npz file
        
        This allows for consistent normalization when resuming training or during inference.
        """
        # Save mean and standard deviation to NPZ file
        np.savez(out_dir, mean=self.stat['mean'], scale=self.stat['scale'])

    def __getitem__(self, idx):
        """
        Retrieve a triplet of consecutive flow field snapshots by index.
        
        This method maps the linear index to specific simulation and timestep indices,
        then extracts and normalizes the corresponding data frames.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            numpy.ndarray: Array of shape [3, height, width] containing
                          three consecutive normalized flow field snapshots
                          
        The cache implementation improves efficiency by avoiding redundant
        normalization operations for frequently accessed samples.
        """
        # Map linear index to simulation index and timestep
        seed = self.idx_lst[idx // len(self.time_step_lst)]
        frame_idx = idx % len(self.time_step_lst)
        
        # Use the linear index as cache key
        id = idx

        # Return cached result if available
        if id in self.cache.keys():
            return self.cache[id]
        else:
            # Preprocess three consecutive frames
            frame0 = self.preprocess_data(self.all_data[seed, frame_idx])
            frame1 = self.preprocess_data(self.all_data[seed, frame_idx+1])
            frame2 = self.preprocess_data(self.all_data[seed, frame_idx+2])

            # Stack frames along a new axis
            frame = np.concatenate((frame0[None, ...], frame1[None, ...], frame2[None, ...]), axis=0)
            
            # Store in cache for future access
            self.cache[id] = frame

            # If cache exceeds size limit, remove a random entry
            if len(self.cache) > self.max_cache_len:
                self.cache.pop(self.cache.keys()[np.random.choice(len(self.cache.keys()))])
                
            return frame






