import torch
import pydicom
from skimage import transform
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
class RSNA_class_numpy_dataset(Dataset):  
    def __init__(self, file_paths, target, transform=None):
        """
        Initialize the RSNA dataset class.

        Args:
            file_paths (list): List of file paths to DICOM files.
            target (numpy.ndarray): Target labels.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.file_paths = file_paths
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):
        """
        Get data and target at the specified index.

        Args:
            index (int): Index of the data sample.

        Returns:
            tuple: Tuple containing the data and target.
        """
        # Read DICOM file at the specified index
        dicom_dataset = pydicom.dcmread(self.file_paths[index])
        # Extract pixel array data
        data = dicom_dataset.pixel_array  
        # Get target label
        y = self.target[index]

        # Apply transformation if specified
        if self.transform:
            data = self.transform(data)

        # Convert data to float32
        data = data.astype(np.float32)
        # Extract RescaleIntercept from DICOM metadata
        x = dicom_dataset.RescaleIntercept

        # Extract WindowCenter from DICOM metadata
        if isinstance(dicom_dataset.WindowCenter, pydicom.multival.MultiValue):
            c = float(dicom_dataset.WindowCenter[0])  
        else:
            c = float(dicom_dataset.WindowCenter)

        # Extract WindowWidth from DICOM metadata
        if isinstance(dicom_dataset.WindowWidth, pydicom.multival.MultiValue):
            w = float(dicom_dataset.WindowWidth[0])  
        else:
            w = float(dicom_dataset.WindowWidth)
        
        # Resize data to (512, 512)
        data = cv2.resize(data, (512, 512))

        return data, y, x, c, w

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.file_paths)
class test_lowdata_numpy_dataset(Dataset):
    def __init__(self, data, target, transform=None):
        """
        Initialize the test_lowdata_numpy_dataset class.

        Args:
            data (numpy.ndarray): Input data.
            target (numpy.ndarray): Target labels.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.data = torch.from_numpy(data).float()  # Convert input data to float tensor
        self.target = torch.from_numpy(target).float()  # Convert target labels to float tensor
        self.transform = transform  # Transformation to be applied to the data

    def __getitem__(self, index):
        """
        Get data and target at the specified index.

        Args:
            index (int): Index of the data sample.

        Returns:
            tuple: Tuple containing the data and target.
        """
        x = self.data[index]  # Get input data at the specified index
        y = self.target[index]  # Get target label at the specified index

        # Apply transformation if specified
        if self.transform:
            x = self.transform(x)

        return x, y  # Return input data and target label as a tuple

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)  # Return the length of the input data



class seg_lowdata_numpy_dataset(Dataset):
    def __init__(self, data, target, do_sampling=True, transform=None):
        """
        Initialize the seg_lowdata_numpy_dataset class.

        Args:
            data (numpy.ndarray): Input data.
            target (numpy.ndarray): Target labels.
            do_sampling (bool, optional): Flag indicating whether to perform weighted sampling. Default is True.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.data = torch.from_numpy(data).float()  # Convert input data to float tensor
        self.target = torch.from_numpy(target).float()  # Convert target labels to float tensor
        self.transform = transform  # Transformation to be applied to the data

        if do_sampling:
            # Assuming target is binary (0 or 1), create binary labels based on the condition
            labels = torch.LongTensor([1 if target[i, 1, :, :].max() == 1 else 0 for i in range(len(target))])

            # Calculate class frequencies for the entire dataset
            labels_counts = Counter(labels.numpy())

            # Calculate weights inversely proportional to class frequencies
            weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])

            # Assign weights based on the binary labels
            class_weights = weights[labels]

            # Create a sampler with replacement, assigning weights to each sample for training
            train_sampler = WeightedRandomSampler(class_weights, len(self.data), replacement=True)
        else:
            # If no sampling is needed, set sampler to None
            train_sampler = None

        self.train_sampler = train_sampler  # Weighted sampler for training data

    def __getitem__(self, index):
        """
        Get data and target at the specified index.

        Args:
            index (int): Index of the data sample.

        Returns:
            tuple: Tuple containing the data and target.
        """
        x = self.data[index]  # Get input data at the specified index
        y = self.target[index]  # Get target label at the specified index

        # Apply transformation if specified
        if self.transform:
            x = self.transform(x)

        return x, y  # Return input data and target label as a tuple

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)  # Return the length of the input data


class class_lowdata_numpy_dataset(Dataset):
    def __init__(self, data, target, do_sampling=True, transform=None):
        """
        Initialize the class_lowdata_numpy_dataset class.

        Args:
            data (numpy.ndarray): Input data.
            target (numpy.ndarray): Target labels.
            do_sampling (bool, optional): Flag indicating whether to perform weighted sampling. Default is True.
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.data = torch.from_numpy(data).float()  # Convert input data to float tensor
        self.target = torch.from_numpy(target).float()  # Convert target labels to float tensor
        self.transform = transform  # Transformation to be applied to the data

        if do_sampling:
            # Assuming target is binary (0 or 1), create binary labels based on the condition
            labelss = target

            # Calculate class frequencies for the entire dataset
            labels_counts = Counter(labelss)

            # Calculate weights inversely proportional to class frequencies
            weights = torch.FloatTensor([1 / labels_counts[0], 1 / labels_counts[1]]) * (labels_counts[0] + labels_counts[1])

            # Assign weights based on the binary labels
            class_weights = weights[target]

            # Create a sampler with replacement, assigning weights to each sample for training
            train_sampler = WeightedRandomSampler(class_weights, len(self.data), replacement=True)
        else:
            # If no sampling is needed, set sampler to None
            train_sampler = None

        self.train_sampler = train_sampler  # Weighted sampler for training data

    def __getitem__(self, index):
        """
        Get data and target at the specified index.

        Args:
            index (int): Index of the data sample.

        Returns:
            tuple: Tuple containing the data and target.
        """
        x = self.data[index]  # Get input data at the specified index
        y = self.target[index]  # Get target label at the specified index

        # Apply transformation if specified
        if self.transform:
            x = self.transform(x)

        return x, y  # Return input data and target label as a tuple

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)  # Return the length of the input data
class RSNA_seg_numpy_dataset(Dataset):  
    def __init__(self, file_paths, folder_address, transform=None):
        self.file_paths = file_paths
        self.folder_address = folder_address
        self.transform = transform

    def load_target(self, index):
        # Construct the file path for the target label using the index
        file_name = f"ID_{index:06d}.nii.gz"
        target_path = os.path.join(self.folder_address, file_name)
        
        # Load target labels from the constructed file path
        target_image = nib.load(target_path)
        target_data = target_image.get_fdata()
        return torch.from_numpy(target_data).float()

    def __getitem__(self, index):
        dicom_dataset = pydicom.dcmread(self.file_paths[index])
        data = dicom_dataset.pixel_array  

        if self.transform:
            data = self.transform(data)
        
        data = data.astype(np.float32)

        # Load target labels dynamically
        #try:
        y = self.load_target(index)
        #except FileNotFoundError:
         #   y = torch.zeros(1)  # Handle the case when target labels are not available
        
        x = dicom_dataset.RescaleIntercept
        
        if isinstance(dicom_dataset.WindowCenter, pydicom.multival.MultiValue):
            c = float(dicom_dataset.WindowCenter[0])  # Assuming you want the first value
        else:
            c = float(dicom_dataset.WindowCenter)
        
        if isinstance(dicom_dataset.WindowWidth, pydicom.multival.MultiValue):
            w = float(dicom_dataset.WindowWidth[0])  # Assuming you want the first value
        else:
            w = float(dicom_dataset.WindowWidth)
        
        data = cv2.resize(data, (512, 512))
        ss=dicom_dataset.RescaleSlope
        return data, y, x, c, w,ss

    def __len__(self):
        return len(self.file_paths)

