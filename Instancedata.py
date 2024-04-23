import nibabel as nib
import numpy as np

def load_ins_data(data, mask):
    """
    Load CT scan data and convert corresponding masks to one-hot encoding from the INSTANCE Challenge dataset.

    Args:
        data (list): List of file paths to CT scan images in NIfTI format (.nii or .nii.gz).
        label (list): List of file paths to corresponding masks in NIfTI format.

    Returns:
        tuple: A tuple containing:
            - ins_data (ndarray): 3D array of CT scan data.
            - ins_mask (ndarray): 3D array masks.
            - ins_label (ndarray): 1D array of binary labels indicating presence of target.

    Note:
        This function is specifically designed for processing CT scans and masks from the INSTANCE Challenge dataset.
        It assumes that the CT scans and masks are stored in NIfTI format.
        The CT scans are assumed to be 3D volumes.
        The masks are assumed to be binary, and they are converted to one-hot encoding.
        The function repeats the last slice if the number of slices in a scan is less than 32 to standardize the number of slices.
        It rotates the data and masks to ensure that the head is facing up.
        The binary labels indicate the presence (1) or absence (0) of the target in each slice.

    """
    img_data = []  # List to store loaded CT scan data
    mask_data = []  # List to store loaded masks
    ins_data = np.zeros((len(data) * 32, 512, 512))  # Initialize array for CT scan data
    ins_mask = np.zeros((len(data) * 32, 2, 512, 512))  # Initialize array for masks
    ins_label = np.zeros(len(data) * 32)  # Initialize array for binary labels
    
    # Load CT scan data and masks
    for i in range(len(data)):
        nifti_data = nib.load(data[i])
        img_data.append(nifti_data.get_fdata())
        
        nifti_mask = nib.load(mask[i])
        z=nifti_mask.get_fdata()
        z[z>0]=1
        mask_data.append(z)
    # Process CT scan data and masks
    for i in range(len(data)):
        for j in range(32):
            if j < data[i].shape[2]:
                ins_data[i * 32 + j, :, :] = np.rot90(img_data[i][:, :, j])
                mask = np.rot90(mask_data[i][:, :, j])
                ins_mask[i * 32 + j, 0, :, :] = 1 - mask  # Background
                ins_mask[i * 32 + j, 1, :, :] = mask  # Target
            else:
                ins_data[i * 32 + j, :, :] = np.rot90(img_data[i][:, :, j-1])
                mask = np.rot90(mask_data[i][:, :, j-1])
                ins_mask[i * 32 + j, 0, :, :] = 1 - mask  # Background
                ins_mask[i * 32 + j, 1, :, :] = mask  # Target
                
            # Set binary label based on the presence of target in the mask
            if ins_mask[i * 32 + j, 1, :, :].max() > 0:
                ins_label[i * 32 + j] = 1
                
    return ins_data, ins_mask, ins_label
