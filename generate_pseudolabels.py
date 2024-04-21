import torch
import numpy as np
import cv2
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from gradcam import GradCamHook
from clustering import   ImageSegmenter
def get_pseudolabels(train_loader, val_loader, model, device, target_layer, threshold_value=0.7):
    """
    Compute GradCAM heatmaps for the given model and dataset.

    Args:
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        model: PyTorch model for which GradCAM is computed.
        device: Device on which to perform computations (e.g., 'cuda' or 'cpu').
        GradCamHook: Hook object for computing GradCAM.
        threshold_value: Threshold value for thresholding the GradCAM heatmap.

    Returns:
        trainpseudolabels: Numpy array containing binary masks for train data.
        valpseudolabels: Numpy array containing binary masks for validation data.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to store binary masks for train and validation data
    trainpseudolabels = []
    valpseudolabels = []

    # Iterate over the data loaders (train and validation)
    for loader, pseudolabels in [(train_loader, trainpseudolabels), (val_loader, valpseudolabels)]:
        # Iterate over batches
        for batch_idx, (data, labels) in enumerate(loader):
            data = data.to(device)
            labels = labels.to(device)

            # Iterate over data samples in the batch
            for i in range(data.shape[0]):
                img_tensor = torch.zeros((1, 3, 512, 512))
                img_tensor[0, :, :, :] = data[i, :, :, :]
                x = data[i, 1, :, :].detach().cpu().numpy()
                img_tensor = img_tensor.float()
                img_tensor = img_tensor.to(device)

                prediction = model(img_tensor)
                predd = torch.sigmoid(prediction)
                predd = torch.round(predd)

                # Compute GradCAM only if label is 1
                if labels[i] == 1:
                    # Compute GradCAM
                    activations = GradCamHook.activations
                    prediction.backward()
                    gradients = GradCamHook.gradients
                    weights = gradients.mean(dim=[2, 3], keepdim=True)
                    gradcam = (weights * activations).sum(dim=1, keepdim=True)

                    # Apply ReLU activation function
                    gradcam = F.relu(gradcam)

                    # Convert GradCAM to numpy array
                    gradcam = gradcam.detach().cpu().numpy()

                    # Resize GradCAM to match image size
                    image = cv2.resize(gradcam[0, 0, :, :], (512, 512))

                    # Normalize image
                    min_val = np.min(image)
                    max_val = np.max(image)
                    normalized_img = (image - min_val) / (max_val - min_val)

                    # Apply thresholding
                    thresholded_img = np.array(normalized_img)
                    thresholded_img[thresholded_img > threshold_value] = 1
                    thresholded_img[thresholded_img <= threshold_value] = 0

                    # Multiply by original subdural_windowed image
                    thresholded_img *= data[i, 1, :, :].cpu().numpy()

                    # Call ImageSegmenter function
                    segmenter = ImageSegmenter(k=4, threshold=140)

                    # Segment the image
                    binary_image = segmenter.segment_image(thresholded_img)
                    binary_mask = ImageSegmenter(thresholded_img)

                    # Perform binary dilation
                    binary_mask = binary_dilation(binary_mask, structure=np.ones((4, 4)))
                    binary_mask_onehot = np.zeros((2, 512, 512))
                    binary_mask_onehot[0] = 1 - binary_mask  # First channel: complement of binary mask
                    binary_mask_onehot[1] = binary_mask      # Second channel: binary mask
                else:
                    # Save zeros if label is not 1
                    binary_mask_onehot[1] = np.zeros(( 512, 512))
                    binary_mask_onehot[0]=np.ones((512,512))
                # Append binary mask to pseudolabels list
                pseudolabels.append(binary_mask_onehot)

    # Convert lists to numpy arrays
    trainpseudolabels = np.array(trainpseudolabels)
    valpseudolabels = np.array(valpseudolabels)

    return trainpseudolabels, valpseudolabels
