import torch
import numpy as np
import cv2
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from gradcam import GradCamHook
from clustering import   ImageSegmenter
from windowing import window_image
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
        for batch_idx, (data, labels,wc,ww,inter,slp) in enumerate(loader):
            data = data.to(device)
            labels = labels.to(device)
            # Iterate over data samples in the batch
            for i in range(data.shape[0]):
                img_tensor = torch.zeros((1, 3, 512, 512))
                r = window_image(data[i,:,:],  window_center=wc[i], window_width=ww[i], intercept=inter[i], slope=slp[i])  # Extract R channel
                g = window_image(data[i,:,:], window_center=80, window_width=200, intercept=inter[i], slope=slp[i]) # Extract G channel
                b = window_image(data[i,:,:], window_center=600, window_width=2800, intercept=inter[i], slope=slp[i]) # Extract B channel

                # Normalize each channel individually
                r = r / r.max()
                g = g / g.max()
                b = b / b.max()
                r = (r * 255).clamp(0, 255)
                g = (g * 255).clamp(0, 255)
                b = (b * 255).clamp(0, 255)

                # Assign R, G, B channels to respective positions in the new tensor
                if not torch.isinf(r).any().item():
                    img_tensor[0, 0, :, :] = r
                if not torch.isinf(g).any().item():
                    img_tensor[0, 1, :, :] = g
                if not torch.isinf(b).any().item():
                    img_tensor[0, 2, :, :] = b
               # subdural_window = img_tensor[0, 1, :, :].detach().cpu().numpy()   # Subdural_Window
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
                    thresholded_img *= img_tensor[0, 1, :, :].cpu().numpy()

                    # Call ImageSegmenter function
                    segmenter = ImageSegmenter(k=4, threshold=140)

                    # Segment the image
                    binary_mask = segmenter.segment_image(thresholded_img)
                    # Perform binary dilation
                    binary_mask = binary_dilation(binary_mask, structure=np.ones((4, 4)))
                    output_folder = 'RSNA_Pseudo_Labels'
                    os.makedirs(output_folder, exist_ok=True)
                    filename = os.path.join(output_folder, f'RSNA_PSEUDOMASK_{32 * batch_idx + i}.nii.gz')

                    nii_img = nib.Nifti1Image(binary_mask, affine=np.eye(4))
                    nib.save(nii_img, filename)


                    output_folder='RSNA_Pseudo_labels'
                    nii_img = nib.Nifti1Image(binary_mask, affine=np.eye(4)) 
 
    return 'RSNA_Pseudo_Labels'
