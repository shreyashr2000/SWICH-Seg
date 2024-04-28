import torch
import numpy as np
from windowing import window_ct
import torch.nn.functional as F
def unet_test(model, test_loader, device):
 model.eval()
 with torch.no_grad():
     pseudounetlabel=[]

     for inputs, targets in test_loader:
                one_d_data = torch.zeros((inputs.shape[0], 1, 512, 512)).to(device)
                for i in range(inputs.shape[0]):
                    gray_img = inputs[i,:,:]
                    r = window_ct(gray_img, w_level=40, w_width=80)  # Extract R channel
     

                    # Normalize each channel individually
                    r = r / r.max()
   

                    # Scale each channel to [0, 255] range
                    r = (r * 255).clamp(0, 255)
 
                    # Assign R channels to respective positions in the new tensor
                    if not torch.isinf(r).any().item():
                        one_d_data[i, 0, :, :] = r
    

                inputs, targets = one_d_data.to(device), targets.to(device)
                outputs = model(inputs)
                pred_seg = F.softmax(outputs, dim=1)
                for i in range(inputs.shape[0]):    
                  threshold_value = 0.5
                  thresholded_img = np.array(pred_seg[i,1,:,:])
                  thresholded_img[thresholded_img >= threshold_value] = 1
                  thresholded_img[thresholded_img < threshold_value] = 0
                  pseudounetlabel.append(thresholded_img)
     pseudo_unet_label=np.array(pseudounetlabel)
     return pseudo_unet_label
