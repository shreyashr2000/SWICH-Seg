import random
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from skimage.util import random_noise
import torch
from torchvision import transforms
from PIL import Image
#=========================================================================================
# Truenet augmentations function
# Vaanathi Sundaresan
# 11-03-2021, Oxford
#=========================================================================================


##########################################################################################
# Define transformations with distance maps
##########################################################################################

def translate1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    translated_label=np.zeros((2, 512, 512))
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_label[1,:,:] = shift(label[1,:,:], (offsetx, offsety), order=0, mode='nearest')
    translated_label[0,:,:]=np.logical_not(translated_label[1,:,:])
    return translated_im, translated_label






def rotate1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    new_lab=np.zeros((2, 512, 512))
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    #new_lab[0,:,:] = rotate(label[0,:,:], float(theta), reshape=False, order=order, mode='nearest')
    new_lab[1,:,:] = rotate(label[1,:,:], float(theta), reshape=False, order=0, mode='nearest')
    new_lab[1,:,:] = (new_lab[1,:,:] > 0.5).astype(float)
    new_lab[0,:,:]=np.logical_not(new_lab[1,:,:])
    return new_img, new_lab





def flip1(image, label):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),  # Set p=1 to always flip
    ])
    
    # Apply the transformation to the image and label
    flipped_image = transform(image)
    flipped_label = transform(label)
    flipped_image=np.array(flipped_image)
    flipped_label=np.array(flipped_label)
    return flipped_image, flipped_label
def scale1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    """
    Apply random scaling augmentation to the given image and label.
    
    :param image: PIL image representing the original image.
    :param label: PIL image representing the label or mask.
    :return: Tuple containing the randomly scaled image and label.
    """
    # Define a random scaling factor sampled uniformly from [0.9, 1.1]
    scale_factor = torch.FloatTensor(1).uniform_(0.9, 1.1).item()
     # Define the transformation to perform random scaling for the image
    scaled_image=image*scale_factor        
    scaled_image=np.array(scaled_image)
    return scaled_image, label
def augment1(image, label):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    if len(image.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'flip': flip1, 'translate': translate1,
                                     'rotate': rotate1, 'scale': scale1}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(0, len(available_transformations))
        num_transformations = 0
        transformed_image = image
        transformed_label = label
        if num_transformations_to_apply==0:
            return image,label
        while num_transformations < num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_label = available_transformations[key](transformed_image, transformed_label)
            num_transformations += 1
        
        return transformed_image, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')



def translate2(image):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """

    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')

    return translated_im




def rotate2(image):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """

    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    return new_img




def flip2(image):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),  # Set p=1 to always flip
    ])
    if isinstance(image,np.ndarray):
     image=torch.from_numpy(image)

    flipped_image = transform(image)

    return flipped_image






def scale2(image):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    """
    Apply random scaling augmentation to the given image and label.
    
    :param image: PIL image representing the original image.
    :param label: PIL image representing the label or mask.
    :return: Tuple containing the randomly scaled image and label.
    """
    
    # Define a random scaling factor sampled uniformly from [0.9, 1.1]
    scale_factor = torch.FloatTensor(1).uniform_(0.9, 1.1).item()
    
     # Define the transformation to perform random scaling for the image

    scaled_image=image*scale_factor    
    
 #   scaled_image=np.array(scaled_image)
    return scaled_image





def augment2(image):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    if len(image.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'flip': flip2, 'translate': translate2,
                                     'rotate': rotate2, 'scale': scale2}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(0, len(available_transformations))
        num_transformations = 0
        transformed_image = image
      #  transformed_label = None
        if num_transformations_to_apply==0:
            return image
        while num_transformations < num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](transformed_image)
            num_transformations += 1
        
        return transformed_image
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')
