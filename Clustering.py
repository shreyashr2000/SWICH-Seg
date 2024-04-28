import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch.autograd import Variable
from sklearn.exceptions import ConvergenceWarning
import cv2
np.random.seed(42)


class ImageSegmenter:
    def __init__(self, k=4, threshold=140):
        self.k = k
        self.threshold = threshold
    
    def segment_image(self, img):
        # Flatten the image array
        img_flat = img.reshape((-1, 1))

        # Perform KMeans clustering with k=self.k
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        kmeans.fit(img_flat.reshape(-1,1))
        
        # Reshape the labels to match the shape of the original image
        labels = kmeans.labels_.reshape(img.shape)

        # Calculate the average intensity of each cluster
        cluster_intensities = np.zeros(self.k)
        for i in range(self.k):
            cluster_pixels = img_flat[kmeans.labels_ == i]
         #   print(self.avg_pixel_value(cluster_pixels))
            if self.avg_pixel_value(cluster_pixels)<140:
             cluster_intensities[i] = self.avg_pixel_value(cluster_pixels)

        # Select clusters with intensity less than the threshold
        highest_intensity_cluster_index = np.argmax(cluster_intensities)
        if cluster_intensities.max()==0:
            blank_img=np.zeros((512,512))
            return blank_img
        # Create a binary image containing only the pixels of the selected cluster
        binary_image = np.where(labels == highest_intensity_cluster_index, 1, 0).astype(np.uint8)
        # Ensure the correct data type (uint8) for dilation
        kernel = np.ones((4, 4), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        return dilated_image
    def avg_pixel_value(self,x):
     if len(x) > 0: 
      if x.max()>0:

       mask = (x != 255) & (x != 0)

  # Calculate the sum of non-masked pixels and their total count
       sum_values = np.sum(x * mask)
       count_values = np.sum(mask)

  # Check if there are any non-masked pixels
       if count_values == 0:
        return np.nan

  # Calculate and return the average pixel value
       return sum_values / count_values
      else:
       return 0
     else:
        return 0
