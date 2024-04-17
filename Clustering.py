import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch.autograd import Variable
from sklearn.exceptions import ConvergenceWarning
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
        kmeans.fit(img_flat)

        # Find the cluster containing intensity values strictly above self.threshold
        above_threshold_cluster = np.where(kmeans.cluster_centers_ > self.threshold)[0]

        # Check if there is a cluster above the threshold
        if len(above_threshold_cluster) > 0:
            # Get the label indices corresponding to the cluster above the threshold
            above_threshold_labels = np.where(np.isin(kmeans.labels_, above_threshold_cluster))[0]

            # Remove the cluster above the threshold from the cluster labels
            labels = np.delete(kmeans.labels_, above_threshold_labels)
        else:
            labels = kmeans.labels_

        # Calculate average pixel value of each remaining cluster
        unique_labels = np.unique(labels)
        avg_pixel_values = np.zeros(len(unique_labels))
        for i, label in enumerate(unique_labels):
            cluster_pixels = img_flat[labels == label]
            avg_pixel_values[i] = self.avg_pixel_value(cluster_pixels)

        # Select the cluster with the highest average pixel value
        selected_cluster_label = unique_labels[np.argmax(avg_pixel_values)]

        # Assign all pixels in the selected cluster to a new cluster label
        new_labels = np.where(labels == selected_cluster_label, 0, labels)

        # Reshape the labels to the shape of the original image
        new_labels = new_labels.reshape(img.shape)
        
        # Convert the selected cluster to a binary image
        binary_image = np.where(new_labels == selected_cluster_label, 1, 0).astype(np.uint8)
        return binary_image

    def avg_pixel_value(self, x):
        if x.max() > 0:
            mask = (x != 255) & (x != 0)
            sum_values = np.sum(x * mask)
            count_values = np.sum(mask)
            if count_values == 0:
                return np.nan
            return sum_values / count_values
        else:
            return 0
