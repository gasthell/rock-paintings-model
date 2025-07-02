import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def unsupervised_segmentation(image, k, spatial_weight=1.0):
    """
    Performs unsupervised segmentation using K-Means on color and spatial features.
    
    Args:
        image (np.array): The input BGR image.
        k (int): The number of desired segments (clusters).
        spatial_weight (float): How much to weight the pixel coordinates. Tune this value.
    
    Returns:
        The segmented image.
    """
    h, w, _ = image.shape

    # --- 1. Create the Feature Vector ---
    # Reshape the image to a list of pixels (N, 3)
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Create the (x, y) coordinates for each pixel
    # np.mgrid creates two grids, one for y-coordinates and one for x-coordinates
    yy, xx = np.mgrid[0:h, 0:w]
    
    # Reshape the coordinate grids to be single columns
    xy_features = np.stack((xx.flatten(), yy.flatten()), axis=1)
    
    # Apply the spatial weight. This balances color vs. position.
    # We normalize coordinates to be roughly in the same range as colors.
    xy_features = np.float32(xy_features) * spatial_weight * (255 / max(h, w))
    
    # Concatenate color features and spatial features
    # Each pixel is now represented by (B, G, R, X, Y)
    features = np.concatenate((pixels, xy_features), axis=1)

    # --- 2. Apply K-Means Clustering ---
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # --- 3. Reconstruct the Segmented Image ---
    # The 'centers' now have 5 dimensions. We only care about the first 3 (the color part).
    quantized_colors = np.uint8(centers[:, :3])
    
    # Map each label to its corresponding color
    quantized_image = quantized_colors[labels.flatten()]
    
    # Reshape the image back to its original dimensions
    quantized_image = quantized_image.reshape((image.shape))
    
    return quantized_image

if __name__ == "__main__":
    image_path = 'Deeplabv3-resnet/augmented_data/2.png'

    image = cv2.imread(image_path)
    
    # --- Perform Segmentation ---
    # Number of segments
    k = 3
    
    # Tune this weight to balance color vs. spatial closeness.
    # Good values are typically between 0.5 and 2.0.
    spatial_weight = 0.5
    
    print("Performing unsupervised segmentation...")
    segmented_image = unsupervised_segmentation(image, k, spatial_weight)
    print("Segmentation complete.")

    # --- Display the Results ---
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Unsupervised Segmentation (k={k}, weight={spatial_weight})')
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.savefig('unsupervised_segmentation_output.png')
    plt.show()