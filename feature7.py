import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import feature
from colorama import Fore, Back, Style

# calculate mean variance
def calculate_mean_variance(image):
    rows, cols = image.shape
    division_rows = np.array_split(image, 3, axis=0)
    division_cols = [np.array_split(row, 3, axis=1) for row in division_rows]

    feature_vector1 = []
    feature_vector2 = []

    for i in range(3):
        for j in range(3):
            region = division_cols[i][j]
            mean_value = np.mean(region)
            variance_value = np.var(region)
            feature_vector1.extend([mean_value])
            feature_vector2.extend([variance_value])

    return feature_vector1, feature_vector2

# calculate lbp
def calculate_lbp(image, radius=1, samples=8):
    quantized_lbp = local_binary_pattern(image, samples, radius, method='uniform')
    return quantized_lbp.astype(np.uint8)

# calculate ltp
def calculate_ltp(image):
    return local_binary_pattern(image, P=8, R=1, method='uniform')

# calculate lpq
def calculate_lpq(image, radius=1, samples=8):
    lbp = local_binary_pattern(image, P=samples, R=radius, method='uniform')
    lbp_min = lbp.min()
    lbp_max = lbp.max()
    quantized_lbp = np.floor((lbp - lbp_min) / (lbp_max - lbp_min) * 255)
    return quantized_lbp.astype(np.uint8)

# calculate hog
def calculate_hog(image):
    return feature.hog(image, pixels_per_cell=(8, 8), block_norm='L2-Hys')


def calculate_features(image_path):
    input_image = cv2.imread(image_path)
    resized_image = cv2.resize(input_image, (120, 120))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    mean_var, variance_var = calculate_mean_variance(gray_image)
    lbp = calculate_lbp(gray_image)
    ltp = calculate_ltp(gray_image)
    lpq = calculate_lpq(gray_image)
    hog = calculate_hog(gray_image)

    return mean_var, variance_var, lbp, ltp, lpq, hog

def compare_images(image_path1, image_path2):
    # Calculate features for the first image
    mean_var1, variance_var1, lbp1, ltp1, lpq1, hog1 = calculate_features(image_path1)

    # Calculate features for the second image
    mean_var2, variance_var2, lbp2, ltp2, lpq2, hog2 = calculate_features(image_path2)

    # Calculate Euclidean distances
    mean_distance = np.linalg.norm(np.array(mean_var1) - np.array(mean_var2))
    variance_distance = np.linalg.norm(np.array(variance_var1) - np.array(variance_var2))
    lbp_distance = np.linalg.norm(lbp1 - lbp2)
    ltp_distance = np.linalg.norm(ltp1 - ltp2)
    lpq_distance = np.linalg.norm(lpq1 - lpq2)
    hog_distance = np.linalg.norm(hog1 - hog2)

    # Calculate correlations
    mean_corr = np.corrcoef(mean_var1, mean_var2)[0, 1]
    variance_corr = np.corrcoef(variance_var1, variance_var2)[0, 1]
    lbp_corr = np.corrcoef(lbp1.flatten(), lbp2.flatten())[0, 1]
    ltp_corr = np.corrcoef(ltp1.flatten(), ltp2.flatten())[0, 1]
    lpq_corr = np.corrcoef(lpq1.flatten(), lpq2.flatten())[0, 1]
    hog_corr = np.corrcoef(hog1, hog2)[0, 1]

    # Print the distances and correlations
    print(Fore.GREEN)
    print("Euclidean Distance between Mean Vectors:", mean_distance)
    print("Correlation between Mean Vectors:", mean_corr)

    print(Fore.WHITE)
    print("Euclidean Distance between Variance Vectors:", variance_distance)
    print("Correlation between Variance Vectors:", variance_corr)

    print(Fore.CYAN)
    print("Euclidean Distance between LBP Vectors:", lbp_distance)
    print("Correlation between LBP Vectors:", lbp_corr)

    print(Fore.YELLOW)
    print("Euclidean Distance between LTP Vectors:", ltp_distance)
    print("Correlation between LTP Vectors:", ltp_corr)

    print(Fore.BLUE)
    print("Euclidean Distance between LPQ Vectors:", lpq_distance)
    print("Correlation between LPQ Vectors:", lpq_corr)

    print(Fore.RED)
    print("Euclidean Distance between HOG Vectors:", hog_distance)
    print("Correlation between HOG Vectors:", hog_corr)

# Example usage
image_path1 = 'image1.jpg'
image_path2 = 'image2.jpg'
image_path3 = 'image3.jpg'
image_path4 = 'image4.jpg'
compare_images(image_path1, image_path1)
compare_images(image_path1, image_path2)
compare_images(image_path1, image_path3)
compare_images(image_path1, image_path4)
compare_images(image_path2, image_path2)
compare_images(image_path2, image_path3)
compare_images(image_path2, image_path4)
compare_images(image_path3, image_path3)
compare_images(image_path3, image_path4)
compare_images(image_path4, image_path4)
