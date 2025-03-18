# Image_denoising
Convex Optimization for Edge-Preserving Image Denoising- CSE 203b


This project implements an edge-preserving image denoising algorithm based on convex optimization using the Accelerated Proximal Gradient (APG) method. The algorithm effectively removes noise while preserving important edges and structures in images.


Algorithm Features

Edge Preservation: Unlike simple blurring filters, this algorithm identifies and preserves important edges in images
Adaptive Processing: Automatically adjusts denoising strength based on local image structures
Mathematical Robustness: Based on convex optimization theory with good convergence and stability properties
Simple Parameters: Only a few parameters need to be adjusted to achieve good result


🚀 Quick Start
Install Dependencies
bashCopypip install numpy scipy matplotlib scikit-image tqdm
Basic Usage
pythonCopyfrom denoising import denoise_apg_edge_preserving_fixed, generate_noisy_image

# Read image
img = imread('your_image.png', as_gray=True)

# Add noise (if testing is needed)
noisy_img = generate_noisy_image(img, noise_level=25)

# Denoise the image
denoised_img = denoise_apg_edge_preserving_fixed(
    noisy_img, 
    lambda_param=0.02,  # Adjust denoising strength
    max_iter=50         # Number of iterations
)

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(noisy_img, cmap='gray'), plt.title('Noisy Image')
plt.subplot(133), plt.imshow(denoised_img, cmap='gray'), plt.title('Denoised Result')
plt.tight_layout()
plt.show()