
# Convex Optimization for Edge-Preserving Image Denoising- CSE 203b

This project implements an edge-preserving image denoising algorithm based on convex optimization using the Accelerated Proximal Gradient (APG) method. The algorithm effectively removes noise while preserving important edges and structures in images.
<img width="1312" alt="image" src="https://github.com/user-attachments/assets/7ab1aa1e-2c10-475d-9871-8de8d1b69659" />



# 📝 Algorithm Features

- **Edge Preservation:** Unlike simple blurring filters, this algorithm identifies and preserves important edges in images
- **Adaptive Processing:** Automatically adjusts denoising strength based on local image structures
- **Mathematical Robustness:** Based on convex optimization theory with good convergence and stability properties
- **Simple Parameters:** Only a few parameters need to be adjusted to achieve good result


# 🚀  Quick Start
Install Dependencies
```python
pip install numpy scipy matplotlib scikit-image tqdm
```

Basic Usage
```python
from denoising import denoise_apg_edge_preserving_fixed, generate_noisy_image

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
```

# 📈 Comparison with Other Methods
The algorithm has been compared with standard denoising methods including Total Variation (TV), Bilateral Filtering, and Wavelet Denoising. Our method provides better edge preservation while achieving competitive PSNR and SSIM metrics.
