import numpy as np
import scipy.fftpack as fft
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, exposure, util
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import urllib.request
import glob
from tqdm import tqdm
import time


def gradient(img):
    """计算图像梯度，使用有限差分法"""
    # 计算水平和垂直梯度
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)
    
    # 前向差分，并处理边界
    grad_x[:, :-1] = img[:, 1:] - img[:, :-1]
    grad_y[:-1, :] = img[1:, :] - img[:-1, :]
    
    return grad_x, grad_y


def divergence(grad_x, grad_y):
    """计算向量场的散度（梯度的反向操作）"""
    # 初始化散度数组
    div = np.zeros_like(grad_x)
    
    # 使用后向差分计算散度（前向梯度的伴随）
    div[:, 1:] -= grad_x[:, :-1]
    div[:, 0] -= grad_x[:, 0]
    
    div[1:, :] -= grad_y[:-1, :]
    div[0, :] -= grad_y[0, :]
    
    return div


def compute_weights(noisy_img, alpha=0.1, sigma=15.0):
    """计算基于局部边缘信息的自适应权重"""
    # 从噪声图像估计梯度
    grad_x, grad_y = gradient(noisy_img)
    
    # 计算梯度幅值
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 计算权重：边缘区域（大梯度）的权重较小
    # 使用梯度幅值的高斯型函数
    weights = np.exp(-alpha * (grad_mag**2) / (sigma**2))
    
    return weights


def proximal_l1(v, threshold):
    """l1范数的近端算子（软阈值）"""
    magnitude = np.sqrt(np.sum(v**2, axis=0))
    # 避免除以零
    mask = magnitude > threshold
    
    result = np.zeros_like(v)
    if np.any(mask):
        factor = np.maximum(0, 1 - threshold / magnitude[mask])
        for i in range(v.shape[0]):
            result[i][mask] = v[i][mask] * factor
            
    return result


def generate_noisy_image(img, noise_level=15):
    """给图像添加高斯噪声"""
    sigma = noise_level / 255.0
    noisy = img + np.random.normal(0, sigma, img.shape)
    # 裁剪到有效范围
    noisy = np.clip(noisy, 0, 1)
    return noisy


def denoise_apg_edge_preserving(noisy_img, lambda_param=0.03, max_iter=100, tol=1e-4, epsilon=1e-6):
    """
    使用加速近端梯度法进行边缘保留图像去噪
    
    参数:
    -----------
    noisy_img : 2D numpy array
        输入的噪声图像（归一化到 [0, 1]）
    lambda_param : float
        控制去噪强度的正则化参数
    max_iter : int
        最大迭代次数
    tol : float
        收敛容差
    epsilon : float
        l1范数平滑近似的小常数
        
    返回:
    --------
    denoised_img : 2D numpy array
        去噪后的输出图像
    """
    # 初始化变量
    x = noisy_img.copy()
    y = x.copy()
    t = 1.0
    prev_x = x.copy()
    
    # 计算自适应权重
    weights = compute_weights(noisy_img)
    
    # 使用FFT实现快速求解线性系统
    dx_filter = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    dy_filter = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    
    # 滤波器的FFT，用于高效卷积
    dx_fft = fft.fft2(dx_filter, shape=noisy_img.shape)
    dy_fft = fft.fft2(dy_filter, shape=noisy_img.shape)
    dxT_fft = np.conj(dx_fft)
    dyT_fft = np.conj(dy_fft)
    
    # 预计算线性系统的项
    weights_term = 1 + lambda_param * weights**2
    
    # 主优化循环
    for iter_num in range(max_iter):
        # 保存前一个迭代结果
        prev_x = x.copy()
        
        # 计算y的梯度
        y_dx = ndimage.convolve(y, dx_filter, mode='wrap')
        y_dy = ndimage.convolve(y, dy_filter, mode='wrap')
        
        # 计算梯度幅值并归一化
        grad_mag = np.sqrt(y_dx**2 + y_dy**2 + epsilon**2)
        
        # 计算边缘保留项的近端映射
        # 这是近端算子的向量化版本
        prox_dx = y_dx * np.maximum(0, 1 - lambda_param / grad_mag)
        prox_dy = y_dy * np.maximum(0, 1 - lambda_param / grad_mag)
        
        # 计算数据拟合梯度项: y - noisy_img + divergence(prox)
        div_prox = divergence(prox_dx, prox_dy)
        
        # 计算自适应空间变化数据项
        data_term = y - noisy_img + lambda_param * weights**2 * (y - noisy_img)
        
        # 合并项
        gradient_term = data_term + div_prox
        
        # 使用FFT更新x以高效求解线性系统
        x_fft = fft.fft2(noisy_img - gradient_term) / weights_term
        x = np.real(fft.ifft2(x_fft))
        
        # 加速更新
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y = x + ((t - 1) / t_new) * (x - prev_x)
        t = t_new
        
        # 检查收敛性
        rel_change = np.linalg.norm(x - prev_x) / np.linalg.norm(x)
        if rel_change < tol:
            print(f"在第 {iter_num+1} 次迭代收敛")
            break
            
    return x


def evaluate_denoising(original, noisy, denoised):
    """使用PSNR和SSIM指标评估去噪质量"""
    psnr_noisy = psnr(original, noisy)
    psnr_denoised = psnr(original, denoised)
    
    ssim_noisy = ssim(original, noisy, data_range=1.0)
    ssim_denoised = ssim(original, denoised, data_range=1.0)
    
    print(f"PSNR - 有噪声: {psnr_noisy:.2f} dB, 去噪后: {psnr_denoised:.2f} dB")
    print(f"SSIM - 有噪声: {ssim_noisy:.4f}, 去噪后: {ssim_denoised:.4f}")
    
    return {
        'psnr_improvement': psnr_denoised - psnr_noisy,
        'ssim_improvement': ssim_denoised - ssim_noisy
    }


def display_results(original, noisy, denoised, title="边缘保留去噪结果"):
    """并排显示原始、有噪声和去噪后的图像"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title('有噪声图像')
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title('去噪后图像')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


def edge_highlight(img):
    """高亮显示图像中的边缘，用于可视化"""
    grad_x, grad_y = gradient(img)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    return exposure.rescale_intensity(edges)


def compare_edge_preservation(original, noisy, denoised, title="边缘保留对比"):
    """比较原始、有噪声和去噪后图像的边缘保留情况"""
    original_edges = edge_highlight(original)
    noisy_edges = edge_highlight(noisy)
    denoised_edges = edge_highlight(denoised)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_edges, cmap='viridis')
    axes[0].set_title('原始图像边缘')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_edges, cmap='viridis')
    axes[1].set_title('有噪声图像边缘')
    axes[1].axis('off')
    
    axes[2].imshow(denoised_edges, cmap='viridis')
    axes[2].set_title('去噪后图像边缘')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


def download_set12(base_dir="./datasets"):
    """下载Set12数据集"""
    os.makedirs(base_dir, exist_ok=True)
    dataset_dir = os.path.join(base_dir, "set12")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Set12图像URL
    set12_urls = [
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/01.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/02.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/03.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/04.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/05.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/06.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/07.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/08.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/09.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/10.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/11.png",
        "https://github.com/cszn/DnCNN/raw/master/testsets/Set12/12.png"
    ]
    
    for i, url in enumerate(set12_urls):
        file_path = os.path.join(dataset_dir, f"{i+1:02d}.png")
        
        if not os.path.exists(file_path):
            print(f"下载Set12图像 {i+1}...")
            try:
                urllib.request.urlretrieve(url, file_path)
            except:
                print(f"警告: 无法下载图像 {i+1}，请检查网络连接或URL是否有效")
        else:
            print(f"Set12图像 {i+1} 已存在")
    
    image_paths = glob.glob(os.path.join(dataset_dir, "*.png"))
    return image_paths


def create_synthetic_test_images(num_images=5, base_dir="./datasets/synthetic"):
    """创建合成测试图像，包含边缘和纹理"""
    os.makedirs(base_dir, exist_ok=True)
    image_paths = []
    
    for i in range(num_images):
        # 创建尺寸为256x256的图像
        img = np.zeros((256, 256))
        
        # 添加不同的测试图案
        if i == 0:
            # 创建方块图案
            img[64:192, 64:192] = 1.0
        elif i == 1:
            # 创建梯度图案
            x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
            img = x
        elif i == 2:
            # 创建圆形图案
            x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
            img = (x**2 + y**2 < 0.5**2).astype(float)
        elif i == 3:
            # 创建十字图案
            img[118:138, :] = 1.0
            img[:, 118:138] = 1.0
        elif i == 4:
            # 创建棋盘图案
            x, y = np.meshgrid(np.arange(256), np.arange(256))
            img = ((x // 32) % 2 != (y // 32) % 2).astype(float)
        
        # 保存图像
        file_path = os.path.join(base_dir, f"synthetic_{i+1}.png")
        plt.imsave(file_path, img, cmap='gray')
        image_paths.append(file_path)
    
    return image_paths


def run_denoising_on_dataset(image_paths, noise_levels=[15, 25, 50], save_dir="./results"):
    """对数据集中的所有图像运行去噪算法"""
    os.makedirs(save_dir, exist_ok=True)
    
    results = {noise_level: {'psnr': [], 'ssim': [], 'time': []} for noise_level in noise_levels}
    
    # 对每个噪声级别进行处理
    for noise_level in noise_levels:
        print(f"\n处理噪声级别: {noise_level}")
        noise_dir = os.path.join(save_dir, f"noise_{noise_level}")
        os.makedirs(noise_dir, exist_ok=True)
        
        # 处理每个图像
        for img_path in tqdm(image_paths, desc="处理图像"):
            # 读取图像
            img_name = os.path.basename(img_path)
            img = img_as_float(io.imread(img_path, as_gray=True))
            
            # 添加噪声
            noisy_img = generate_noisy_image(img, noise_level)
            
            # 去噪处理
            start_time = time.time()
            denoised_img = denoise_apg_edge_preserving(
                noisy_img, 
                lambda_param=0.03 * (noise_level/15),  # 根据噪声强度调整参数
                max_iter=100,
                tol=1e-4
            )
            end_time = time.time()
            
            # 计算指标
            psnr_val = psnr(img, denoised_img)
            ssim_val = ssim(img, denoised_img, data_range=1.0)
            elapsed_time = end_time - start_time
            
            # 保存结果
            results[noise_level]['psnr'].append(psnr_val)
            results[noise_level]['ssim'].append(ssim_val)
            results[noise_level]['time'].append(elapsed_time)
            
            # 保存图像
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title('原始图像')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(noisy_img, cmap='gray')
            plt.title(f'噪声图像 (噪声: {noise_level})')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(denoised_img, cmap='gray')
            plt.title(f'去噪图像 (PSNR: {psnr_val:.2f}dB)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(noise_dir, f"result_{img_name}"))
            plt.close()
    
    # 打印平均结果
    print("\n========== 结果摘要 ==========")
    for noise_level in noise_levels:
        avg_psnr = np.mean(results[noise_level]['psnr'])
        avg_ssim = np.mean(results[noise_level]['ssim'])
        avg_time = np.mean(results[noise_level]['time'])
        
        print(f"噪声级别: {noise_level}")
        print(f"  平均 PSNR: {avg_psnr:.2f} dB")
        print(f"  平均 SSIM: {avg_ssim:.4f}")
        print(f"  平均处理时间: {avg_time:.2f} 秒")
    
    return results


def compare_with_other_denoisers(img, noise_level=25):
    """将我们的方法与其他去噪方法进行比较"""
    try:
        from skimage.restoration import denoise_tv_bregman, denoise_bilateral, denoise_wavelet, estimate_sigma
        
        # 添加噪声
        noisy_img = generate_noisy_image(img, noise_level)
        
        # 应用不同的去噪方法
        print("应用我们的边缘保留去噪方法...")
        start_time = time.time()
        our_denoised = denoise_apg_edge_preserving(noisy_img, lambda_param=0.03 * (noise_level/15), max_iter=100)
        our_time = time.time() - start_time
        
        print("应用总变差去噪...")
        start_time = time.time()
        tv_denoised = denoise_tv_bregman(noisy_img, weight=0.1)
        tv_time = time.time() - start_time
        
        print("应用双边滤波去噪...")
        start_time = time.time()
        bilateral_denoised = denoise_bilateral(noisy_img, sigma_color=0.1, sigma_spatial=1)
        bilateral_time = time.time() - start_time
        
        print("应用小波去噪...")
        start_time = time.time()
        wavelet_denoised = denoise_wavelet(noisy_img, multichannel=False)
        wavelet_time = time.time() - start_time
        
        # 计算PSNR和SSIM
        metrics = {
            "有噪声图像": (psnr(img, noisy_img), ssim(img, noisy_img, data_range=1.0), 0),
            "我们的方法": (psnr(img, our_denoised), ssim(img, our_denoised, data_range=1.0), our_time),
            "总变差": (psnr(img, tv_denoised), ssim(img, tv_denoised, data_range=1.0), tv_time),
            "双边滤波": (psnr(img, bilateral_denoised), ssim(img, bilateral_denoised, data_range=1.0), bilateral_time),
            "小波": (psnr(img, wavelet_denoised), ssim(img, wavelet_denoised, data_range=1.0), wavelet_time)
        }
        
        # 显示结果
        plt.figure(figsize=(20, 10))
        images = [img, noisy_img, our_denoised, tv_denoised, bilateral_denoised, wavelet_denoised]
        titles = list(metrics.keys())
        titles.insert(0, "原始图像")
        
        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(2, 3, i+1)
            plt.imshow(image, cmap='gray')
            if i > 0:  # 跳过原始图像
                psnr_val, ssim_val, t = metrics[title]
                plt.title(f"{title}\nPSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f}\n时间: {t:.2f}s")
            else:
                plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("denoisers_comparison.png")
        plt.show()
        
        # 显示边缘保留情况
        plt.figure(figsize=(20, 10))
        edge_images = [edge_highlight(img) for img in images]
        
        for i, (image, title) in enumerate(zip(edge_images, titles)):
            plt.subplot(2, 3, i+1)
            plt.imshow(image, cmap='viridis')
            plt.title(f"{title} - 边缘")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("edge_preservation_comparison.png")
        plt.show()
        
        # 打印比较结果表格
        print("\n========== 去噪方法比较 ==========")
        print(f"{'方法':<15} {'PSNR (dB)':<12} {'SSIM':<10} {'时间 (秒)':<10}")
        print("-" * 47)
        for method, (psnr_val, ssim_val, t) in metrics.items():
            print(f"{method:<15} {psnr_val:<12.2f} {ssim_val:<10.4f} {t:<10.2f}")
        
        return metrics
        
    except ImportError:
        print("警告: 无法导入某些scikit-image去噪函数。请确保已安装最新版本的scikit-image。")
        return None


def main():
    """主函数，提供多种使用选项"""
    
    # 创建结果目录
    if not os.path.exists("./results"):
        os.makedirs("./results")
    
    print("边缘保留图像去噪演示程序")
    print("=" * 50)
    print("请选择操作模式:")
    print("1. 对单个图像进行去噪演示")
    print("2. 下载Set12数据集并对其进行测试")
    print("3. 创建合成测试图像并进行测试")
    print("4. 比较不同去噪方法的性能")
    
    choice = input("请输入选择 (1-4): ")
    
    if choice == '1':
        # 单图像去噪演示
        synthetic = np.zeros((256, 256))
        
        # 创建测试图像（有清晰边缘的图像）
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        # 添加圆形和方块
        synthetic = (x**2 + y**2 < 0.5**2).astype(float)
        synthetic[x > 0.1] += 0.5
        synthetic = np.clip(synthetic, 0, 1)
        
        # 添加噪声
        noise_level = 25  # 可以调整 (0-50)
        noisy_img = generate_noisy_image(synthetic, noise_level)
        
        # 去噪处理
        lambda_param = 0.03 * (noise_level/15)  # 根据噪声级别调整参数
        print(f"正在对图像进行去噪处理 (噪声级别: {noise_level}, lambda: {lambda_param:.4f})...")
        denoised_img = denoise_apg_edge_preserving(
            noisy_img, 
            lambda_param=lambda_param,
            max_iter=100
        )
        
        # 显示结果
        evaluate_denoising(synthetic, noisy_img, denoised_img)
        display_results(synthetic, noisy_img, denoised_img, "边缘保留去噪演示")
        compare_edge_preservation(synthetic, noisy_img, denoised_img)
        
    elif choice == '2':
        # 下载并测试Set12数据集
        print("下载Set12数据集...")
        image_paths = download_set12()
        
        if len(image_paths) > 0:
            print(f"成功下载 {len(image_paths)} 张图像")
            noise_levels = [15, 25, 35]
            print(f"测试噪声级别: {noise_levels}")
            
            results = run_denoising_on_dataset(image_paths, noise_levels)
        else:
            print("无法下载数据集。请检查您的互联网连接，或尝试其他选项。")
        
    elif choice == '3':
        # 创建并测试合成图像
        print("创建合成测试图像...")
        image_paths = create_synthetic_test_images(5)
        
        print(f"创建了 {len(image_paths)} 张合成图像")
        noise_levels = [15, 25, 35]
        print(f"测试噪声级别: {noise_levels}")
        
        results = run_denoising_on_dataset(image_paths, noise_levels)
        
    elif choice == '4':
        # 比较不同去噪方法
        print("创建测试图像用于比较...")
        
        # 创建一个包含不同特征的测试图像
        img = np.zeros((256, 256))
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        
        # 添加圆形
        img += 0.7 * (x**2 + y**2 < 0.5**2).astype(float)
        
        # 添加线条
        img[100:110, :] += 0.8
        img[:, 180:190] += 0.8
        
        # 添加渐变
        img += 0.2 * np.exp(-((x-0.5)**2 + (y+0.5)**2) / 0.1)
        
        img = np.clip(img, 0, 1)
        
        noise_level = 25
        metrics = compare_with_other_denoisers(img, noise_level)
        
    else:
        print("无效选择，请运行程序并选择1-4之间的数字。")
    
    print("\n程序完成！结果保存在 './results' 目录中。")


if __name__ == "__main__":
    main()