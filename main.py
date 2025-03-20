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
from skimage.metrics import mean_squared_error as mse



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


def compute_weights(noisy_img, alpha=0.1, sigma=15.0, patch_size=3):
    """结合局部梯度与非局部相似性的权重"""
    # 原梯度计算
    grad_x, grad_y = gradient(noisy_img)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 非局部相似性权重
    pad_width = patch_size // 2
    padded_img = np.pad(noisy_img, pad_width, mode='reflect')
    nl_weights = np.zeros_like(noisy_img)

    for i in range(noisy_img.shape[0]):
        for j in range(noisy_img.shape[1]):
            patch = padded_img[i:i + patch_size, j:j + patch_size]
            distances = ndimage.gaussian_filter((padded_img - patch) ** 2, sigma=1.0)
            nl_weights[i, j] = np.exp(-np.mean(distances) / (sigma ** 2))

    # 组合权重
    total_weights = 0.7 * np.exp(-alpha * (grad_mag ** 2) / (sigma ** 2)) + 0.3 * nl_weights
    return total_weights


def generate_noisy_image(img, noise_level=15):
    """给图像添加高斯噪声"""
    sigma = noise_level / 255.0
    noisy = img + np.random.normal(0, sigma, img.shape)
    # 裁剪到有效范围
    noisy = np.clip(noisy, 0, 1)
    return noisy


def denoise_apg_edge_preserving(noisy_img, lambda_param=0.03, max_iter=100, tol=1e-4):
    # 初始化变量
    x = noisy_img.copy()
    y = x.copy()
    t = 1.0
    prev_x = x.copy()
    prev_grad = None  # 新增：存储前一次梯度
    step_size = 0.2  # 初始步长（后续会动态更新）

    # 新增：Barzilai-Borwein步长计算函数
    def compute_bb_step(x_prev, x_current, grad_prev, grad_current):
        s = x_current - x_prev
        y_grad = grad_current - grad_prev
        # 两种BB步长计算方式，选择更稳定的版本
        numerator = np.sum(s * s)
        denominator = np.sum(s * y_grad)
        if denominator == 0:
            return step_size  # 避免除零
        return np.abs(numerator / denominator)

    # 主优化循环
    for iter_num in range(max_iter):
        prev_x = x.copy()

        # 计算当前y的梯度
        grad_x, grad_y = gradient(y)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-10)

        # 软阈值处理（Total Variation）
        scale = np.maximum(0, 1 - lambda_param / (grad_mag + 1e-10))
        prox_grad_x = grad_x * scale
        prox_grad_y = grad_y * scale

        # 计算散度项
        div_term = divergence(prox_grad_x, prox_grad_y)

        # 数据保真项梯度
        data_term = y - noisy_img

        # 动态步长调整（从第二次迭代开始）
        if iter_num > 0:
            current_grad = data_term - div_term  # 当前梯度方向
            step_size = compute_bb_step(prev_x, x, prev_grad, current_grad)
            step_size = np.clip(step_size, 1e-3, 0.5)  # 限制步长范围
        prev_grad = data_term - div_term  # 保存当前梯度用于下次计算

        # 更新x
        x = y - step_size * (data_term - div_term)
        x = np.clip(x, 0, 1)  # 像素值裁剪

        # FISTA加速
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = x + ((t - 1) / t_new) * (x - prev_x)
        t = t_new

        # 收敛性检查
        rel_change = np.linalg.norm(x - prev_x) / (np.linalg.norm(x) + 1e-10)
        if rel_change < tol:
            print(f"Converged at iteration {iter_num + 1}")
            break

    # 在返回前添加锐化
    denoised_img = x.copy()
    denoised_img = np.clip(denoised_img + 0.5 * (denoised_img - ndimage.gaussian_filter(denoised_img, sigma=1)), 0,
                               1)

    return x


def evaluate_denoising(original, noisy, denoised):
    """评估去噪质量，新增多种量化指标"""
    # 原有PSNR和SSIM
    psnr_noisy = psnr(original, noisy)
    psnr_denoised = psnr(original, denoised)
    ssim_noisy = ssim(original, noisy, data_range=1.0)
    ssim_denoised = ssim(original, denoised, data_range=1.0)

    # 新增指标 --------------------------------------------------
    # 1. 均方误差 (MSE)
    mse_noisy = np.mean((original - noisy) ** 2)
    mse_denoised = np.mean((original - denoised) ** 2)

    # 2. 多尺度SSIM (需安装 pip install ms-ssim)
    try:
        from ms_ssim import MultiScaleSSIM
        ms_ssim_noisy = MultiScaleSSIM(original, noisy)
        ms_ssim_denoised = MultiScaleSSIM(original, denoised)
    except ImportError:
        ms_ssim_noisy = ms_ssim_denoised = None

    # 3. 边缘保留指数 (EPI)
    def edge_preservation_index(orig, denoised):
        grad_orig = np.sqrt(gradient(orig)[0] ** 2 + gradient(orig)[1] ** 2)
        grad_denoised = np.sqrt(gradient(denoised)[0] ** 2 + gradient(denoised)[1] ** 2)
        return np.corrcoef(grad_orig.flatten(), grad_denoised.flatten())[0, 1]

    epi = edge_preservation_index(original, denoised)

    # 4. 噪声减少率 (NRR)
    nrr = 1 - (np.std(denoised - original) / np.std(noisy - original))

    # 输出结果
    print("\n========== 量化指标 ==========")
    print(f"PSNR - 有噪声: {psnr_noisy:.2f} dB, 去噪后: {psnr_denoised:.2f} dB")
    print(f"SSIM - 有噪声: {ssim_noisy:.4f}, 去噪后: {ssim_denoised:.4f}")
    print(f"MSE  - 有噪声: {mse_noisy:.4f}, 去噪后: {mse_denoised:.4f}")
    if ms_ssim_noisy is not None:
        print(f"MS-SSIM - 有噪声: {ms_ssim_noisy:.4f}, 去噪后: {ms_ssim_denoised:.4f}")
    print(f"边缘保留指数 (EPI): {epi:.4f}")
    print(f"噪声减少率 (NRR): {nrr:.4f}")

    return {
        'psnr_improvement': psnr_denoised - psnr_noisy,
        'ssim_improvement': ssim_denoised - ssim_noisy,
        'mse_improvement': mse_noisy - mse_denoised,
        'ms_ssim': ms_ssim_denoised if ms_ssim_denoised else None,
        'epi': epi,
        'nrr': nrr
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


def run_denoising_on_dataset(image_paths, noise_levels=[15, 25, 50], save_dir="./results"):
    """对数据集中的所有图像运行去噪算法"""
    os.makedirs(save_dir, exist_ok=True)

    results = {noise_level: {'psnr': [], 'ssim': [], 'mse': [], 'isnr': [], 'time': []}
                for noise_level in noise_levels}

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

            # 确定lambda参数，根据噪声级别自适应调整
            lambda_val = 0.01 * (noise_level / 15)

            # 去噪处理
            start_time = time.time()
            denoised_img = denoise_apg_edge_preserving(
                noisy_img,
                lambda_param=lambda_val,
                max_iter=50
            )
            end_time = time.time()

            # 计算指标
            psnr_val = psnr(img, denoised_img)
            ssim_val = ssim(img, denoised_img, data_range=1.0)
            mse_val = mse(img, denoised_img)
            isnr_val = 10 * np.log10(mse(img, noisy_img) / mse_val)  # ISNR计算
            elapsed_time = end_time - start_time

            # 保存结果
            results[noise_level]['psnr'].append(psnr_val)
            results[noise_level]['ssim'].append(ssim_val)
            results[noise_level]['mse'].append(mse_val)
            results[noise_level]['isnr'].append(isnr_val)
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
        avg_mse = np.mean(results[noise_level]['mse'])
        avg_isnr = np.mean(results[noise_level]['isnr'])
        avg_time = np.mean(results[noise_level]['time'])

        print(f"噪声级别: {noise_level}")
        print(f"  平均 PSNR: {avg_psnr:.2f} dB")
        print(f"  平均 SSIM: {avg_ssim:.4f}")
        print(f"  平均 MSE: {avg_mse:.4f}")
        print(f"  平均 ISNR: {avg_isnr:.2f} dB")
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
        our_denoised = denoise_apg_edge_preserving(
            noisy_img, 
            lambda_param=0.01 * (noise_level/15), 
            max_iter=50)
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
    print("3. 输入自定义图像路径进行去噪")
    print("4. 比较不同去噪方法的性能")
    
    choice = input("请输入选择 (1-4): ")
    
    if choice == '1':
        # 单图像去噪演示
        print("创建测试图像...")
        
        # 创建测试图像（有清晰边缘的图像）
        img = np.zeros((256, 256))
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        
        # 添加圆形
        img += 0.7 * (x**2 + y**2 < 0.5**2).astype(float)
        
        # 添加方块
        img[(x > -0.2) & (x < 0.2) & (y > -0.2) & (y < 0.2)] = 0.9
        
        # 添加线条
        img[120:130, :] += 0.5
        img[:, 180:190] += 0.5
        
        img = np.clip(img, 0, 1)
        
        # 添加噪声
        noise_level = int(input("请输入噪声级别 (推荐15-50): ") or "25")
        noisy_img = generate_noisy_image(img, noise_level)
        
        # 去噪处理
        lambda_param = 0.01 * (noise_level/15)  # 根据噪声级别自适应调整
        print(f"正在对图像进行去噪处理 (噪声级别: {noise_level}, lambda: {lambda_param:.4f})...")
        denoised_img = denoise_apg_edge_preserving(
            noisy_img, 
            lambda_param=lambda_param,
            max_iter=50
        )
        
        # 显示结果
        evaluate_denoising(img, noisy_img, denoised_img)
        display_results(img, noisy_img, denoised_img, "边缘保留去噪演示")
        compare_edge_preservation(img, noisy_img, denoised_img)
        
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
        # 使用自定义图像
        image_path = input("请输入图像文件路径: ")
        
        try:
            img = img_as_float(io.imread(image_path, as_gray=True))
            
            # 添加噪声
            noise_level = int(input("请输入噪声级别 (推荐15-50): ") or "25")
            noisy_img = generate_noisy_image(img, noise_level)
            
            # 去噪处理
            lambda_param = 0.01 * (noise_level/15)  # 根据噪声级别自适应调整
            print(f"正在对图像进行去噪处理 (噪声级别: {noise_level}, lambda: {lambda_param:.4f})...")
            denoised_img = denoise_apg_edge_preserving(
                noisy_img, 
                lambda_param=lambda_param,
                max_iter=50
            )
            
            # 显示结果
            evaluate_denoising(img, noisy_img, denoised_img)
            display_results(img, noisy_img, denoised_img, "自定义图像去噪结果")
            compare_edge_preservation(img, noisy_img, denoised_img)
            
        except Exception as e:
            print(f"错误: 无法读取或处理图像。详细信息: {e}")
        
    elif choice == '4':
        # 比较不同去噪方法
        print("创建测试图像用于比较...")
        
        # 创建一个包含不同特征的测试图像
        img = np.zeros((256, 256))
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        
        # 添加圆形
        img += 0.7 * (x**2 + y**2 < 0.5**2).astype(float)
        
        # 添加方块
        img[(x > 0.1) & (x < 0.6) & (y > -0.7) & (y < -0.2)] = 0.9
        
        # 添加线条
        img[100:110, :] += 0.8
        img[:, 180:190] += 0.8
        
        # 添加渐变
        img += 0.2 * np.exp(-((x-0.5)**2 + (y+0.5)**2) / 0.1)
        
        img = np.clip(img, 0, 1)
        
        noise_level = int(input("请输入噪声级别 (推荐15-50): ") or "25")
        metrics = compare_with_other_denoisers(img, noise_level)
        
    else:
        print("无效选择，请运行程序并选择1-4之间的数字。")
    
    print("\n程序完成！结果保存在 './results' 目录中。")


if __name__ == "__main__":
    main()