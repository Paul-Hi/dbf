import colorsys
from torch import tensor, cuda, nn, optim, autograd
import pydbf
import numpy as np
import time
import requests
from io import BytesIO
import torch
from scipy.stats import norm
from torch.nn.functional import conv2d
import math

print("DBF version:", pydbf.__version__)
print("DBF author:", pydbf.__author__)
print("DBF license:", pydbf.__license__)
print("DBF doc:", pydbf.__doc__)

# Set to True to display images using matplotlib, False to save them using imageio
SHOW_ONLINE = True


def rl_filter_test():
    import imageio.v3 as iio
    img_url = "https://instasize.com/content/film-grain-woman-facing-beach-ocean.jpeg"
    response = requests.get(img_url)
    img = iio.imread(BytesIO(response.content))

    img = (img.astype(np.float32) / 255.0)

    img = tensor(img).permute(
        2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
    img = img.to('cuda')
    print("Image shape:", img.shape)

    spatial_sigma = tensor(
        [19.0, 19.0], device='cuda')  # (sigmaX, sigmaY)
    range_sigma = tensor([0.05], device='cuda')         # (sigmaR)
    n = 1
    filtered_img = img.clone()
    start_time = time.time()
    for i in range(n):
        filtered_img = pydbf.bilateral_filter_cuda(
            filtered_img.contiguous(), spatial_sigma, range_sigma)
    vram_allocated = cuda.memory_allocated() / (1024 ** 2)
    vram_reserved = cuda.memory_reserved() / (1024 ** 2)
    print(
        f"VRAM allocated: {vram_allocated:.2f} MB, reserved: {vram_reserved:.2f} MB")
    end_time = time.time()
    print(
        f"Bilateral filter applied {n} times in {end_time - start_time:.4f} seconds")
    filtered_img = filtered_img.clamp(0.0, 1.0).cpu().squeeze(
        0).permute(1, 2, 0).numpy()  # (H, W, C)

    if SHOW_ONLINE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Filtered Image")
        plt.imshow(filtered_img)
        plt.axis('off')
        plt.show()
    else:
        import imageio.v3 as iio
        iio.imwrite("example_filtered.png",
                    (filtered_img * 255).astype(np.uint8))


def sigma_optimization_test():
    import imageio.v3 as iio
    img_url = "https://images.unsplash.com/photo-1474511320723-9a56873867b5?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=2072"
    response = requests.get(img_url)
    img = iio.imread(BytesIO(response.content))
    img = (img.astype(np.float32) / 255.0)
    print("Original image shape:", img.shape)

    # Add film grain-like noise (Gaussian noise per channel)
    img_rgb = img
    noise_std = 0.075
    grain_noise = np.random.normal(0, noise_std, img_rgb.shape).astype(np.float32)
    noisy_img = np.clip(img_rgb + grain_noise, 0.0, 1.0)

    noisy_img_tensor = tensor(noisy_img).permute(
        2, 0, 1).unsqueeze(0).float().to('cuda')  # (1, C, H, W)
    img_tensor = tensor(img).permute(
        2, 0, 1).unsqueeze(0).float().to('cuda')

    spatial_sigma = nn.Parameter(tensor([1.0, 1.0], device='cuda'))
    range_sigma = nn.Parameter(tensor([0.5], device='cuda'))

    mse_loss = nn.MSELoss()

    def loss_fn(filtered, target):
        return mse_loss(filtered, target)

    optimizer = optim.Adam([
        {'params': [spatial_sigma], 'lr': 0.01},
        {'params': [range_sigma], 'lr': 0.01}
    ])

    max_iters = 2**11
    tol = 1e-8
    patience = 512
    plateau_count = 0

    start_time = time.time()

    prev_loss = None
    for i in range(max_iters):
        optimizer.zero_grad()

        filtered_img = pydbf.bilateral_filter_cuda(
            noisy_img_tensor.contiguous(), spatial_sigma, range_sigma)
        loss = loss_fn(filtered_img, img_tensor)
        loss.backward()
        optimizer.step()

        spatial_sigma.data.clamp_(min=1e-6)
        range_sigma.data.clamp_(min=1e-6)

        if prev_loss is not None and abs(prev_loss - loss.item()) < 1e-9:
            plateau_count += 1
            if plateau_count >= patience:
                print(
                    f"Early stopping: loss plateau for {patience} iterations at iter {i+1}")
                break
        else:
            plateau_count = 0
        prev_loss = loss.item()

        if i % 10 == 0 or loss.item() < tol or i == max_iters - 1:
            print(f"Iter {i+1}/{max_iters}, Loss: {loss.item():.6e}, spatial_sigma: {spatial_sigma.data.cpu().numpy()}, range_sigma: {range_sigma.data.cpu().numpy()}")
        if loss.item() < tol:
            print("Early stopping due to tolerance.")
            break

    end_time = time.time()
    print(
        f"Optimization finished in {end_time - start_time:.2f} seconds.")

    import matplotlib.pyplot as plt
    filtered_img_opt = pydbf.bilateral_filter_cuda(
        noisy_img_tensor.contiguous(), spatial_sigma, range_sigma)
    filtered_img_opt_clamped = filtered_img_opt.clamp(0.0, 1.0)
    img_tensor_clamped = img_tensor.clamp(0.0, 1.0)

    mse_per_pixel = ((filtered_img_opt_clamped - img_tensor_clamped)
                     ** 2).mean(dim=1).squeeze(0).cpu().detach().numpy()
    loss_map = mse_per_pixel

    filtered_img_opt_np = filtered_img_opt_clamped.detach(
    ).cpu().squeeze(0).permute(1, 2, 0).numpy()

    noisy_img_np = noisy_img_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    loss_map_norm = (loss_map - np.min(loss_map)) / \
        (np.max(loss_map) - np.min(loss_map) + 1e-8)
    if SHOW_ONLINE:
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.title("Noisy Image")
        plt.imshow(noisy_img_np)
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.title(
            f"Filtered (Optimized Params)\nLoss: {prev_loss:.6e}\nspatial_sigma: {spatial_sigma.data.cpu().numpy()}\nrange_sigma: {range_sigma.data.cpu().numpy()}")
        plt.imshow(filtered_img_opt_np)
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.title("Loss Map (MSE Loss)")
        plt.imshow(loss_map_norm, cmap='inferno')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        import imageio.v3 as iio
        iio.imwrite("noisy_image.png", (noisy_img_np * 255).astype(np.uint8))
        iio.imwrite("filtered_optimized.png",
                    (filtered_img_opt_np * 255).astype(np.uint8))
        iio.imwrite("loss_map.png", (loss_map_norm * 255).astype(np.uint8))

    print(f"Optimized spatial_sigma: {spatial_sigma.data.cpu().numpy()}")
    print(f"Optimized range_sigma: {range_sigma.data.cpu().numpy()}")


def input_optimization_test():
    import imageio.v3 as iio
    img_url = "https://images.unsplash.com/photo-1542664408056-8ec64f24de47?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1287"
    response = requests.get(img_url)
    img = iio.imread(BytesIO(response.content))
    img = (img.astype(np.float32) / 255.0)
    print("Original image shape:", img.shape)

    img_tensor = tensor(img).permute(2, 0, 1).unsqueeze(
        0).float().to('cuda')  # (1, C, H, W)

    noisy_input_init = torch.randn_like(img_tensor).clamp(0.0, 1.0)
    noisy_input = nn.Parameter(noisy_input_init.clone())

    spatial_sigma = tensor([6.0, 6.0], device='cuda')
    range_sigma = tensor([0.2], device='cuda')

    mse_loss = nn.MSELoss()

    def loss_fn(filtered, target):
        return mse_loss(filtered, target)

    optimizer = optim.Adam([
        {'params': [noisy_input], 'lr': 0.1},
    ])

    max_iters = 2**11
    tol = 1e-8
    patience = 32
    plateau_count = 0

    start_time = time.time()
    prev_loss = None
    for i in range(max_iters):
        optimizer.zero_grad()
        filtered_img = pydbf.bilateral_filter_cuda(
            noisy_input.contiguous(), spatial_sigma, range_sigma)
        loss = loss_fn(filtered_img, img_tensor)
        loss.backward()
        optimizer.step()

        noisy_input.data.clamp_(0.0, 1.0)

        if prev_loss is not None and abs(prev_loss - loss.item()) < 1e-8:
            plateau_count += 1
            if plateau_count >= patience:
                print(
                    f"Early stopping: loss plateau for {patience} iterations at iter {i+1}")
                break
        else:
            plateau_count = 0
        prev_loss = loss.item()

        if i % 10 == 0 or loss.item() < tol or i == max_iters - 1:
            print(f"Iter {i+1}/{max_iters}, Loss: {loss.item():.6e}")
        if loss.item() < tol:
            print("Early stopping due to tolerance.")
            break

    end_time = time.time()
    print(
        f"Input optimization finished in {end_time - start_time:.2f} seconds.")

    import matplotlib.pyplot as plt
    filtered_img_opt = pydbf.bilateral_filter_cuda(
        noisy_input.contiguous(), spatial_sigma, range_sigma)
    print(
        f"Spatial sigma: {spatial_sigma.detach().cpu().numpy()}, Range sigma: {range_sigma.detach().cpu().numpy()}")
    filtered_img_opt_clamped = filtered_img_opt.clamp(0.0, 1.0)
    img_tensor_clamped = img_tensor.clamp(0.0, 1.0)

    noisy_input_init_np = noisy_input_init.detach(
    ).cpu().squeeze(0).permute(1, 2, 0).numpy()
    noisy_input_np = noisy_input.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    filtered_img_opt_np = img_tensor_clamped.detach(
    ).cpu().squeeze(0).permute(1, 2, 0).numpy()

    if SHOW_ONLINE:
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 4, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.title("Initial Noisy Input")
        plt.imshow(noisy_input_init_np)
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.title("Optimized Input")
        plt.imshow(noisy_input_np)
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.title("Filtered (Optimized Input)")
        plt.imshow(filtered_img_opt_np)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        iio.imwrite("initial_noisy_input.png",
                    (noisy_input_init_np * 255).astype(np.uint8))
        iio.imwrite("optimized_input.png",
                    (noisy_input_np * 255).astype(np.uint8))
        iio.imwrite("filtered_optimized_input.png",
                    (filtered_img_opt_np * 255).astype(np.uint8))

def main():
    rl_filter_test()
    sigma_optimization_test()
    input_optimization_test()


if __name__ == "__main__":
    main()
