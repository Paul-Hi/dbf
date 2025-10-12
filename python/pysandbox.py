import colorsys
from torch import tensor, cuda, nn, optim, autograd
import pydbf
import numpy as np
import time
import requests
from io import BytesIO

print("DBF version:", pydbf.__version__)
print("DBF author:", pydbf.__author__)
print("DBF license:", pydbf.__license__)
print("DBF doc:", pydbf.__doc__)

# Set to True to display images using matplotlib, False to save them using imageio
SHOW_ONLINE = False


def simple_filter_test():
    # Load image - convert to tensor and apply bilateral filter
    import imageio.v3 as iio
    img = iio.imread("noisyP.png")

    img = (img.astype(np.float32) / 255.0)

    img = tensor(img).permute(
        2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
    print("Image shape:", img.shape)
    img = img.to('cuda')

    # Apply bilateral filter
    spatial_sigma = tensor([4.4, 4.6], device='cuda')  # (sigmaX, sigmaY)
    range_sigma = tensor([0.55], device='cuda')         # (sigmaR)
    n = 1  # Number of times to apply the filter
    filtered_img = img
    start_time = time.time()
    for i in range(n):
        filtered_img = pydbf.bilateral_filter(
            filtered_img, spatial_sigma, range_sigma / (i + 1))
    end_time = time.time()
    print(
        f"Bilateral filter applied {n} times in {end_time - start_time:.4f} seconds")
    filtered_img = filtered_img.clamp(0.0, 1.0).cpu().squeeze(
        0).permute(1, 2, 0).numpy()  # (H, W, C)

    if SHOW_ONLINE:
        # Display original and filtered images
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
        # Save filtered image
        import imageio.v3 as iio
        iio.imwrite("filteredP.png", (filtered_img * 255).astype(np.uint8))


def rl_filter_test():
    # Load image - convert to tensor and apply bilateral filter
    import imageio.v3 as iio
    # Load image directly from website
    img_url = "https://assets-global.website-files.com/60e4d0d0155e62117f4faef3/620c0292e79375ecd81ce99b_Example%20of%20luminance%20noise%20zoomed.jpg"
    response = requests.get(img_url)
    img = iio.imread(BytesIO(response.content))

    img = (img.astype(np.float32) / 255.0)

    img = tensor(img).permute(
        2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
    img = img.to('cuda')

    # Apply bilateral filter
    spatial_sigma = tensor(
        [20.0, 20.0], device='cuda')  # (sigmaX, sigmaY)
    range_sigma = tensor([0.1], device='cuda')         # (sigmaR)
    n = 1  # Number of times to apply the filter
    filtered_img = img
    start_time = time.time()
    for i in range(n):
        filtered_img = pydbf.bilateral_filter(
            filtered_img, spatial_sigma, range_sigma / (i + 1), 21)
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
        # Display original and filtered images
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
        # Save filtered image
        import imageio.v3 as iio
        iio.imwrite("example_filtered.png",
                    (filtered_img * 255).astype(np.uint8))


def simple_filter_optimization_test():
    # autograd.set_detect_anomaly(True)
    # Load image - convert to tensor and apply bilateral filter
    import imageio.v3 as iio
    # img_url = "https://val.cds.iisc.ac.in/reflecting-reality.github.io/assets/figures/chair_normal.png"
    # response = requests.get(img_url)
    # img = iio.imread(BytesIO(response.content))
    # img = (img.astype(np.float32) / 255.0)
    img = iio.imread("test_image.png")
    img = (img.astype(np.float32) / 255.0)

    # fix seed
    # np.random.seed(32)

    # Pick a random 512x512 crop from the image
    h, w = img.shape[:2]
    crop_h, crop_w = 512, 512
    if h < crop_h or w < crop_w:
        raise ValueError(f"Image is too small for 512x512 crop: got {h}x{w}")
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    img = img[top:top + crop_h, left:left + crop_w]

    # Add noise in HSV space
    img_rgb = img
    # Convert to HSV
    img_hsv = np.empty_like(img_rgb)
    for y in range(img_rgb.shape[0]):
        for x in range(img_rgb.shape[1]):
            img_hsv[y, x] = colorsys.rgb_to_hsv(*img_rgb[y, x])
    # Add noise to HSV channels
    hsv_noise = np.random.normal(0, 0.045, img_hsv.shape).astype(np.float32)
    noisy_hsv = np.clip(img_hsv + hsv_noise, 0.0, 1.0)
    # Convert back to RGB
    noisy_img = np.empty_like(img_rgb)
    for y in range(noisy_hsv.shape[0]):
        for x in range(noisy_hsv.shape[1]):
            noisy_img[y, x] = colorsys.hsv_to_rgb(*noisy_hsv[y, x])

    noisy_img_tensor = tensor(noisy_img).permute(
        2, 0, 1).unsqueeze(0).float().to('cuda')  # (1, C, H, W)
    img_tensor = tensor(img).permute(
        2, 0, 1).unsqueeze(0).float().to('cuda')

    # Optimize spatial_sigma and range_sigma to make filtered noisy_img_tensor as close as possible to img_tensor
    spatial_sigma = nn.Parameter(tensor([1.0, 1.0], device='cuda'))
    range_sigma = nn.Parameter(tensor([1.0], device='cuda'))
    mse_loss = nn.MSELoss()

    def loss_fn(filtered, target):
        return mse_loss(filtered, target)

    optimizer = optim.Adam([
        {'params': [spatial_sigma], 'lr': 0.1},
        {'params': [range_sigma], 'lr': 0.12}
    ])

    max_iters = 2**10
    tol = 1e-4
    patience = 128
    plateau_count = 0

    prev_loss = None
    for i in range(max_iters):
        optimizer.zero_grad()
        filtered_img = pydbf.bilateral_filter(
            noisy_img_tensor, spatial_sigma, range_sigma).clamp(0.0, 1.0)
        # filtered_img = pydbf.bilateral_filter(
        #     filtered_img, spatial_sigma, range_sigma).clamp(0.0, 1.0)
        loss = loss_fn(filtered_img, img_tensor)
        loss.backward()
        optimizer.step()
        # Clamp parameters to be strictly positive
        spatial_sigma.data.clamp_(min=1e-6)
        range_sigma.data.clamp_(min=1e-6)

        # Early stopping if loss plateaus for 'patience' iterations
        if prev_loss is not None and abs(prev_loss - loss.item()) < 1e-7:
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

    # Display results
    import matplotlib.pyplot as plt
    # Compute loss map between filtered and original image using the same loss as in optimization
    filtered_img_opt = pydbf.bilateral_filter(
        noisy_img_tensor, spatial_sigma, range_sigma)
    # filtered_img_opt = pydbf.bilateral_filter(
    #     filtered_img_opt, spatial_sigma, range_sigma)
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
        plt.title("Filtered (Optimized Params)")
        plt.imshow(filtered_img_opt_np)
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.title("Loss Map (MSE Loss)")
        plt.imshow(loss_map_norm, cmap='inferno')
        plt.axis('off')
        plt.show()
    else:
        import imageio.v3 as iio
        iio.imwrite("noisy_image.png", (noisy_img_np * 255).astype(np.uint8))
        iio.imwrite("filtered_optimized.png",
                    (filtered_img_opt_np * 255).astype(np.uint8))
        iio.imwrite("loss_map.png", (loss_map_norm * 255).astype(np.uint8))

    print(f"Optimized spatial_sigma: {spatial_sigma.data.cpu().numpy()}")
    print(f"Optimized range_sigma: {range_sigma.data.cpu().numpy()}")


def synthetic_optimization_test():
    img = np.linspace(0, 1, 512).reshape(1, 512).repeat(512, axis=0)
    img = np.stack([img, img, img], axis=-1)  # (512, 512, 3)
    noise = np.random.normal(0, 0.08, img.shape).astype(np.float32)
    noisy_img = np.clip(img + noise, 0.0, 1.0)

    img_tensor = tensor(img).permute(
        2, 0, 1).unsqueeze(0).float().to('cuda')
    noisy_img_tensor = tensor(noisy_img).permute(
        2, 0, 1).unsqueeze(0).float().to('cuda')

    spatial_sigma = nn.Parameter(tensor([2.0, 2.0], device='cuda'))
    range_sigma = nn.Parameter(tensor([0.3], device='cuda'))
    mse_loss = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': [spatial_sigma], 'lr': 0.08},
        {'params': [range_sigma], 'lr': 0.09}
    ])

    max_iters = 256
    tol = 1e-4
    for i in range(max_iters):
        optimizer.zero_grad()
        filtered_img = pydbf.bilateral_filter(
            noisy_img_tensor, spatial_sigma, range_sigma).clamp(0.0, 1.0)
        loss = mse_loss(filtered_img, img_tensor)
        loss.backward()
        optimizer.step()
        spatial_sigma.data.clamp_(min=1e-6)
        range_sigma.data.clamp_(min=1e-6)
        if i % 16 == 0 or loss.item() < tol or i == max_iters - 1:
            print(f"Iter {i+1}/{max_iters}, Loss: {loss.item():.6e}, spatial_sigma: {spatial_sigma.data.cpu().numpy()}, range_sigma: {range_sigma.data.cpu().numpy()}")
        if loss.item() < tol:
            print("Early stopping due to tolerance.")
            break

    filtered_img_np = filtered_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    if SHOW_ONLINE:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title("Noisy")
        plt.imshow(noisy_img)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title("Filtered")
        plt.imshow(filtered_img_np)
        plt.axis('off')
        plt.show()
    else:
        import imageio.v3 as iio
        filtered_img_np = filtered_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        iio.imwrite("synthetic_filtered.png",
                    (filtered_img_np * 255).astype(np.uint8))


def main():
    simple_filter_test()
    rl_filter_test()
    simple_filter_optimization_test()
    synthetic_optimization_test()


if __name__ == "__main__":
    main()
