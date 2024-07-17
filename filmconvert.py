import os
import sys
import imageio
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import rawpy
import argparse
import time
from skimage import exposure
from scipy.ndimage import convolve
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import matplotlib

matplotlib.use("Agg")


def is_raw_file(file_path):
    """Check if the file is a RAW image file supported by rawpy."""
    try:
        with rawpy.imread(file_path) as _:
            return True
    except:
        return False


def is_tiff_file(file_path):
    """Check if the file is a TIFF image file."""
    try:
        with Image.open(file_path) as img:
            return img.format == "TIFF" and file_path.lower().endswith(".tiff")
    except:
        return False


def white_balance(img, central_region):
    """Apply white balance to a 16-bit image using simple scaling."""
    avgR, avgG, avgB = np.mean(central_region, axis=(0, 1))
    kR = avgG / avgR
    kB = avgG / avgB
    img[:, :, 0] = np.clip(img[:, :, 0] * kR, 0, 65535)
    img[:, :, 2] = np.clip(img[:, :, 2] * kB, 0, 65535)
    return img


def balance_border(image, border_width, display_edges=False):
    def find_dynamic_edge_values(image, border_width):
        height, width = image.shape[:2]
        border_width_px = int(min(height, width) * border_width / 100)
        central_region = get_central_region(image, border_width_px)
        central_brightest_value = np.percentile(central_region, 85)

        border_regions = np.concatenate(
            [
                image[:border_width_px, :].flatten(),
                image[-border_width_px:, :].flatten(),
                image[border_width_px:-border_width_px, :border_width_px].flatten(),
                image[border_width_px:-border_width_px, -border_width_px:].flatten(),
            ]
        )

        valid_border_values = border_regions[
            (central_brightest_value <= border_regions)
            & (border_regions < np.iinfo(image.dtype).max - 5000)
        ]

        edge_value = (
            np.median(valid_border_values)
            if valid_border_values.size
            else central_brightest_value
        )
        border_mask = (image >= edge_value) & (image < np.iinfo(image.dtype).max - 5000)
        border_mask[
            border_width_px:-border_width_px, border_width_px:-border_width_px
        ] = False

        return edge_value, border_mask

    def white_balance_with_edge(image, edge_value):
        avg_edge_value = np.mean(edge_value)
        balanced_image = np.clip(
            image * (avg_edge_value / edge_value), 0, 65535
        ).astype(np.uint16)
        return balanced_image

    edge_values = []
    _, combined_border_mask = find_dynamic_edge_values(image[..., 0], border_width)

    for i in range(3):
        border_values = image[..., i][combined_border_mask]
        edge_value = (
            np.median(border_values)
            if border_values.size
            else np.percentile(get_central_region(image[..., i], border_width), 95)
        )
        edge_values.append(edge_value)

    if display_edges:
        edge_image = np.zeros_like(image[..., 0])
        edge_image[combined_border_mask] = 65535
        plt.imshow(edge_image, cmap="gray")
        plt.title("Detected Edges")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"detected_edges_{timestamp}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Detected edges plot saved as {plot_filename}")

    return white_balance_with_edge(image, np.array(edge_values))


def get_central_region(image, border_width):
    h, w = image.shape[:2]
    border_width = min(border_width, h // 2, w // 2)
    return image[border_width : h - border_width, border_width : w - border_width]


def invert_image(image):
    """Invert a 16-bit image."""
    return np.iinfo(image.dtype).max - image


def process_raw_image(path, output, index):
    with rawpy.imread(path) as raw:
        output[index] = raw.postprocess(**PARAMS_CONVERSION)


def merge_images(path_red, path_green, path_blue):
    images = [None, None, None]
    threads = [
        threading.Thread(target=process_raw_image, args=(path, images, i))
        for i, path in enumerate([path_red, path_green, path_blue])
        if is_raw_file(path)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    red, green, blue = images
    merged_frames = np.zeros(red.shape, dtype=np.uint16)
    merged_frames[:, :, 0] = red[:, :, 0]
    merged_frames[:, :, 1] = green[:, :, 1]
    merged_frames[:, :, 2] = blue[:, :, 2]
    return merged_frames


def contrast_stretch(image, border_width_percentage):
    """Apply contrast stretching to the image based on the central region."""
    border_width = int(min(image.shape[:2]) * border_width_percentage / 100)
    central_region = get_central_region(image, border_width)

    lower_bound, upper_bound = np.percentile(central_region, [5, 99])
    if lower_bound == upper_bound:
        raise ValueError(
            "The central region has no contrast. Please adjust the border width."
        )

    return exposure.rescale_intensity(
        image, in_range=(lower_bound, upper_bound), out_range=(10, 65535)
    ).astype(np.uint16)


def crop_image(image, border_width_percentage):
    """Crop the image by a given percentage of its border width."""
    return get_central_region(
        image, int(min(image.shape[:2]) * border_width_percentage / 100)
    )


def noise_reduction(image, filter_size=2):
    """Perform noise reduction using a simple mean filter."""
    kernel = np.ones((filter_size, filter_size)) / (filter_size**2)
    smoothed_image = np.zeros_like(image)

    for i in range(image.shape[2]):  # Apply filter to each channel
        smoothed_image[..., i] = convolve(image[..., i], kernel, mode="reflect")

    return smoothed_image


def process_image_set(
    path_red, path_green, path_blue, operations, border_width_percentage, crop
):
    print("Merging RGB channels...")
    merged_image = merge_images(path_red, path_green, path_blue)
    print("Merged image shape:", merged_image.shape)

    if operations.get("color_balance"):
        print("Applying white balance...")
        border_width = int(min(merged_image.shape[:2]) * border_width_percentage / 100)
        central_region = get_central_region(merged_image, border_width)
        merged_image = white_balance(merged_image, central_region)
        print("Balancing black points...")
        merged_image = balance_border(merged_image, border_width_percentage)

    if operations.get("invert"):
        print("Inverting image...")
        merged_image = invert_image(merged_image)

    final_image = contrast_stretch(merged_image, border_width_percentage)

    print("Performing noise reduction...")
    final_image = noise_reduction(final_image)
    print("Noise reduction completed.")

    if crop:
        final_image = crop_image(final_image, border_width_percentage)

    output_path = os.path.join(
        os.path.dirname(path_red),
        f'{os.path.basename(path_red).split(".")[0]}_final.tiff',
    )
    imageio.imsave(output_path, final_image.astype("uint16"))
    print(f"Final image saved to {output_path}")


def process_single_raw_file(image_path, operations, border_width_percentage, crop):
    print(f"Processing single RAW file: {image_path}")
    image = debayer_image(image_path)

    if operations.get("color_balance"):
        print("Applying white balance...")
        border_width = int(min(image.shape[:2]) * border_width_percentage / 100)
        central_region = get_central_region(image, border_width)
        image = white_balance(image, central_region)
        print("Balancing black points...")
        image = balance_border(image, border_width_percentage)

    if operations.get("invert"):
        print("Inverting image...")
        final_image = invert_image(image)
    else:
        final_image = image

    # final_image = contrast_stretch(final_image, border_width_percentage)

    print("Performing noise reduction...")
    final_image = noise_reduction(final_image)
    print("Noise reduction completed.")

    if crop:
        final_image = crop_image(final_image, border_width_percentage)

    print(
        f"Final image dtype: {final_image.dtype}, min: {np.min(final_image)}, max: {np.max(final_image)}"
    )
    output_path = os.path.join(
        os.path.dirname(image_path),
        f'{os.path.basename(image_path).split(".")[0]}_final.tiff',
    )
    imageio.imsave(output_path, final_image.astype("uint16"))
    print(f"Final image saved to {output_path}")


def process_single_tiff_file(image_path, operations, border_width_percentage, crop):
    print(f"Processing single TIFF file: {image_path}")
    with Image.open(image_path) as img:
        image = np.array(img)
        if image.dtype != np.uint16:
            image = image.astype(np.uint16) * 257  # Scale up to 16-bit

    if operations.get("color_balance"):
        print("Applying white balance...")
        border_width = int(min(image.shape[:2]) * border_width_percentage / 100)
        central_region = get_central_region(image, border_width)
        image = white_balance(image, central_region)
        print("Balancing black points...")
        image = balance_border(image, border_width_percentage)

    if operations.get("invert"):
        print("Inverting image...")
        final_image = invert_image(image)
    else:
        final_image = image

    final_image = contrast_stretch(final_image, border_width_percentage)

    print("Performing noise reduction...")
    final_image = noise_reduction(final_image)
    print("Noise reduction completed.")

    if crop:
        final_image = crop_image(final_image, border_width_percentage)

    print(
        f"Final image dtype: {final_image.dtype}, min: {np.min(final_image)}, max: {np.max(final_image)}"
    )
    output_path = os.path.join(
        os.path.dirname(image_path),
        f'{os.path.basename(image_path).split(".")[0]}_final.tiff',
    )
    imageio.imsave(output_path, final_image.astype("uint16"))
    print(f"Final image saved to {output_path}")


def process_bw_film(image_path, border_width_percentage, crop):
    print(f"Processing black and white film file: {image_path}")

    if is_raw_file(image_path):
        image = debayer_image(image_path)
    elif is_tiff_file(image_path):
        with Image.open(image_path) as img:
            image = np.array(img)
            if image.dtype != np.uint16:
                image = image.astype(np.uint16) * 257  # Scale up to 16-bit
    else:
        raise ValueError(
            "Unsupported file format. Only RAW and TIFF files are supported."
        )

    if crop:
        image = crop_image(image, border_width_percentage)

    bw_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint16)
    final_image = contrast_stretch(bw_image, border_width_percentage)
    final_image = invert_image(final_image)

    print(
        f"Final image dtype: {final_image.dtype}, min: {np.min(final_image)}, max: {np.max(final_image)}"
    )
    output_path = os.path.join(
        os.path.dirname(image_path),
        f'{os.path.basename(image_path).split(".")[0]}_bw_final.tiff',
    )
    imageio.imsave(output_path, final_image.astype(np.uint16))
    print(f"Final black and white film image saved to {output_path}")


def process_folder(
    folder_path, capture_type, operations, border_width_percentage, crop
):
    files = sorted(os.listdir(folder_path))
    raw_files = [f for f in files if is_raw_file(os.path.join(folder_path, f))]
    tiff_files = [f for f in files if is_tiff_file(os.path.join(folder_path, f))]

    if capture_type == "rgb":
        triplets = [
            (
                os.path.join(folder_path, raw_files[i]),
                os.path.join(folder_path, raw_files[i + 1]),
                os.path.join(folder_path, raw_files[i + 2]),
            )
            for i in range(0, len(raw_files), 3)
            if i + 2 < len(raw_files)
        ]
        triplets += [
            (
                os.path.join(folder_path, tiff_files[i]),
                os.path.join(folder_path, tiff_files[i + 1]),
                os.path.join(folder_path, tiff_files[i + 2]),
            )
            for i in range(0, len(tiff_files), 3)
            if i + 2 < len(tiff_files)
        ]

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(
                    process_triplet, triplet, operations, border_width_percentage, crop
                )
                for triplet in triplets
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

    elif capture_type == "i":
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    process_single_raw_file,
                    os.path.join(folder_path, raw_file),
                    operations,
                    border_width_percentage,
                    crop,
                )
                for raw_file in raw_files
            ] + [
                executor.submit(
                    process_single_tiff_file,
                    os.path.join(folder_path, tiff_file),
                    operations,
                    border_width_percentage,
                    crop,
                )
                for tiff_file in tiff_files
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

    elif capture_type == "bw":
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    process_bw_film,
                    os.path.join(folder_path, raw_file),
                    border_width_percentage,
                    crop,
                )
                for raw_file in raw_files
            ] + [
                executor.submit(
                    process_bw_film,
                    os.path.join(folder_path, tiff_file),
                    border_width_percentage,
                    crop,
                )
                for tiff_file in tiff_files
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")


def process_triplet(triplet, operations, border_width_percentage, crop):
    path_red, path_green, path_blue = triplet
    process_image_set(
        path_red, path_green, path_blue, operations, border_width_percentage, crop
    )


def debayer_image(image_path):
    with rawpy.imread(image_path) as raw:
        return raw.postprocess(**PARAMS_CONVERSION)


def main():
    parser = argparse.ArgumentParser(
        description="Image processing utility",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
    python script.py -m file -r red.tiff -g green.tiff -b blue.tiff -ct rgb --merge --color_balance --invert --border_width 5 --crop
    python script.py -m folder -f /path/to/folder -ct i --merge --color_balance --invert --border_width 5 --crop
    python script.py -m file -r bw_film.tiff -ct bw --border_width 5 --crop
    """,
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["file", "folder"],
        required=True,
        help="Processing mode: file or folder",
    )
    parser.add_argument(
        "-ct",
        "--capture_type",
        choices=["rgb", "i", "bw"],
        required=True,
        help="Capture type: rgb, individual (i), or bw",
    )
    parser.add_argument("-r", "--red", help="Path to the red channel image")
    parser.add_argument("-g", "--green", help="Path to the green channel image")
    parser.add_argument("-b", "--blue", help="Path to the blue channel image")
    parser.add_argument(
        "-f", "--folder", help="Path to the folder containing RAW or TIFF files"
    )
    parser.add_argument("--merge", action="store_true", help="Only merge images")
    parser.add_argument(
        "--color_balance", action="store_true", help="Merge and apply color balance"
    )
    parser.add_argument(
        "--invert", action="store_true", help="Merge, color balance, and invert"
    )
    parser.add_argument(
        "--border_width",
        type=float,
        default=5,
        help="Border width as a percentage of the image dimensions (e.g., 5 for 5%)",
    )
    parser.add_argument("--crop", action="store_true", help="Crop the images on output")
    parser.add_argument(
        "--quick_help", action="store_true", help="Show quick help with usage examples"
    )

    args = parser.parse_args()

    if args.quick_help:
        parser.print_help()
        sys.exit(0)

    operations = {"color_balance": args.color_balance, "invert": args.invert}

    global PARAMS_CONVERSION
    PARAMS_CONVERSION = {
        "demosaic_algorithm": rawpy.DemosaicAlgorithm.AHD,
        "use_camera_wb": True,
        "highlight_mode": rawpy.HighlightMode(0),
        "output_color": rawpy.ColorSpace(0),
        "dcb_enhance": False,
        "output_bps": 16,
        "gamma": (2.2, 0),
        "exp_shift": 4,
        "no_auto_bright": True,
        "no_auto_scale": True,
        "use_auto_wb": True,
        "user_wb": [1, 1, 1, 1],
    }

    start = time.time()

    if args.mode == "file":
        if args.capture_type == "rgb":
            if not (args.red and args.green and args.blue):
                print(
                    "In file mode with rgb capture type, you must provide paths for the red, green, and blue channel images."
                )
                sys.exit(1)
            process_image_set(
                args.red,
                args.green,
                args.blue,
                operations,
                args.border_width,
                args.crop,
            )
        elif args.capture_type == "i":
            if not args.red:
                print(
                    "In file mode with individual capture type, you must provide the path to the RAW or TIFF image."
                )
                sys.exit(1)
            if is_raw_file(args.red):
                process_single_raw_file(
                    args.red, operations, args.border_width, args.crop
                )
            elif is_tiff_file(args.red):
                process_single_tiff_file(
                    args.red, operations, args.border_width, args.crop
                )
        elif args.capture_type == "bw":
            if not args.red:
                print(
                    "In file mode with bw capture type, you must provide the path to the RAW or TIFF image."
                )
                sys.exit(1)
            process_bw_film(args.red, args.border_width, args.crop)

    elif args.mode == "folder":
        if not args.folder:
            print(
                "In folder mode, you must provide a folder path containing RAW or TIFF files."
            )
            sys.exit(1)

        process_folder(
            args.folder, args.capture_type, operations, args.border_width, args.crop
        )

    end = time.time()
    print(f"Processing took {end-start} seconds")


if __name__ == "__main__":
    main()
