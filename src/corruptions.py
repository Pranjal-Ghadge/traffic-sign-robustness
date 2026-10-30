import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64

from imagecorruptions import corrupt, get_corruption_names

def apply_corruption(image_np, corruption_name, severity):
    """
    Applies a specific environmental noise/corruption to a numpy image (H, W, C) 
    with a severity level between 1 and 5.
    """
    
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
        
    corrupted_img = corrupt(image_np, corruption_name=corruption_name, severity=severity)
    return corrupted_img

def get_required_corruptions():
    """
    Returns the exact list of 14 corruption types specified in your lab guidelines.
    """
    
    corruptions = [
        "gaussian_noise",
        "impulse_noise",
        "shot_noise",
        "snow",
        "fog",
        "frost",
        "motion_blur",
        "zoom_blur",
        "defocus_blur",
        "brightness",
        "contrast",
        "pixelate",
        "elastic_transform",
        "jpeg_compression"
    ]
    return corruptions

if __name__ == "__main__":
  
    print("Available default corruptions:", get_corruption_names())
    print("Project required corruptions count:", len(get_required_corruptions()))