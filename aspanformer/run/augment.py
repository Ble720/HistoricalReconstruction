import cv2
import numpy as np

def apply_augments(image):
    image = blur(image)
    return image

def blur(image):
    return cv2.GaussianBlur(image, (15,15), 0)

def scale_brightness(image):
    target_mean, target_std = 0, 1
    current_mean, current_std = image.mean(), image.std()

    # Avoid divide-by-zero
    if current_std == 0:
        return image

    # Scale factor for adjusting contrast
    scale = target_std / current_std
    # Offset for adjusting brightness
    offset = target_mean - scale * current_mean

    # Apply the adjustments
    im = image.astype(np.float32) * scale + offset
    im = np.clip(im, 0, 255).astype(np.uint8)
    #im = cv2.bitwise_not(im)
    return im


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)