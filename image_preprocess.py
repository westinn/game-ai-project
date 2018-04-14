import scipy.ndimage


class ImagePreprocessors:

    def pre_process_image(image):
        final_frame = image[-1]
        state_img_size = len(image) * len(image[0])
        # grayscale
        r, g, b = final_frame[:, :, 0], final_frame[:, :, 1], final_frame[:, :, 2]
        img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
        # downsample
        ds_image = scipy.misc.imresize(img_gray, size=state_img_size, interp='bicubic')
        return ds_image
