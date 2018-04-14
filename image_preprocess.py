import scipy.misc


class ImagePreprocessors:

    @staticmethod
    def pre_process_image(image):
        final_img_size = (128, 112)
        # grayscale
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
        # downsample
        ds_image = scipy.misc.imresize(img_gray, size=final_img_size, interp='bicubic')
        return ds_image
