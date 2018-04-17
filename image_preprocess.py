import scipy.misc
import numpy as np


class ImagePreprocessors:

    @staticmethod
    def pre_process_image(image):
        img_size = (84, 96)
        # grayscale
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
        # downsample
        ds_image = scipy.misc.imresize(img_gray, size=img_size, interp='bicubic')
        # erase padding of wide resolutions
        w_h_diff = abs(img_size[1] - img_size[0])
        is_square_image = w_h_diff == 0
        if not is_square_image:
            padding = int(w_h_diff / 2)
            padding_to_delete = [i for i in range(padding)]
            padding_to_delete_end = [i for i in range(len(ds_image[0]) - 1, (len(ds_image[0]) - 1 - padding), -1)]
            padding_to_delete.extend(padding_to_delete_end)
            # modify downsampled image to be square
            ds_image = np.delete(ds_image, padding_to_delete, axis=1)
        return ds_image
