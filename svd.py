import cv2
import numpy as np


def reconstruct(img):
    u, s, vh = np.linalg.svd(img)
    d = np.zeros((height, weight))
    for i in range(0, num_sv):
        d[i, i] = s[i]
    s = d

    ret = np.matmul(np.matmul(u, s), vh)
    val_min = np.min(ret)
    val_max = np.max(ret)
    ret = (ret - val_min) / (val_max - val_min) * 255
    ret = ret.astype(np.uint8)
    return ret


if __name__ == '__main__':
    image = cv2.imread('image/sophie_marceau.jpg')
    # image = cv2.imread('image/fun.jpeg')

    height, weight = image.shape[:2]
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    for num_sv in [1, 5, 20, 50, 100]:
        b_re = reconstruct(b)
        g_re = reconstruct(g)
        r_re = reconstruct(r)

        vis = np.zeros((height, weight, 3), np.uint8)

        vis[:, :, 0] = b_re
        vis[:, :, 1] = g_re
        vis[:, :, 2] = r_re

        # cv2.imshow('old', image)
        # cv2.imshow('svd', vis)
        cv2.imwrite('image/sv-{}.png'.format(num_sv), vis)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()
