import cv2
import numpy as np

num_sv = 100


def reconstruct(img):
    u, s, vh = np.linalg.svd(img)
    e = np.eye(height, weight)
    d = np.zeros(s.shape)
    d[0:num_sv] = s[0:num_sv]
    s = e * d

    ret = np.matmul(np.matmul(u, s), vh)
    val_min = np.min(ret)
    val_max = np.max(ret)
    ret = (ret - val_min) / (val_max - val_min) * 255
    ret = ret.astype(np.uint8)
    return ret


if __name__ == '__main__':
    image = cv2.imread('image/sophie_marceau.jpg')

    height, weight = image.shape[:2]
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    b_re = reconstruct(b)
    g_re = reconstruct(g)
    r_re = reconstruct(r)

    vis = np.zeros((height, weight, 3), np.uint8)

    vis[:, :, 0] = b_re
    vis[:, :, 1] = g_re
    vis[:, :, 2] = r_re

    cv2.imshow('old', image)
    cv2.imshow('svd', vis)
    cv2.imwrite('image/sv-{}.png'.format(num_sv), vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
