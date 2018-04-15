import cv2
import numpy as np


def reconstruct(img):
    u, s, vh = np.linalg.svd(img)
    e = np.eye(681, 492)
    d = np.zeros(s.shape)
    # d[0:20] = s[0:20]
    d[0:100] = s[0:100]
    s = e * d

    #print(d.shape)
    #print(vh.shape)

    re = np.matmul(np.matmul(u, s), vh)
    min = np.min(re)
    max = np.max(re)
    re = (re - min) / (max - min) * 255
    re = re.astype(np.uint8)
    return re

image = cv2.imread('image/sophie_marceau.jpg')

b = image[:, :, 0]
g = image[:, :, 1]
r = image[:, :, 2]

b_re = reconstruct(b)
g_re = reconstruct(g)
r_re = reconstruct(r)

color = np.zeros((681, 492, 3), np.uint8)
print(color.shape)
color[:, :, 0] = b_re
color[:, :, 1] = g_re
color[:, :, 2] = r_re

cv2.imshow('old', image)
cv2.imshow('svd', color)
cv2.waitKey(0)
cv2.destroyAllWindows()

