from PPO.Model import Model1, V_Model
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

img = cv2.imread("real1.png")
# img = cv2.imread("1.jpg")
img = cv2.resize(img, (64, 64))

img = np.array(img)

gray_img = img[:, :, 1]

plt.imshow(gray_img, cmap="gray")
plt.show()

actor = Model1(64, 64)

gray_img = np.reshape(gray_img, (1, 64, 64, 1))

gray_img = tf.cast(gray_img, tf.float32)



actor(gray_img)

actor.load_weights("trier3/policy.h5")

actor(gray_img)
