import matplotlib.pyplot as plt
import tensorflow as tf

a = tf.random_normal([2, 20])
sess = tf.Session()
out = sess.run(a)
x, y = out

plt.scatter(x, y)
plt.show()
