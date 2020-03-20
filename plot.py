import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = 'WenQuanYi Micro Hei'

col = row = 5
size = 10
fig = plt.figure(figsize=(size, size))

c = np.load('./datasets/captcha.npz')
ci = c.get('images')
cl = c.get('labels')

labels = open('./datasets/texts.txt').readlines()

ax = []
for i in range(1, col*row+1):
    ax.append(fig.add_subplot(row, col, i))
    ax[-1].set_title(labels[cl[i]])
    plt.imshow(ci[i])

plt.show()
