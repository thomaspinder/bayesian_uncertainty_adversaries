import matplotlib.pyplot as plt
from scipy.misc import imread

root = '/home/tpin3694/Documents/python/bayesian_uncertainty/data/chest_xray/train/'
pne = imread('{}pneumonia/person705_virus_1302.jpeg'.format(root))
nor = imread('/home/tpin3694/Documents/python/bayesian_uncertainty/data/chest_xray/train/normal/NORMAL2-IM-0604-0001.jpeg')

fig = plt.figure(figsize=(10,5))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
plt.subplots_adjust(wspace = 0.33 )
ax1.imshow(nor)
ax2.imshow(pne)
ax1.text(0.5, -0.15, 'Normal X-Ray',  ha="center", transform=ax1.transAxes)
ax2.text(0.5, -0.15, 'Pneumonia X-Ray',  ha="center", transform=ax2.transAxes)
plt.savefig('/home/tpin3694/Documents/python/bayesian_uncertainty/results/plots/pneumonia_ex.png')
plt.show()
