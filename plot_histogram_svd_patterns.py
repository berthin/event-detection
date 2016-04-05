
img = io.imread (PATH_KTH_VR + action + '/person01_boxing_d1_uncomp.png', as_grey=True)
#im1 = img[:, 180*4:180*7] / 255.
im1 = img / 255.

mean = np.mean (im1, axis=0)
std = np.std (im1, axis=0)

df = pandas.DataFrame (np.vstack ((std, std/mean)).T, columns=['Std', 'Coeff. of Variation'])

df.plot (kind='area', stacked=False, figsize=(27, 5), legend=True)

plt.show()

plt.savefig ('/tmp/myfig.pdf', bbox_inches='tight')

