thr_min_gap = 10
thr_min_pixels = 5

idx = 0
kernel = np.ones([3,3], np.uint8)
kernel = kernel * -1
kernel[1, :] = 2
list_h = []
for im_1 in np.hsplit(img, 120/5):
    im_1 = np.array(im_1 > 0, np.uint8)
    for col_1 in xrange(5, im_1.shape[1]):
        if im_1[:, col_1].sum() >= thr_min_pixels: break
    if col_1 + 1 >= im_1.shape[1]-5: continue
    for col_2 in xrange(im_1.shape[1]-5, col_1, -1):
        if im_1[:, col_2].sum() >= thr_min_pixels: break
    if col_2 == 0: continue
    if col_2 - col_1 + 1 < thr_min_gap: continue

    pattern = im_1[:, col_1:col_2+1]
    pattern = cv2.filter2D(pattern, cv2.CV_8U, kernel)

    pattern = cv2.resize(pattern, (100, 100), cv2.INTER_CUBIC)

    radius = 20
    lbp = local_binary_pattern(pattern, 3 * radius, radius, 'uniform');

    fig = plt.figure();
    ax = fig.add_subplot(1, 3, 1);
    plt.imshow(pattern, cmap='gray');
    """
    ax = fig.add_subplot(1,3,2);
    hist(plt, lbp)
    #plt.show()
    """
    ax = fig.add_subplot(1,3,2)
    list_h.append(np.array([np.sum(pattern[:, col]) for col in xrange(pattern.shape[1])]))
    plt.bar(range(pattern.shape[1]), list_h[-1])

    ax  = fig.add_subplot(1,3,3)
    h = hog(pattern, 4, (8,8), (1,1))
    plt.hist(h)

    print idx,
    print ('%.3f %.3f %.3f') % ((pattern>0).sum() * 1.0 / pattern.size, pattern.mean(), pattern.std())


    plt.savefig('/tmp/test/test%d.png' % idx)
    idx+=1
