def canny_for_video(obj, sigma_1=2.5, sigma_2=2.5):
    kernel_A = np.array([[0,0,1],[0,1,0],[1,0,0]], np.uint8)
    kernel_B = np.array([[1,0,0],[0,1,0],[0,0,1]], np.uint8)
    kernel_C = np.array([[0,0,0],[1,1,1],[0,0,0]], np.uint8)
    kernel_D = np.array([[0,1,0],[0,1,0],[0,1,0]], np.uint8)
    kernels = [kernel_A, kernel_B, kernel_C, kernel_D]

    def get_edge(img, params):
        #return 1*feature.canny(img, *params)
        return filters.sobel(img)
        #return filters.roberts(img)

    def run_morph(edge):
        global kernels
        if True: return edge
        _edge = np.zeros(edge.shape, np.uint8)
        while not (edge == _edge).all():
            for k in kernels:
                #edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k)
                edge = morphology.binary_closing(edge, selem=k)
            _edge = np.copy(edge)
        return morphology.skeletonize(edge) * 255

    video = []
    while True:
        ret, frame = obj.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        video.append(frame)
    video = np.array(video)
    print video.shape

    video_vr = np.zeros(video.shape, np.uint8)
    for i_row in xrange(video.shape[1]):
        edge = get_edge(video[:, i_row, :], [sigma_1])
        video_vr[:, i_row, :] = run_morph(edge) #cv2.normalize(run_morph(edge), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    for i_frame in xrange(video.shape[0]):

        a = video[i_frame, ...]
        edge = get_edge(video[i_frame, :, :], [sigma_2])
        b = run_morph(edge)
        c = video_vr[i_frame, ...]
        #b = cv2.normalize(run_morph(edge), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        c = cv2.normalize(video_vr[i_frame, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cv2.imshow('tmp', np.hstack((a,b,c)))
        k = cv2.waitKey(30) & 0xff
        if k == 27: break
    cv2.destroyAllWindows()
    return (video, video_vr)



#anisotropic diffusion

import numpy as np
from scipy.ndimage.filters import gaussian_filter

def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    """
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to np.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    """
    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return np.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)

    # initialize output array
    out = np.array(img, dtype=np.float32, copy=True)

    # set default voxel spacong if not suppliec
    if None == voxelspacing:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [np.zeros_like(out) for _ in xrange(out.ndim)]

    for _ in xrange(niter):

        # calculate the diffs
        for i in xrange(out.ndim):
            slicer = [slice(None, -1) if j == i else slice(None) for j in xrange(out.ndim)]
            deltas[i][slicer] = np.diff(out, axis=i)

        # update matrices
        matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in xrange(out.ndim):
            slicer = [slice(1, None) if j == i else slice(None) for j in xrange(out.ndim)]
            matrices[i][slicer] = np.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (np.sum(matrices, axis=0))

    return out
