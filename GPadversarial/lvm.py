import GPy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

"""
There are two different types of GP classifier here.

1) A simple GP classifier, using the Laplace approximation.

2) A GPLVM with a linear classifier ontop.
"""

def build_model(X,y):
    """
    """
    q = 4
    kern = GPy.kern.RBF(q, ARD=True)
    m = GPy.models.GPLVM(X, q, kernel=kern)
    m.optimize(messages=1, max_iters=100)
    
    #Linear
    clf = SGDClassifier(loss="hinge", alpha=0.01, n_iter=200, fit_intercept=True)
    clf.fit(m.X,y)
    return m, clf


#### This was all to try and produce adv. samples by moving about in latent space to a point on/near the boundary...
#### it turns out not to really work...

def gram_schmidt_columns(X):
    """The gram schmidt transform will provide orthogonal vectors. If the vectors we provide exist in a planar manifold
    the resulting vectors should remain in the manifold. TBC!!!"""
    Q, R = np.linalg.qr(X)
    return Q

def gen_ortho_basis(v):
    """Here we get a set of orthonormal vectors in a hyperplane, defined by the normal v. We can use these orthonormal 
    vectors as a basis to generate a set of points spanning some of the hyperplane."""
    M = np.zeros([len(v),len(v)-1])
    for i in range(len(v)-1):
        M[0,i] = v[i+1]
        M[i+1,i] = -v[0]
    w = gram_schmidt_columns(M)
    
    #check bases are orthonormal
    for i in range(len(v)-1):
        for j in range(i+1,len(v)-1):
            np.testing.assert_almost_equal(np.dot(w[:,i],w[:,j]),0)
        np.testing.assert_almost_equal(np.dot(w[:,i],v),0)
        
        np.testing.assert_almost_equal(np.sum(w[:,i]**2),1) #check normal
    return w

def nearest_point(v,offset,p):
    """
    Find the nearest point on the hyperplane defined by the normal vector v, and offset, to the point p.
    """
    #if v is a unit vector, then we can scrap the denominator
    scale = np.sqrt(np.dot(v,v))
    vnorm = v / scale
    offsetnorm = offset / scale
    c = -offsetnorm-np.dot(p,vnorm)
    plane_point = p + c * vnorm
    return plane_point

def gen_grid(v,offset,p,stepsize=5.0,num_steps=15):
    """
    Given a hyperplane (defined by normal v and offset), generate a grid of points
    across the hyperplane, spaced stepsize apart with numsteps*2+1 points in each
    dimension. The centre of these points is organised to be as close to the point
    p as possible."""
    w = gen_ortho_basis(v/np.sqrt(np.sum(v**2)))
    import itertools
    iterables = []
    for i in range(len(v)-1):
        iterables.append(np.arange(-stepsize*num_steps,stepsize*num_steps+0.0001,stepsize))
    steps = len(iterables[0])
    out = np.zeros([steps**len(iterables),len(iterables)])
    for i,t in enumerate(itertools.product(*iterables)):
        out[i,:] = t
    res = np.zeros([len(v),out.shape[0]])
    for d in range(w.shape[0]):
        #for each dimension, the value is w.t:
        # ie. we add up that dimension of each basis, scaled by the
        # amount we want to travel in the direction of that basis        
        res[d,:]=np.dot(w[d,:],out.T)
        
    #check all vectors are orthogonal to normal
    for i in range(res.shape[1]):
        np.testing.assert_almost_equal(np.dot(res[:,i],v),0)

    #we need to move all the points so they lie on the plane
    #the easiest way to do that is to shift them along one
    #axis, so that v.p + offset = 0
    #res[0,:]-=offset/v[0]
    
    #alternative, this is more likely to put the grid in the
    #right place:
    #the point at the centre of the grid is at
    #[0,0,0,0,0]
    #the nearest point on the plane to point p (which we're trying
    #to perturb) is calculated as 'nearest' which we then use.
    #so we want to shift all the points by nearest
    
    nearest = nearest_point(v,offset,p)[:,None]
    np.testing.assert_almost_equal(np.dot(v,nearest),-offset) #check nearest is on the hyperplane
    res += np.repeat(nearest,res.shape[1],1)
    
    #check all points lie on hyperplane    
    for i in range(res.shape[1]):
        np.testing.assert_almost_equal(np.dot(res[:,i],v),-offset)
    return res
    
    
def total_pixel_change(A,B):
    """Compare matrix of values in A with vector B
    add up the total pixel change"""
    temp = np.repeat(B,A.shape[0],0)
    return np.sum(np.abs(temp-A),1) #sum of absolute change
 #   return np.sum((temp-A)**2,1) #sum-squared


import copy #just to make a copy of the clf
def generate_adversarial_example_near_boundary(m,clf,advXin):
    #we need to find a point that is actually over the boundary, so we head along the normal until the predictions are swapped
    #I found that relying on just nudging over the boundary in latent space
    #is not sufficient; the predictions when fed through the GPLVM transform
    #and back end up on the wrong side of the boundary. Hence empirically finding
    #a good point...
    
    #More detail:
    # I've found the numerical step leads to our assumptions about the values
    #of the predictions to be wrong. Ideally I should be able to search along
    #the boundary for the points with fewest changes, but the boundary turns
    #out not to be in the place I expect.
    #It might be be
    latx = m.infer_newX(advXin)[0].values
    oldpred = get_pred(m,clf,advXin)
    pred = oldpred
    while np.sign(pred)==np.sign(oldpred):
        latx -= clf.coef_*np.sign(oldpred)
        pred = get_pred(m,clf,m.predict(latx)[0])        
    latx -= clf.coef_*np.sign(oldpred) #a little extra nudge over
    
    print(oldpred,pred)
    #option - just return point nearest in latent space;;;
    #return m.predict(latx)[0]
    
    ###TODO: Currently just returning point nearest in latent space (that's just over the boundard)
    #I've found that hunting along the linear plane that should all be on one side
    #doesn't seem to work...    
    intercept = np.dot(clf.coef_/np.sqrt(np.sum(clf.coef_**2)), latx.T)[0]

    #gen a list of points in latent space along the decision boundary, centred on a point
    #that is closest to the seed point passed
    ##latentX = m.infer_newX(advXin)[0].values[0,:]
    latentX = latx[0]
    ps = gen_grid(clf.coef_[0],intercept,latentX,stepsize=5.0,num_steps=10)
    #find the image for each of these latent points
    A = m.predict(ps.T)[0]
    #these are ones that haven't managed to cross the boundary:
    failed = np.sign(get_pred(m,clf,A))==np.sign(oldpred)
    
    
    #find how far each of these is from the original 'seed' point (in pixels)
    diffs = total_pixel_change(A,advXin) 
    diffs[failed[:,0]] = np.Inf #we don't want to select these...
    
    #pick the one with the smallest change
    indx = np.argmin(diffs)
    
    if np.min(diffs)==np.Inf:
        print("Failed: All candidate points remained on wrong side of boundary")
        return None
    originalX = ps[:,indx]
    advX = m.predict(originalX[None,:])[0]
    return advX, m.predict(latx)[0]
    
#### This uses a simple pixel by pixel process to produce the sample...
def get_pred(m,clf,sample):
    """Get the linear classifier prediction
    m = GPLVM model
    clf = linear classifier
    sample = sample to classify"""
    return (np.dot(m.infer_newX(sample)[0].values,clf.coef_.T)+clf.intercept_)
    

def get_numerical_grad(m,clf,advX):
    """
    advX should be of shape 1 x D
    #returns approximation to dpi/dx_i
    """
    eps = 0.00001
   
    pred = get_pred(m,clf,advX)
    delta_advX = advX.repeat(advX.shape[1],0) + np.eye(advX.shape[1])*eps
    delta_preds = get_pred(m,clf,delta_advX)
    return (delta_preds - pred)/eps
        
    
    
def generate_adversarial_example(m,clf,advXin,changedir=0):
    """
    Produce an adversarial example using the 'advX' as the seed
    We can either set pixels high or low. Use:
      changedir = -1 to only set them low
                  +1 to only set them high
                   0 to allow the algorithm to choose.
    The algorithm will try to move the prediction across the boundary and
    stop when it has done so, or return None if it fails.
    """
    listofperturbed = []
    advX = advXin.copy()
    pred = get_pred(m,clf,advXin)
    preddir = np.sign(-pred) #direction to go in
    N = 0
    while (np.sign(pred)!=preddir):
        oldsum = np.sum(advX)
        while np.sum(advX)==oldsum:
            pred = get_pred(m,clf,advX)
            est = get_numerical_grad(m,clf,advX)
            est[listofperturbed] = 0
            if changedir!=0:
                perti = np.argmax(changedir*preddir*est)
                advX[0,perti] = 128+128*changedir
            else:
                perti = np.argmax(np.abs(est))
                advX[0,perti] = 128+128*preddir*np.sign(est[perti])
            pred = get_pred(m,clf,advX)
            if perti in listofperturbed:
                return None, pred, N
            listofperturbed.append(perti)
            print(pred)        
        N+=1
    return advX, pred, N          
            
            
