import GPy
import numpy as np
import matplotlib.pyplot as plt

"""
There are two different types of GP classifier here.

1) A simple GP classifier, using the Laplace approximation.

2) A GPLVM with a linear classifier ontop.
"""

def build_model(X,y,ARD=False):
    """
    This builds a simple model to perform GP classification.
    
    Feel free to replace with your own GP Classifier.
    
    Note that you are likely to want to use the Laplace inference method
    as this is assumed later (for generating the adv. samples, and in the
    computation of the bounds.
    
    ARD = Automatic Relevance Determination (whether the lengthscales can
    vary, or all have to be equal). By setting it to true, it allows some
    length scales to be much shorter. The effect is to allow single pixels
    to have a greater effect on the result.
    
    X = NxD numpy array
    y = Nx1 numpy array
    """
    k = GPy.kern.RBF(X.shape[1],ARD=ARD)
    k.lengthscale=100.0 #initialise
    m = GPy.models.GPClassification(X,y,k)
    m.inference_method = GPy.inference.latent_function_inference.laplace.Laplace()
    m.optimize(messages=True)
    return m
    
def calc_df_dx(m,advX):
    """
    Calculate the gradient in the latent function wrt the test point (advX)
    """
    assert type(m.kern)==GPy.kern.src.rbf.RBF, "Currently we assume RBF kernel."
    
    dims = advX.shape[1]
    df_dxs = np.zeros(dims)
    kstar = m.kern.K(advX,m.X)
    K = m.kern.K(m.X,m.X)
    for dim in range(dims): #iterate over the dimensions
        if len(m.kern.lengthscale)>1:
            l = m.kern.lengthscale.values[dim]
        else:
            l = m.kern.lengthscale.values
            
        #assumes RBF kernel:
        dk_dx = l**(-2)*(m.X[:,dim]-advX[0,dim])*kstar
        
        df_dk = np.dot(np.linalg.inv(K),m.inference_method.f_hat)
        df_dx = np.dot(dk_dx[0,:].T,df_dk)
        df_dxs[dim] = df_dx
    return df_dxs
    
def get_numerical_pi_grad(m,advX):
    """
    advX should be of shape 1 x D
    #returns approximation to dpi/dx_i
    """
    eps = 0.00001

    pi, _ = m.predict(advX)
    delta_pis, _ = m.predict(np.repeat(advX,advX.shape[1],0) + eps*np.eye(advX.shape[1]))
    return (delta_pis - pi)/eps

def get_numerical_f_grad(m,advX):
    """This computes the gradient of the latent function (we have access to through predict_noiseless) at advX
    adv should be of shape 1 x D
    returns approximation to
    df/dx_i"""
    
    eps = 0.00001

    f, _ = m.predict_noiseless(advX)
    #predict_noiseless returns the latent function, f.
    delta_fs, _ = m.predict_noiseless(np.repeat(advX,advX.shape[1],0) + eps*np.eye(advX.shape[1]))
    return (delta_fs - f)/eps

def generate_adversarial_example(m,advXin,changedir=0,threshold=0):
    """
    Produce an adversarial example using the 'advX' as the seed
    We can either set pixels high or low. Use:
      changedir = -1 to only set them low
                  +1 to only set them high
                   0 to allow the algorithm to choose.
    The algorithm will try to move the prediction across the 50% boundary and
    stop when it has done so, or return None if it fails.
    """
    listofperturbed = []
    advX = advXin.copy()
    pred, var = m.predict(advX)
    prednoiseless, _ = m.predict_noiseless(advX)
    preddir = np.sign(0.5-pred[0][0]) #direction to go in
    N = 0
    #while (np.sign(pred-0.5)!=preddir):
    print(prednoiseless,'*',preddir,'<',threshold,'*',preddir,'=',prednoiseless*preddir,'<',threshold*preddir)
    
    while (prednoiseless*preddir<threshold*preddir):
        print(prednoiseless,'*',preddir,'<',threshold,'*',preddir,'=',prednoiseless*preddir,'<',threshold*preddir)
        if N>advX.shape[1]:
            print("Changed all pixels, but threshold not reached")
            return advX, pred, N, listofperturbed
        oldsum = np.sum(advX)
        while np.sum(advX)==oldsum:
            pred, var = m.predict(advX)   
            est = get_numerical_pi_grad(m, advX)
            est[listofperturbed] = 0

            #perti = np.argmax(est)
            #advX[0,perti] = 0
            print("!")
            if (changedir==-1) or (changedir==1):
                perti = np.argmax(changedir*preddir*est)
                advX[0,perti] = 128+128*changedir
                print(changedir)
            if changedir==0:
                print(np.abs(est))
                perti = np.argmax(np.abs(est))
                advX[0,perti] = 128+128*preddir*np.sign(est[perti])
            if changedir is None:
                stepstocheck = 20
                perti = np.argmax(np.abs(est))
                temp = np.repeat(advX,stepstocheck,0)
                temp[:,perti]=np.linspace(0,255,stepstocheck)
                idx = np.argmax(m.predict(temp)[0]*preddir)
                print(m.predict(temp)[0]*preddir)
                advX[0,perti] = temp[idx,perti]
                print(idx)
            pred, var = m.predict(advX)
            prednoiseless, _ = m.predict_noiseless(advX)
            if perti in listofperturbed:
                print("Changed all pixels, but threshold not reached.")
                return advX, pred, N, listofperturbed
            listofperturbed.append(perti)
        N+=1
        print(prednoiseless,'*',preddir,'<',threshold,'*',preddir,'=',prednoiseless*preddir,'<',threshold*preddir)
    return advX, pred, N, listofperturbed
