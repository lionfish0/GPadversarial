import GPy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

"""
This classifier separates the data with GPLVM then classifies with a standard GP classifier.
"""

def build_model(X,y):
    """
    """
    q = 2
    kern = GPy.kern.RBF(q, ARD=True)
    m = GPy.models.GPLVM(X, q, kernel=kern)
    m.optimize(messages=1, max_iters=100)
    m2 = GPy.models.GPClassification(m.X.values,y)
    m2.inference_method = GPy.inference.latent_function_inference.laplace.Laplace()
    m2.optimize()
    return m,m2
    
def get_pred(m,m2,sample):
    newX = m.infer_newX(sample)[0].values
    return m2.predict(newX)[0]


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


def generate_adversarial_example(m,m2,advXin):
    """
    Produce an adversarial example using the 'advX' as the seed
    """
    latent_advX = m.infer_newX(advXin)[0].values
    pred, var = m2.predict(latent_advX)
    preddir = np.sign(0.5-pred[0][0]) #direction to go in
    N = 0
    while (np.sign(pred-0.5)!=preddir):
        pred, var = m2.predict(latent_advX)   
        est = calc_df_dx(m2,latent_advX)
        latent_advX += est * 0.01
        #print(latent_advX, pred)
        if N>1000:
            return None, None, pred
        N+=1
    return m.predict(latent_advX)[0], latent_advX, pred
