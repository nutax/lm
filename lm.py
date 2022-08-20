import numpy as np

def lm(xx, y, err = 0.20, alpha = 0.01):
    
    def norm(_x):
        xmin = min(_x)
        rang = max(_x) - xmin
        return (_x - xmin)/rang
    
    #Normalize
    nxx = np.apply_along_axis(norm, 0, xx)
    ny = np.apply_along_axis(norm, 0, y)


    # Add bias
    bias_col = np.ones(nxx.shape[0])
    nxx = np.append(bias_col,nxx, 1)

    print(nxx)
