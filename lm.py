# Imports
import numpy as np
from pipe import select


# lm: find best linear predictor, get training evolution and test results
def lm(xx, y, err = 0.20, alpha = 0.01, test_p = 0.3, seed = 2001):

    # Create normalization parameters
    xxmin, xxmax = np.apply_along_axis(min, 0, xx), np.apply_along_axis(max, 0, xx)
    ymin, ymax = min(y), max(y)



    # Normalize
    def norm_col(x, xmin, xmax):
        return (x - xmin)/(xmax-xmin) if xmax > xmin else x**0

    def norm_cols(_xx, _xxmin, _xxmax):
        return np.array(list(
            zip(_xx.T, _xxmin, _xxmax) | select(lambda x : norm_col(*x))
        )).T
    
    nxx = norm_cols(xx, xxmin, xxmax)
    ny = norm_col(y, ymin, ymax)



    # Add bias
    def add_bias(_xx):
        bias_col = np.ones((_xx.shape[0],1))
        return np.append(bias_col,_xx, 1)
    
    nxx = add_bias(nxx)



    # Split train and test data
    def split_train_test(_xx, _y):
        m = _xx.shape[0]
        train_p = int(m*(1-test_p))
        indices = np.random.permutation(m)
        test_i, train_i = indices[:train_p], indices[train_p:]
        _train_xx, _test_xx = _xx[train_i,:], _xx[test_i,:]
        _train_y, _test_y = _y[train_i,:], _y[test_i,:]
        return _train_xx, _train_y, _test_y, _test_xx
    
    train_xx, train_y, test_y, test_xx = split_train_test(nxx, ny)



    # Difference function
    def difference(_xx, _y, _w):
        return (_y.T - np.matmul(_xx,_w))[0]
    
    # Loss function
    def average_error(_dif):
        return sum(_dif**2)/(_dif.shape[0] * 2)
    
    # Derivate loss function
    def delta(_dif, x):
        return sum(-x * _dif)/_dif.shape[0]
    
    # Train
    def train(_xx, _y, _err, _alpha, _seed):
        np.random.seed(_seed)
        w = np.random.rand(_xx.shape[1])
        loss = []
        epochs = []

        while True:
            dif = difference(_xx, _y, w)
            avg_err = average_error(dif)

            if avg_err <= _err:
                break

            loss.append(loss)
            epochs.append(w)

            dw = np.apply_along_axis(lambda x: delta(dif, x), 0, _xx)
            w = w - _alpha*dw

        return w, loss, epochs
    
    w, loss, epochs = train(train_xx, train_y, err, alpha, seed)
    train_evolution = {'loss': loss, 'epochs': epochs}



    # Test
    def test(_xx, _y, _w):
        return average_error(difference(_xx, _y, _w))
    
    test_error = test(test_xx, test_y, w)



    # Final product: predict function
    def predict(_xx):
        _nxx = norm_cols(_xx, xxmin, xxmax)
        _nxx = add_bias(_nxx)
        _ny = np.matmul(_nxx, w)
        return _ny*(ymax - ymin) + ymin

    return predict, train_evolution, test_error
