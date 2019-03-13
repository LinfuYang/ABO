import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, RBF
from sklearn.metrics import mean_squared_error
import GPy

def f_h(x):
    '''
    :param x:
    :return:
    '''
    return 2 * x ** 1.2 * np.sin(2 * x) + 2

def f_l(x):
    return 0.7 * f_h(x) + (x ** 1.3 - 0.3) * np.sin(3 * x - 0.5) + 4 * np.cos(2 * x) - 5
con_list = []
'''
for i in range(10):

    x = np.array([np.random.uniform(0, 6) for i in range(15)]).reshape(-1, 1)
    y = np.array([f_l(x[i, 0]) for i in range(x.shape[0])]).reshape(-1, 1)


    k_RBF = GPy.kern.RBF(1, 1, 0.1)
    gp = GPy.models.GPRegression(x, y, kernel=k_RBF)

    gp_model = GaussianProcessRegressor()
    gp_model.fit(x, y)
    gp.optimize(messages=False)

    x_test = np.array([np.random.uniform(0, 6) for i in range(200)]).reshape(-1, 1)
    y_pre_var = [gp.predict(np.array(x_test[i, 0]).reshape(-1, 1))[0][0] for i in range(100)]
    # y_pre_mu, y_pre_var = gp.predict(np.array([3.17271295]).reshape(-1, 1))

    # y_gp_mu, y_gp_var = gp_model.predict(np.array([3.17271295]).reshape(-1, 1), return_std=True)

    # print(np.shape(y_pre_var))
    print(y_pre_mu, end=' ')
    print(y_pre_var)
    print('**********************')

    print(y_gp_mu, end=' ')
    print(y_gp_var)
    print('11111111')

    # y_test = np.array([f(x_test[i, 0]) for i in range(x_test.shape[0])]).reshape(-1, 1)




    # con_list.append(mean_squared_error(y_test, y_pre_var))
# print(np.mean(con_list))

from scipy import stats

print(stats.norm.pdf(-12.97804587, -12.98036902125285, 1.00518202e-05))

stats.norm.pdf(0.1)

print(stats.norm.pdf(0, 0, 0.01))
'''

from scipy.optimize.lbfgsb import fmin_l_bfgs_b

bou = np.array([[-1, 1], [0, 1]])
def f(x):
    x_1 = x[0]
    x_2 = x[1]
    return np.sin(x_1 + x_2) * x_1

print(len(bou))
x0 = [0.2, 0.3]
x_1, f_1, d_1 = fmin_l_bfgs_b(f, x0, bounds=bou, maxfun=1500, approx_grad=True)
print(x_1)
print(f_1)
print(d_1)





'''

def f_d(x_1, x_2):
    return pdist(np.vstack([x_1, x_2]))



dis = [f_d(x[i], y) for i in range(x.shape[0])]

print(dis)


'''
