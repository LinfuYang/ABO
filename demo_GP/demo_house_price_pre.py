from sklearn.datasets import load_boston
from GPy.models import GPRegression
from GPy.kern import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import lightgbm as lgb
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def pre_gp_mu_var(x_new, model, return_var=False):
    if return_var:
        mu, var = model.predict(x_new)
        return mu[0, 0], var[0, 0]
    else:
        mu, _ = model.predict(x_new)
        return mu[0, 0]

x_data, y_data = load_boston(return_X_y=True)
x_data = np.array(x_data)
y_data = np.array(y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
list_con = []
list_lgb = []
list_svr = []
for i in range(1):
    x_train_h, x_tran_l, y_train_h, y_train_l = train_test_split(x_train, y_train, test_size=0.6)

    m, n = np.shape(x_train_h)
    print(m)
    k_rbf = RBF(input_dim=n, variance=1, lengthscale=0.5)

    gp_model = GPRegression(x_train_h, np.reshape(y_train_h, (-1, 1)), kernel=k_rbf)
    gp_model.optimize(messages=False)
    y_pre_con = [pre_gp_mu_var(np.reshape(x_test[i], (1, -1)), gp_model) for i in range(np.shape(x_test)[0])]

    
    model_lgb = lgb.LGBMRegressor()
    model_lgb.fit(x_train_h, y_train_h)
    y_pre_lgb = [model_lgb.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]

    model_svr = SVR()
    model_svr.fit(x_train_h, y_train_h)
    y_pre_svr = [model_svr.predict(np.reshape(x_test[i], (1, -1))) for i in range(np.shape(x_test)[0])]
    # print(y_pre_con)
    list_con.append(mean_squared_error(y_test, y_pre_con))
    list_lgb.append(mean_squared_error(y_test, y_pre_lgb))
    list_svr.append(mean_squared_error(y_test, y_pre_svr))
# print('mse_lgb:', mean_squared_error(y_test, y_pre_lgb))
# print('mse_svr:', mean_squared_error(y_test, y_pre_svr))


print(np.mean(list_con))
print(np.mean(list_lgb))
print(np.mean(list_svr))