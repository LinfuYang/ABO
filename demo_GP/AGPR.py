import numpy as np
from sklearn.gaussian_process import GaussianProcess, GaussianProcessRegressor, GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared

import GPy
from GPy.kern import RBF

from scipy import stats
from scipy.spatial.distance import pdist
from scipy.optimize import fmin_l_bfgs_b
class A_GPR:

    def __init__(self, f_kernel=None):
        if f_kernel != None:
            self.kernel_f = f_kernel

    def creat_gpr_model(self, x_data, y_data):
        '''
        :param x_data:  初始化的自变量
        :param y_data:  初始化的函数值
        :return: 新建的模型
        '''
        # gp = GaussianProcessRegressor(normalize_y=True)
        # gp.fit(x_data, y_data)
        m, n = np.shape(x_data)
        k_RBF = GPy.kern.RBF(input_dim=n, variance=1, lengthscale=0.5)
        gp = GPy.models.GPRegression(x_data, y_data, kernel=k_RBF)
        gp.optimize(messages=False)
        return gp

    def sample_point(self, round_xy=None, iter=None, sample_tpye='uniform'):
        m, n = np.shape(round_xy)
        x_temp = np.zeros((iter, m))
        if sample_tpye == 'uniform':
            for k in range(iter):
                for i in range(m):
                    x_temp[k, i] = np.random.uniform(round_xy[i, 0]+10e-99, round_xy[i, 1])

        elif sample_tpye == 'linspace':
            for i in range(m):

                x_temp[:, i] = np.linspace(round_xy[i, 0]+10e-99, round_xy[i, 1], iter)

        return x_temp

    def predict_mu_var(self, x_new, model_gp, re_var=True):
        '''
        :param x_new: 需要做出预测的点
        :param x_lf: 已知低质量点
        :param y_lf: 低质量点的对应函数值
        :return: 关于x_new的预测值 均值和方差
        '''
        # 求L-F dataset 的GP 均值 和  协方差矩阵
        # print(gp_lf._y_train_mean)
        # print(gp_lf.tianjia_k)
        # 输出任意自变量点的均值和标准差
        if re_var:

            # y_mean_lf, y_std_lf = model_gp.predict(x_new, return_std=re_var)
            y_mean_lf, y_var_lf = model_gp.predict(x_new)
            # print(y_mean_lf)
            # print(y_std_lf)
            return y_mean_lf[0][0], y_var_lf[0][0]
        else:
            # y_mean_lf = model_gp.predict(x_new, return_std=re_var)
            y_mean_lf, _ = model_gp.predict(x_new)
            return y_mean_lf[0][0]

    def find_next_point(self, model_gp_hf=None, mode_gp_lf=None, lf_w=None, hf_w=None, round_1D=None, restar_iter=20, single_iter=200):
        '''
        :param model_gp:
        :param lf_w:
        :param hf_w:
        :param restar_iter:
        :param single_iter:
        :return:
        '''

        cond_x = []
        cond_y = []
        for i in range(restar_iter):
            point_array = self.sample_point(round_1D, iter=single_iter)

            temp_best = []
            for j in range(single_iter):
                if len(point_array[j]) == 1:
                    x_temp = np.array(point_array[j]).reshape(-1, 1)
                else:
                    x_temp = np.array(point_array[j]).reshape(1, -1)
                lf_mu, lf_var = self.predict_mu_var(x_temp, mode_gp_lf)
                hf_mu, hf_var = self.predict_mu_var(x_temp, model_gp_hf)
                p_1 = hf_var ** (-1)
                p_2 = lf_var ** (-1)
                post_mu = (hf_mu * hf_w * p_1 + lf_mu * lf_w * p_2) / (hf_w * p_1 + lf_w * p_2)
                post_cov = (hf_w * p_1 + lf_w * p_2) ** (-1)

                # 为找的下一个合适的采样点：
                f = 2 * post_cov ** 0.5
                temp_best.append(f)
            # print('每次迭代的结果', temp_best)
            s_best_index = np.argmax(temp_best)

            cond_y.append(temp_best[s_best_index])
            cond_x.append(point_array[s_best_index])
        # print('历次迭代的最佳值', cond_y)

        f_best_index = np.argmax(cond_y)

        if len(cond_x[f_best_index]) == 1:
            next_point = np.array(cond_x[f_best_index]).reshape(-1, 1)
        else:
            next_point = np.array(cond_x[f_best_index]).reshape(1, -1)
        # print('next_point:', next_point, end=' ')
        # print('max_y:', cond_y[f_best_index])
        return next_point
    def find_next_point_3(self, model_gp_hf=None, mode_gp_lf=None, lf_w=None, hf_w=None, round_1D=None, restar_iter=20, single_iter=200):
        '''
        :param model_gp:
        :param lf_w:
        :param hf_w:
        :param restar_iter:
        :param single_iter:
        :return:
        '''
        def f_data(x):
            x_temp = np.array(x, ndmin=2)
            lf_mu, lf_var = self.predict_mu_var(x_temp, mode_gp_lf)
            hf_mu, hf_var = self.predict_mu_var(x_temp, model_gp_hf)
            p_1 = hf_var ** (-1)
            p_2 = lf_var ** (-1)
            post_mu = (hf_mu * hf_w * p_1 + lf_mu * lf_w * p_2) / (hf_w * p_1 + lf_w * p_2)
            post_cov = (hf_w * p_1 + lf_w * p_2) ** (-1)
            # 为找的下一个合适的采样点：
            return -2 * post_cov ** 0.5
        x0 = [0.0, 0.5, 0.5, 0.5]

        x_1, f_2, d_2 = fmin_l_bfgs_b(f_data, x0, bounds=round_1D, maxfun=1500, approx_grad=True)
        print('next_point_x:', x_1, end=' ')
        print('the un confis:',  -1 *f_2)

        return np.array(x_1, ndmin=2)
    def find_next_point_2(self, model_gp_hf=None, mode_gp_lf=None, lf_w=None, hf_w=None, round_1D=None, x_init=None, single_iter=1000):
        '''
        :param model_gp:
        :param lf_w:
        :param hf_w:
        :param restar_iter:
        :param single_iter:
        :return:
        '''

        distance_arr = np.zeros((x_init.shape[0], single_iter))
        print(np.shape(distance_arr))
        point_array = self.sample_point(round_1D, iter=single_iter)

        for i in range(x_init.shape[0]):
            for j in range(single_iter):
                distance_arr[i, j] = self.Eu_dist(x_init[i], point_array[j])
        best_point = np.argmax(distance_arr)
        # print(distance_arr)
        row = best_point // single_iter
        col = best_point - row * single_iter

        next_point = np.array(point_array[col], ndmin=2)


        '''
                lf_mu, lf_var = self.predict_mu_var(x_temp, mode_gp_lf)
                hf_mu, hf_var = self.predict_mu_var(x_temp, model_gp_hf)
                p_1 = hf_var ** (-1)
                p_2 = lf_var ** (-1)
                post_mu = (hf_mu * hf_w * p_1 + lf_mu * lf_w * p_2) / (hf_w * p_1 + lf_w * p_2)
                post_cov = (hf_w * p_1 + lf_w * p_2) ** (-1)

                # 为找的下一个合适的采样点：
                f = 2 * post_cov ** 0.5
                temp_best.append(f)
            # print('每次迭代的结果', temp_best)
            s_best_index = np.argmax(temp_best)

            cond_y.append(temp_best[s_best_index])
            cond_x.append(point_array[s_best_index])
        # print('历次迭代的最佳值', cond_y)

        f_best_index = np.argmax(cond_y)

        if len(cond_x[f_best_index]) == 1:
            next_point = np.array(cond_x[f_best_index]).reshape(-1, 1)
        else:
            next_point = np.array(cond_x[f_best_index]).reshape(1, -1)
        '''

        return next_point


    def like_hood_func(self, y_pre, mu, var):
        one = pow((2 * np.pi * var), 0.5)
        two = np.exp(-1 * (y_pre - mu) ** 2 / (2 * var)) + 10e-6

        return 1 / one * two

    def Eu_dist(self, x_1, x_2):

        return pdist(np.vstack([x_1, x_2]))

    def creat_gp_model(self, max_loop=15, func_nd=None,  x_init_l=None, y_init_l=None, x_init_h=None,
                       y_init_h=None, round_x=None, n_start = 1, n_single=200, min_dist=0.02):
        '''

        :param max_loop:
        :param func_nd:
        :param a_gpr:
        :param x_init_l:
        :param y_init_l:
        :param x_init_h:
        :param y_init_h:
        :param round_x:
        :return:
        '''
        # 初始化模型
        lf_gp = self.creat_gpr_model(x_init_l, y_init_l)
        hf_gp = self.creat_gpr_model(x_init_h, y_init_h)

        # 找最小值
        # index_p = np.argmax(y_init_h)
        # max_point_y = y_init_h[index_p]
        # max_point_x = x_init_h[index_p]
        # 定义初始权重
        w_lf = 0.5
        w_hf = 1 - w_lf
        list_w_hf = list([w_hf])

        for it in range(max_loop):

            next_point_x = self.find_next_point(model_gp_hf=hf_gp, mode_gp_lf=lf_gp, lf_w=w_lf, hf_w=w_hf,
                                                 round_1D=round_x, restar_iter=n_start, single_iter=n_single)

            # 检查该点是否已存在于历史数据中
            flag = True
            while flag:
                f_flag=0
                for l_t in x_init_h:
                    if list(next_point_x[0]) == list(l_t):
                        print(next_point_x[0])
                        print(l_t)
                        f_flag = 1
                        print('重复了')
                        break
                if f_flag == 1:
                        next_point_x = self.find_next_point(model_gp_hf=hf_gp, mode_gp_lf=lf_gp, lf_w=w_lf, hf_w=w_hf,
                                                            round_1D=round_x, restar_iter=n_start, single_iter=n_single)
                else:
                    flag= False
            # print('it:', it)

            # 计算w_lf的先验预估
            w_lf = w_lf**0.9 / (w_lf**0.9 + (1 - w_lf)**0.9)
            w_hf = 1 - w_lf
            # 计算似然
            if next_point_x.shape[1] == 1:
                y_pre_next_point = np.array([func_nd.f_obj(next_point_x[r, 0]) for r in range(next_point_x.shape[0])]).reshape(-1, 1)
            elif next_point_x.shape[1] == 2:
                y_pre_next_point = np.array(
                    [func_nd.f_obj(next_point_x[r, 0], next_point_x[r, 1]) for r in range(next_point_x.shape[0])]).reshape(-1, 1)
            elif next_point_x.shape[1] == 4:
                y_pre_next_point = np.array(
                    [func_nd.f_obj(next_point_x[r, 0], next_point_x[r, 1], next_point_x[r, 2], next_point_x[r, 3]) for r in
                     range(next_point_x.shape[0])]).reshape(-1, 1)
            # print('next_point_x:', next_point_x, end=' ')
            # print('y_pre_next_point:', y_pre_next_point)

            mu_next_point_lf, var_next_point_lf = self.predict_mu_var(next_point_x, lf_gp)
            mu_next_point_hf, var_next_point_hf = self.predict_mu_var(next_point_x, hf_gp)
            # print('mu_next_point_lf:', mu_next_point_lf, end=' ')
            # print('cov_next_point_lf:', var_next_point_lf)
            # print('mu_next_point_hf:', mu_next_point_hf, end=' ')
            # print('cov_next_point_hf:', var_next_point_hf)

            like_hood_lf = round(stats.norm.pdf(y_pre_next_point[0, 0], mu_next_point_lf, var_next_point_lf ** 0.5), 10) + 10e-10
            like_hood_hf = round(stats.norm.pdf(y_pre_next_point[0, 0], mu_next_point_hf, var_next_point_hf ** 0.5), 10) + 10e-10
            print('like_hood_lf:', like_hood_lf, end=' ')
            print('like_hood_hf:', like_hood_hf)
            # like_hood_lf_1 = a_gpr.like_hood_func(y_pre_next_point, mu_next_point_lf, var_next_point_lf)
            # like_hood_hf_1 = a_gpr.like_hood_func(y_pre_next_point, mu_next_point_hf, var_next_point_hf)
            # print('like_hood_lf_1:', like_hood_lf_1, end=' ')
            # print('like_hood_hf_1:', like_hood_hf_1)

            dist_list = [self.Eu_dist(x_init_h[k], next_point_x) for k in range(x_init_h.shape[0])]
            # dist_list_l = [self.Eu_dist(x_init_l[k], next_point_x) for k in range(x_init_l.shape[0])]
            # dist_list_l = [self.Eu_dist(x_init_l[k], next_point_x) for k in range(x_init_l.shape[0])]
            print('the min diatance', np.min(dist_list))
            # print('the min diatance _l', np.min(dist_list_l))
            #print('the mean diatance', np.mean(dist_list))
            # print('the min diatance x_nit_l:', np.min(dist_list_l))
            if np.min(dist_list) > min_dist:
                w_lf_1 = w_lf * like_hood_lf / (w_lf * like_hood_lf + w_hf * like_hood_hf)

                w_lf = w_lf_1
                # max_point_y = y_pre_next_point
                # max_point_x = next_point_x
            else:
                w_lf = w_lf
            w_hf = 1 - w_lf
            print('w_lf:', w_lf, end=' ')
            print('w_hf:', w_hf)
            print('****************************')
            list_w_hf.append(w_hf)

            x_init_h = np.r_[x_init_h, next_point_x]
            y_init_h = np.r_[y_init_h, y_pre_next_point]

            hf_gp = self.creat_gpr_model(x_init_h, y_init_h)

        return hf_gp, list_w_hf