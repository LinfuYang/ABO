import numpy as np
from demo_GP.AGPR import A_GPR
from demo_GP.func_ND import func_4D_4
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from  sklearn.svm  import SVR
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

list_mse_gp_pre = []
list_mse_gp_con = []
list_mse_svm = []
plt.figure(figsize=(10, 5))
ax_1 = plt.subplot(121)
ax_2 = plt.subplot(122)

A_gpr = A_GPR(f_kernel=None)
func_4d_case_4 = func_4D_4(round_4d=None)
round_xy = func_4d_case_4.round_x

# 测试数据
test_point_mun = 500
x_test = A_gpr.sample_point(func_4d_case_4.round_x, iter=test_point_mun)
y_test = [func_4d_case_4.f_obj(x_test[i, 0], x_test[i, 1], x_test[i, 2], x_test[i, 3]) for i in range(x_test.shape[0])]

lf_point_num = 10
# l-f 数据采集
x_4D_l = A_gpr.sample_point(round_xy=round_xy, iter=lf_point_num)
y_4D_l = np.array([func_4d_case_4.f_l(x_4D_l[i, 0], x_4D_l[i, 1], x_4D_l[i, 3], x_4D_l[i, 3]) for i in
                   range(x_4D_l.shape[0])]).reshape(-1, 1)

left = 20
right = 80
dist = 10
for it in np.array(range(left, right, dist)):
    mean_mse_gp_pre = []
    mean_mse_gp_con = []
    mean_mse_svm = []
    list_w = []
    print('采样点个数为：%s' % str(it))
    for i in range(10):

        hf_point_num = 2
        temp_it = it - hf_point_num


        # h_f 初始化
        x_init = A_gpr.sample_point(round_xy=round_xy, iter=hf_point_num, sample_tpye='linspace')
        y_init = np.array([func_4d_case_4.f_obj(x_init[i, 0], x_init[i, 1], x_init[i, 2], x_init[i, 3]) for i in range(x_init.shape[0])]).reshape(-1, 1)

        hf_gp, list_w_hf = A_gpr.creat_gp_model(max_loop=temp_it, func_nd=func_4d_case_4, x_init_l=x_4D_l, y_init_l=y_4D_l,
                                                x_init_h=x_init, y_init_h=y_init, round_x=round_xy,
                                                n_start=1, n_single=200, min_dist=0.01)
        list_w.append(list_w_hf)


        # 预测模型进行预测
        y_pre_list = [A_gpr.predict_mu_var(np.array(x_test[r]).reshape(1, -1), hf_gp, re_var=False) for r in range(test_point_mun)]

        # 对比试验
        x_sample_point = A_gpr.sample_point(func_4d_case_4.round_x, iter=it)
        y_sample_point = [func_4d_case_4.f_obj(x_sample_point[i, 0], x_sample_point[i, 1], x_sample_point[i, 2], x_sample_point[i, 3]) for i in range(x_sample_point.shape[0])]

        lgb_m = lgb.LGBMRegressor()
        lgb_m.fit(x_sample_point, y_sample_point)
        svr_m = SVR()
        svr_m.fit(x_sample_point, y_sample_point)


        # y_con_list = [A_gpr.predict_mu_var(np.array(x_test[r]).reshape(1, -1), gp_con, re_var=False) for r in range(test_point_mun)]
        y_con_list = [lgb_m.predict(np.array(x_test[r]).reshape(1, -1)) for r in range(test_point_mun)]
        svr_m_list = [svr_m.predict(np.array(x_test[r]).reshape(1, -1)) for r in range(test_point_mun)]
        mean_mse_gp_pre.append(mean_squared_error(y_test, y_pre_list))
        mean_mse_gp_con.append(mean_squared_error(y_test, y_con_list))
        mean_mse_svm.append(mean_squared_error(y_test, svr_m_list))

    list_mse_gp_pre.append(np.mean(mean_mse_gp_pre))
    list_mse_gp_con.append(np.mean(mean_mse_gp_con))
    list_mse_svm.append(np.mean(mean_mse_svm))

    list_average = np.mean(list_w, axis=0)
    plt.sca(ax_1)
    plt.plot(list_average, lw=1.5, label='%s-st' % str(it))

plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('w_hf')
plt.xlabel('iter')
plt.title('4D_case4')

plt.sca(ax_2)
plt.plot(list(range(left, right, dist)), list_mse_gp_con, lw=1.5, label='lgb_m')
plt.plot(list(range(left, right, dist)), list_mse_gp_pre, lw=1.5, label='gp_pre')
plt.plot(list(range(left, right, dist)), list_mse_svm, lw=1.5, label='svr_m')
plt.axis('tight')
plt.legend(loc=0)  # 图例位置自动
plt.ylabel('MSE')
plt.xlabel('iter')
plt.title('4D_case4')

print('pore_model', list_mse_gp_pre)
print('sample_model', list_mse_gp_con)
print('sample_model', list_mse_svm)

plt.show()
