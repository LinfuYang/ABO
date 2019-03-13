import numpy as np
from demo_GP.AGPR import A_GPR
from demo_GP.func_ND import func_1D
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from  sklearn.svm  import SVR
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

list_mse_pre = []
list_mse_lgb = []
list_mse_svm = []
plt.figure(figsize=(10, 5))
ax_1 = plt.subplot(121)
ax_2 = plt.subplot(122)

A_gpr = A_GPR(f_kernel=None)
func_1d = func_1D(round_1d=None)
round_xy = func_1d.round_x

# 测试数据
test_point_mun = 500
x_test = A_gpr.sample_point(func_1d.round_x, iter=test_point_mun)
y_test = [func_1d.f_obj(x_test[i, 0]) for i in range(x_test.shape[0])]

lf_point_num = 10
# l-f 数据采集
x_4D_l = A_gpr.sample_point(round_xy=round_xy, iter=lf_point_num)
y_4D_l = np.array([func_1d.f_l(x_4D_l[i, 0]) for i in
                   range(x_4D_l.shape[0])]).reshape(-1, 1)

left = 15
right = 46
dist = 5
for it in np.array(range(left, right, dist)):
    mean_mse_pre = []
    mean_mse_lgb = []
    mean_mse_svm = []
    list_w = []
    print('采样点个数为：%s' % str(it))
    for i in range(10):

        hf_point_num = 2
        temp_it = it - hf_point_num


        # h_f 初始化
        x_init = A_gpr.sample_point(round_xy=round_xy, iter=hf_point_num)
        y_init = np.array([func_1d.f_obj(x_init[i, 0]) for i in range(x_init.shape[0])]).reshape(-1, 1)

        hf_gp, list_w_hf = A_gpr.creat_gp_model(max_loop=temp_it, func_nd=func_1d, x_init_l=x_4D_l, y_init_l=y_4D_l,
                                                x_init_h=x_init, y_init_h=y_init, round_x=round_xy,
                                                n_start=1, n_single=200, min_dist=0.01)
        list_w.append(list_w_hf)


        # 预测模型进行预测
        y_pre_list = [A_gpr.predict_mu_var(np.array(x_test[r]).reshape(1, -1), hf_gp, re_var=False) for r in range(test_point_mun)]

        # 对比试验
        x_sample_point = A_gpr.sample_point(func_1d.round_x, iter=it)
        y_sample_point = [func_1d.f_obj(x_sample_point[i, 0]) for i in range(x_sample_point.shape[0])]
        lgb_m = lgb.LGBMRegressor()
        lgb_m.fit(x_sample_point, y_sample_point)


        x_sample_point_svr = A_gpr.sample_point(func_1d.round_x, iter=it)
        y_sample_point_svr = [func_1d.f_obj(x_sample_point_svr[i, 0]) for i in range(x_sample_point_svr.shape[0])]
        svr_m = SVR()
        svr_m.fit(x_sample_point_svr, y_sample_point_svr)


        # y_con_list = [A_gpr.predict_mu_var(np.array(x_test[r]).reshape(1, -1), gp_con, re_var=False) for r in range(test_point_mun)]
        y_con_list = [lgb_m.predict(np.array(x_test[r]).reshape(1, -1)) for r in range(test_point_mun)]
        svr_m_list = [svr_m.predict(np.array(x_test[r]).reshape(1, -1)) for r in range(test_point_mun)]
        mean_mse_pre.append(mean_squared_error(y_test, y_pre_list))
        mean_mse_lgb.append(mean_squared_error(y_test, y_con_list))
        mean_mse_svm.append(mean_squared_error(y_test, svr_m_list))

    list_mse_pre.append(np.mean(mean_mse_pre))
    list_mse_lgb.append(np.mean(mean_mse_lgb))
    list_mse_svm.append(np.mean(mean_mse_svm))


    list_average = np.mean(list_w, axis=0)
    plt.sca(ax_1)
    plt.plot(list_average, lw=1.5, label='%s-st' % str(it))

plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('w_hf')
plt.xlabel('iter')
plt.title('1D_case')

plt.sca(ax_2)
plt.plot(list(range(left, right, dist)), list_mse_lgb, lw=1.5, label='lgb_m')
plt.plot(list(range(left, right, dist)), list_mse_pre, lw=1.5, label='gpr_m')
plt.plot(list(range(left, right, dist)), list_mse_svm, lw=1.5, label='svr_m')
plt.axis('tight')
plt.legend(loc=0)  # 图例位置自动
plt.ylabel('MSE')
plt.xlabel('iter')
plt.title('1D_case')

print('lgb_model', list_mse_lgb)
print('gpr_model', list_mse_pre)
print('svm_model', list_mse_svm)

plt.show()
