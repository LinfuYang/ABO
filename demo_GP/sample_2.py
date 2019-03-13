import numpy as np
from demo_GP.AGPR import A_GPR
from demo_GP.func_ND import func_2D
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')




list_mse_gp_pre = []
list_mse_gp_con = []
plt.figure(figsize=(10, 5))

for it in np.array(range(20, 45, 5)):
    mean_mse_gp_pre = []
    mean_mse_gp_con = []
    list_w = []
    print('采样点个数为：%s' % str(it))
    for i in range(10):
        lf_point_num = 5
        hf_point_num = 2
        temp_it = it - hf_point_num
        pre_list = []
        con_list = []
        A_gpr = A_GPR(f_kernel=None)
        func_2d = func_2D(round_2d=None)
        round_xy = func_2d.round_x

        # l-f 数据采集
        x_2D_l = A_gpr.sample_point(round_xy=round_xy, iter=lf_point_num, sample_tpye='linspace')
        y_2D_l = np.array([func_2d.f_l(x_2D_l[i, 0], x_2D_l[i, 1]) for i in range(x_2D_l.shape[0])]).reshape(-1, 1)

        # h_f 初始化

        x_init = A_gpr.sample_point(round_xy=round_xy, iter=hf_point_num)
        y_init = np.array([func_2d.f_obj(x_init[i, 0], x_init[i, 1]) for i in range(x_init.shape[0])]).reshape(-1, 1)

        hf_gp, list_w_hf = A_gpr.creat_gp_model(max_loop=temp_it, func_nd=func_2d, x_init_l=x_2D_l, y_init_l=y_2D_l,
                                                x_init_h=x_init, y_init_h=y_init, round_x=round_xy, n_start=1, n_single=200)

        list_w.append(list_w_hf)

        test_point_mun = 200
        x_test = A_gpr.sample_point(func_2d.round_x, iter=test_point_mun)
        y_test = [func_2d.f_obj(x_test[i, 0], x_test[i, 1], ) for i in
                  range(x_test.shape[0])]

        # 预测模型进行预测
        y_pre_list = [A_gpr.predict_mu_var(np.array(x_test[r]).reshape(1, -1), hf_gp, re_var=False) for r in
                      range(test_point_mun)]

        # 对比试验
        x_sample_point = A_gpr.sample_point(func_2d.round_x, iter=it)
        y_sample_point = np.array([func_2d.f_obj(x_sample_point[i, 0], x_sample_point[i, 1]) for i in range(x_sample_point.shape[0])]).reshape(-1, 1)
        gp_con = A_gpr.creat_gpr_model(x_sample_point, y_sample_point)

        y_con_list = [A_gpr.predict_mu_var(np.array(x_test[r]).reshape(1, -1), gp_con, re_var=False) for r in range(test_point_mun)]

        mean_mse_gp_pre.append(mean_squared_error(y_test, y_pre_list))
        mean_mse_gp_con.append(mean_squared_error(y_test, y_con_list))

    list_mse_gp_pre.append(np.mean(mean_mse_gp_pre))
    list_mse_gp_con.append(np.mean(mean_mse_gp_con))

    list_average = np.mean(list_w, axis=0)
    plt.plot(list_average, lw=1.5, label='%s-st' % str(it))

    list_average = np.mean(list_w, axis=0)

    plt.plot(list_average, lw=1.5, label='%s-st'% str(it))




plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('w_hf')
plt.xlabel('iter')
plt.title('2D_case')

plt.figure(figsize=(10, 5))
plt.plot(list_mse_gp_con, lw=1.5, label='go_con')
plt.plot(list_mse_gp_pre, lw=1.5, label='go_pre')

plt.axis('tight')

plt.legend(loc=0) #图例位置自动
plt.ylabel('MSE')
plt.xlabel('iter')
plt.title('2D_case')

print('pore_model', list_mse_gp_pre)
print('sample_model', list_mse_gp_con)

plt.show()

