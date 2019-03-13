import numpy as np
from demo_GP.AGPR import A_GPR
from demo_GP.func_ND import func_4D
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

list_mse_gp_con = []
plt.figure(figsize=(10, 5))
for it in np.array(range(100, 520, 20)):
    mean_mse_gp_con = []
    print('采样点个数为：%s' % str(it))
    for i in range(10):


        A_gpr = A_GPR(f_kernel=None)
        func_4d = func_4D(round_4d=None)
        round_xy = func_4d.round_x
        # 测试数据
        test_point_mun = 500

        x_test = A_gpr.sample_point(func_4d.round_x, iter=test_point_mun)
        y_test = [func_4d.f_obj(x_test[i, 0], x_test[i, 1], x_test[i, 2], x_test[i, 3]) for i in range(x_test.shape[0])]

        # 预测模型进行预测
        # y_pre_list = [A_gpr.predict_mu_var(np.array(x_test[r]).reshape(1, -1), hf_gp, re_var=False) for r in range(test_point_mun)]

        # 对比试验
        x_sample_point = A_gpr.sample_point(func_4d.round_x, iter=it)
        y_sample_point = np.array([func_4d.f_obj(x_sample_point[i, 0], x_sample_point[i, 1], x_sample_point[i, 2], x_sample_point[i, 3]) for i in range(x_sample_point.shape[0])]).reshape(-1, 1)
        gp_con = A_gpr.creat_gpr_model(x_sample_point, y_sample_point)
        y_con_list = [A_gpr.predict_mu_var(np.array(x_test[r]).reshape(1, -1), gp_con, re_var=False) for r in range(test_point_mun)]

        # mean_mse_gp_pre.append(mean_squared_error(y_test, y_pre_list))
        mean_mse_gp_con.append(mean_squared_error(y_test, y_con_list))
    list_mse_gp_con.append(np.mean(mean_mse_gp_con))

plt.plot(list(range(100, 520, 20)), list_mse_gp_con, lw=1.5, label='go_con')

plt.axis('tight')
plt.legend(loc=0) #图例位置自动
plt.ylabel('MSE')
plt.xlabel('iter')
plt.title('4D_case3')

# print('pore_model', list_mse_gp_pre)
print('sample_model', list_mse_gp_con)

plt.show()