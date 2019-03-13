import numpy as np
class func_4D:

    def __init__(self, round_4d=None):
        if round_4d == None:
            self.round_x = np.array([[0, 1-(1e-9)], [0, 1-(1e-9)], [0, 1-(1e-9)], [0, 1-(1e-9)]])
        else:
            self.round_x = round_4d


    def f_obj(self, x_1, x_2, x_3, x_4):
        one = (x_1 / 2) * (np.sqrt(1 + (x_2 + x_3 ** 2) * x_4 / x_1 ** 2) - 1)
        two = (x_1 + 3 * x_4) * np.exp(1 + np.sin(x_3))
        return one + two

    def f_l(self, x_1, x_2, x_3, x_4):
        one = (1 + np.sin(x_1) / 10) * self.f_obj(x_1, x_2, x_3, x_4)
        two = -2 * x_1 + x_2 ** 2 + x_3 ** 2 + 0.5
        return one + two

class func_4D_4:

    def __init__(self, round_4d=None):
        if round_4d == None:
            self.round_x = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        else:
            self.round_x = round_4d


    def f_obj(self, x_1, x_2, x_3, x_4):
        one = 2/3 * np.exp(x_1 + x_2)
        two = -1 * x_4 * np.sin(x_3) + x_3
        return one + two

    def f_l(self, x_1, x_2, x_3, x_4):
        one = 1.2 * self.f_obj(x_1, x_2, x_3, x_4) - 1
        return one
class func_2D:
    def __init__(self, round_2d=None):

        if round_2d == None:
            self.round_x = np.array([[0, 1], [0, 1]])
        else:
            self.round_x = round_2d

    def f_obj(self, x_1, x_2):
        if x_2 == 0:
            x_2 = 1e-99

        one = 1 - np.exp(-1 / (2 * x_2))
        two = (2300 * x_1 ** 3 + 1900 * x_1 ** 2 + 2092 * x_1 + 60) / (100 * x_1 ** 3 + 500 * x_1 ** 2 + 4 * x_1 + 20)
        return one * two

    def f_l(self, x1, x2):
        temp = max(0, x2 - 0.05)
        one = (self.f_obj(x1 + 0.05, x2 + 0.05))
        two = self.f_obj(x1 + 0.05, temp)
        three = self.f_obj(x1 - 0.05, x2 + 0.05)
        four = self.f_obj(x1 - 0.05, temp)

        return 1/4 * (one + two) + 1/4 * (three + four)


class func_1D:

    def __init__(self, round_1d=None):
        if round_1d == None:
            self.round_x = np.array([[0, 6]])
        else:
            self.round_x = round_1d

    def f_obj(self, x):
        '''
        :param x:
        :return:
        '''
        return 2 * x ** 1.2 * np.sin(2 * x) + 2

    def f_l(self, x):
        return 0.7 * self.f_obj(x) + (x ** 1.3 - 0.3) * np.sin(3 * x - 0.5) + 4 * np.cos(2 * x) - 5