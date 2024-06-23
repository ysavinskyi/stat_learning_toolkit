import numpy as np
import matplotlib.pyplot as plt


class Additive:

    def __init__(self, trend, rand_distr):
        """
        Приймає дві вибірки та об'єднує їх в адитивну модель
        :param trend: вибірка котру вважаємо трендом
        :param rand_distr: вибірка котру вважаємо похибкою
        """
        if type(rand_distr) is tuple:
            self._rand_distr = rand_distr[0]
            self._rand_distr_type = rand_distr[1].lower()
        else:
            self._rand_distr = rand_distr
            self._rand_distr_type = 'misc'

        self._trend = trend[0]
        self._trend_type = trend[1].lower()
        self._additive_model = self._trend + self._rand_distr

    def show_plot(self):
        """
        Виводить графік тренду та графік адитивної моделі для порівняння
        """
        trend_names = {
            'linear': 'Лінійн',
            'quadratic': 'Квадратич',
            'constant': 'Постійн',
        }
        distr_names = {
            'exponential': 'Експоненційн',
            'uniform': 'Рівномірн',
            'normal': 'Нормальн',
            'misc': 'Довільн'
        }

        x = np.linspace(0, self._additive_model.shape[0], self._additive_model.shape[0])

        plt.clf()
        plt.plot(x, self._trend, 'r-', zorder=2, label=f'{trend_names[self._trend_type]}ий тренд')
        plt.plot(x, self._additive_model, zorder=1, label='Адитивна модель')
        plt.title(f'Адитивна Модель з {trend_names[self._trend_type]}им Трендом'
                  f' і {distr_names[self._rand_distr_type]}ою Похибкою')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_additive_model(self):
        """
        :return: масив Numpy з даними адитивної моделі
        """
        return self._additive_model
