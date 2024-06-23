import numpy as np
import matplotlib.pyplot as plt


class Trend:

    def __init__(self, trend_type):
        """
        Визначає заданий тип тренду майбутньої вибірки
        :param trend_type: назва типу тренду текстом
        """
        self._trend_type = trend_type
        self._is_linear = 'linear' in trend_type.lower()
        self._is_quadratic = 'quadratic' in trend_type.lower()
        self._is_constant = 'constant' in trend_type.lower()
        self._distribution = None
        self._linear_space = None

        if True not in [self._is_quadratic, self._is_linear, self._is_constant]:
            raise Exception("Допустимі значення тренду: 'Linear', 'Quadratic' та 'Constant'")

    def create_trend(self, min_val, max_val, size, **kwargs):
        """
        Створює вибірку за визначеним трендом та параметрами, передає дані для побудови графіку
        :param min_val: мінімальне значення вибірки
        :param max_val: максимальне значення вибірки
        :param size: об'єм вибірки
        :param kwargs: спеціальні параметри для обраного типу тренду
        """
        self._linear_space = np.linspace(min_val, max_val, size)

        if self._is_linear:
            try:
                slope = kwargs['slope']
                intercept = kwargs['intercept']
            except Exception:
                raise Exception("Переконайтесь що в метод передано значення градієнту 'slope' та перетину 'intercept'")

            self._distribution = slope * self._linear_space + intercept

        elif self._is_quadratic:
            try:
                a = kwargs['a'] # Квадратичний коефіцієнт
                b = kwargs['b'] # Лінійний коефіцієнт
                c = kwargs['c'] # Константа
            except Exception:
                raise Exception("Переконайтесь що в метод передано коефіцієнти 'a', 'b' та константу 'c'")

            self._distribution = a * self._linear_space**2 + b * self._linear_space + c

        elif self._is_constant:
            try:
                c = kwargs['c']  # Константа
            except Exception:
                raise Exception("Переконайтесь що в метод передано константу 'c'")

            self._distribution = np.full(size, c)

    def show_plot(self):
        """
        Відображує графік зміни досліджуваного процесу (тренд)
        """
        trend_name = 'Лінійний' if self._is_linear else 'Постійний' if self._is_constant else 'Квадратичний'
        plt.plot(self._linear_space, self._distribution, label=f'{trend_name} тренд')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(f'{trend_name} тренд')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_trend(self):
        """
        :return: f(x) вибірка відповідно до визначеного тренду у вигляді numpy.array() та тип тренду текстом
        """
        return self._distribution, self._trend_type
