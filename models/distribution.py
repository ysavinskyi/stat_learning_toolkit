import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon


class Distribution:

    def __init__(self, dist_type):
        """
        Визначає заданий закон розподілу майбутньої вибірки
        :param dist_type: назва закону розподілу текстом
        """
        self._dist_type = dist_type
        self._is_normal = 'normal' in dist_type.lower()
        self._is_uniform = 'uniform' in dist_type.lower()
        self._is_exponential = 'exponential' in dist_type.lower()
        self._distribution = None
        self._density_func = None
        self._linear_space = None
        self._data_dict = {}

        if True not in [self._is_normal, self._is_uniform, self._is_exponential]:
            raise Exception("Допустимі значення закону розподілу: 'Normal', 'Uniform' та 'Exponential'")

    def create_distribution(self, size, **kwargs):
        """
        Створює вибірку за заданими параметрами та законом розподілу, передає дані для побудови графіку
        :param size: об'єм вибірки
        :param kwargs: спеціальні параметри для обраного типу розподілу
        """
        if self._is_normal:
            if kwargs['sigma'] <= 0:
                raise Exception('sigma не може бути менше або дорівнювати 0')

            self._distribution = np.random.normal(kwargs['mu'], kwargs['sigma'], size)
            self._linear_space = np.linspace(min(self._distribution), max(self._distribution), len(self._distribution))
            self._density_func = norm.pdf(self._linear_space, loc=kwargs['mu'], scale=kwargs['sigma'])

        elif self._is_uniform:
            min_value = kwargs['min_val']
            max_value = kwargs['max_val']

            self._distribution = np.random.uniform(min_value, max_value, size)
            self._linear_space = np.linspace(min(self._distribution), max(self._distribution), len(self._distribution))
            self._density_func = uniform.pdf(self._linear_space, loc=min_value, scale=max_value - min_value)

        elif self._is_exponential:
            self._distribution = np.random.exponential(scale=1 / kwargs['lambda_'], size=size)
            self._linear_space = np.linspace(min(self._distribution), max(self._distribution), len(self._distribution))
            self._density_func = expon.pdf(self._linear_space, scale=1 / kwargs['lambda_'])

    def show_plot(self):
        """
        Відображує графік щільності ймовірності відносно до заданого закону розподілу
        """
        if self._is_normal:
            plt.title('Нормальний розподіл ВВ')

        elif self._is_uniform:
            plt.title("Рівномірний розподіл ВВ")

        elif self._is_exponential:
            plt.title('Експоненційний розподіл ВВ')

        plt.plot(self._linear_space, self._density_func, label="Щільність ймовірності $f(x)$")
        plt.hist(self._distribution, bins=30, density=True, alpha=0.5, edgecolor='black', label="Випадкові числа")
        plt.xlabel('Значення $x$')
        plt.ylabel('Щільність ймовірності $f(x)$')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_distribution(self):
        """
        :return: f(x) вибірка відповідно до застосованого закону у вигляді numpy.array() та тип закону розподілу текстом
        """
        return self._distribution, self._dist_type
