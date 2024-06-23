import numpy as np
import matplotlib.pyplot as plt


class Predictive:
    """
    Презентує функіонал предиктивної моделі, що базується на МНК
    """
    def __init__(self, dist, third_polynomial_degree=False):
        """
        Ініціалізує клас
        :param dist: вибірка у вигляді np.array
        :param third_polynomial_degree: активує опцію застосування поліному 3 порядку до МНК методу
        """
        self._distribution = dist
        self._third_polynomial_degree = third_polynomial_degree
        self._processed_dist = None
        self._lsm_coefficients = None
        self._feature_matrix = None
        self._get_lsm_coefficients()

    def _get_lsm_coefficients(self):
        """
        Виконується при ініціалізації класу. Застосовує МНК до вибірки класу та збирає поліноміальні коефіцієнти
        Має опцію застосування поліному 3 порядку за змінною self._third_polynomial_degree
        """
        f_matrix_dimension = 4 if self._third_polynomial_degree else 3
        y_matrix = np.zeros((len(self._distribution), 1))
        f_matrix = np.ones((len(self._distribution), f_matrix_dimension))

        for i in range(len(self._distribution)):
            y_matrix[i, 0] = float(self._distribution[i])
            f_matrix[i, 1] = float(i)
            f_matrix[i, 2] = float(i * i)
            if self._third_polynomial_degree:
                f_matrix[i, 3] = float(i * i * i)

        self._feature_matrix = f_matrix
        f_matrix_transposed = f_matrix.T
        feature_matrix = f_matrix_transposed.dot(f_matrix)
        feature_matrix_inverted = np.linalg.inv(feature_matrix)
        self._lsm_coefficients = feature_matrix_inverted.dot(f_matrix_transposed).dot(y_matrix)

    def lsm_extrapolate(self, predict_range):
        """
        Виконує екстраполяцію на встановлений інтервал використовуючи поліноміальні коефіцієнти визначені за МНК
        :param predict_range: довжина інтервалу екстраполяції
        """
        predicted_dist = np.zeros((len(self._distribution) + predict_range, 1))
        c = self._lsm_coefficients

        for i in range(len(self._distribution) + predict_range):
            if self._third_polynomial_degree:
                predicted_dist[i] = c[0] + c[1] * i + (c[2] * i * i + c[3] * i * i * i)
            else:
                predicted_dist[i] = c[0] + c[1] * i + (c[2] * i * i)

        self._processed_dist = predicted_dist

    def lsm_fit(self):
        """
        Виконує згладжування вибірки використовуючи поліноміальні коефіцієнти визначені за МНК
        """
        self._processed_dist = self._feature_matrix.dot(self._lsm_coefficients)

    def r2_score(self):
        """
        Оцінює детермінацію прогнозованих даних відносно оригінальної вибірки
        :return: коефіцієнт детермінації у числовому вигляді
        """
        numerator = 0
        denominator_1 = 0
        for i in range(len(self._processed_dist)):
            numerator += (self._distribution[i] - self._processed_dist[i]) ** 2
            denominator_1 = denominator_1 + self._distribution[i]
        denominator_2 = 0
        for i in range(len(self._processed_dist)):
            denominator_2 = denominator_2 + (self._distribution[i] - (denominator_1 / len(self._processed_dist))) ** 2
        r2_score = 1 - (numerator / denominator_2)
        print('Коефіцієнт детермінації (ймовірність апроксимації)=', r2_score)

        return r2_score

    def show_plot(self, original_trend=None):
        """
        Демонструє графік вибірки та накладену на нього лінію згладженого чи екстрапольованого тренду
        Опціонально приймає вибірку оригінального тренду для порівняння
        :param original_trend: оригінальний тренд у вигляді np.array
        """
        x = np.linspace(0, self._processed_dist.shape[0], self._processed_dist.shape[0])
        additional_range = self._processed_dist.shape[0] - self._distribution.shape[0]
        original_dist = np.append(self._distribution, np.full(additional_range, np.nan))

        plt.clf()
        if original_trend is not None:
            original_trend = np.append(original_trend, np.full(additional_range, np.nan))
            plt.plot(x, original_trend, 'y-', zorder=3, label=f'Оригінальний тренд')
        plt.plot(x, self._processed_dist, 'r-', zorder=2, label=f'Передбачений тренд')
        plt.plot(x, original_dist, zorder=1, label='Адитивна модель')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend()
        plt.grid(True)
        plt.show()
