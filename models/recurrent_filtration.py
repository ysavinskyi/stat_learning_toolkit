import numpy as np
import matplotlib.pyplot as plt


class RecurrentFilter:
    """
    Виконує екстраполяцію на основі методу рекурентної фільтрації
    """
    def __init__(self, filter_type):
        """
        Ініціалізує клас
        :param filter_type: тип виокристованого фільтру
        """
        self._is_alpha_beta = 'alpha-beta' == filter_type.lower()
        self._is_alpha_beta_gamma = 'alpha-beta-gamma' == filter_type.lower()
        self._original_dist = None
        self._processed_dist = None

    def process(self, dist):
        """
        Запускає ітерацію фільтру за кожним значенням переданої вибірки і збирає екстрапольовані результати
        :param dist: тренд змінюваного процесу в форматі np.array
        """
        self._original_dist = dist

        if self._is_alpha_beta:
            self._alpha_beta_filter()
        elif self._is_alpha_beta_gamma:
            self._alpha_beta_gamma_filter()

    def _alpha_beta_filter(self):
        """
        Реалізація альфа-бета фільтру
        """
        Yin = np.zeros((len(self._original_dist), 1))
        YoutAB = np.zeros((len(self._original_dist), 1))
        T0 = 1
        for i in range(len(self._original_dist)):
            Yin[i, 0] = float(self._original_dist[i])

        Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
        Yextra = Yin[0, 0] + Yspeed_retro
        alpha = self._update_alpha(1)
        beta = self._update_beta(1)
        YoutAB[0, 0] = Yin[0, 0] + alpha * (Yin[0, 0])

        for i in range(1, len(self._original_dist)):
            error = Yin[i, 0] - Yextra

            YoutAB[i, 0] = Yextra + alpha * error
            Yspeed = Yspeed_retro + (beta / T0) * error
            Yspeed_retro = Yspeed
            Yextra = YoutAB[i, 0] + Yspeed_retro
            alpha = self._update_alpha(i)
            beta = self._update_beta(i)

        self._processed_dist = YoutAB

    def _alpha_beta_gamma_filter(self):
        """
        Реалізація альфа-бета-гамма фільтру що адаптується на основі точністі кожного передбачення
        """
        Yin = np.zeros((len(self._original_dist), 1))
        YoutABG = np.zeros((len(self._original_dist), 1))
        T0 = 1
        for i in range(len(self._original_dist)):
            Yin[i, 0] = float(self._original_dist[i])

        Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
        Yaccel_retro = 0  # Початкове прискорення
        Yextra = Yin[0, 0] + Yspeed_retro

        base_alpha = self._update_alpha(1)
        beta = self._update_beta(1)
        gamma = 0.001

        YoutABG[0, 0] = Yin[0, 0] + base_alpha * (Yin[0, 0])

        for i in range(1, len(self._original_dist)):
            error = Yin[i, 0] - Yextra

            # Проста адаптація параметрів на основі помилки
            accuracy_penalty = 1.0 / (1.0 + abs(error))

            alpha = base_alpha * accuracy_penalty
            beta = beta * accuracy_penalty
            gamma = gamma * accuracy_penalty

            YoutABG[i, 0] = Yextra + alpha * error
            Yspeed = Yspeed_retro + (beta / T0) * error
            Yaccel = Yaccel_retro + (gamma / T0) * error

            Yspeed_retro = Yspeed
            Yaccel_retro = Yaccel

            Yextra = YoutABG[i, 0] + Yspeed + 0.5 * Yaccel * T0

        self._processed_dist = YoutABG

    @staticmethod
    def _update_alpha(n):
        return (2 * (2 * n - 1)) / (n * (n + 1))

    @staticmethod
    def _update_beta(n):
        return 6 / (n * (n + 1))

    def get_processed_dist(self):
        """
        Повертає опрацьовану вибірку як np.array
        :return:
        """
        return self._processed_dist

    def show_plot(self):
        """
        Демонструє графік з оригінальною лінією тренду та екстрапольованою рекурентним фільтром
        """
        x = np.linspace(0, self._processed_dist.shape[0], self._processed_dist.shape[0])

        plt.clf()
        plt.plot(x, self._original_dist, 'y-', zorder=3, label=f'Оригінальний тренд')
        plt.plot(x, self._processed_dist, 'r-', zorder=2, label=f'Передбачений тренд')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend()
        plt.grid(True)
        plt.show()
