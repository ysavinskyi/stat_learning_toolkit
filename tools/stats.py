import numpy as np
import matplotlib.pyplot as plt


class Statistic:

    def __init__(self, model):
        """
        Ініціалізує переданий масив даних як внутрішній параметр класу
        """
        self._model = model
        self._stats_dict = dict()

    def _get_stats(self, ):
        """
        Проводить вирахування параметрів вибірки та записує їх в словник self._stats_dict
        """
        self._stats_dict['Матемтичне очікування'] = np.mean(self._model)
        self._stats_dict['Дисперсія'] = np.var(self._model)
        self._stats_dict['Стандартне відхилення'] = np.std(self._model)

    def print_stats(self):
        """
        Виводить дані self._stats_dict словника у вигляді ліній тексту
        """
        self._get_stats()
        for stat, value in self._stats_dict.items():
            print(f'\n{stat}: {value}')

    def show_histogram(self):
        """
        Відображує гістограму розподілу величин в вибірці
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self._model, bins=30, color='blue', alpha=0.7, label='Розподіл ВВ')
        plt.title('Гістограма закону розподілу')
        plt.xlabel('Значення')
        plt.ylabel('Частота')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_plot(self):
        """
        Відображує графік зміни тренду вибірки
        """
        linear_space = np.linspace(0, len(self._model), len(self._model))
        plt.plot(linear_space, self._model, label="Тренд")
        plt.title('Графік зміни тренду')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.legend()
        plt.grid(True)
        plt.show()
