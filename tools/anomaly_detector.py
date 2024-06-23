import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm


class AnomalyDetector:
    """
     Презентує функціонал для виявлення аномалій та їх усунення, валідації результатів очищення
    """
    def __init__(self, method):
        """
        Ініціалізує клас
        :param method: назва методу котрий буде використаний для детекції аномалій
        """
        # self._is_lsm_method = 'lsm' in method.lower()
        self._is_custom_method = 'custom' in method.lower()
        self._is_sliding_wind_method = 'sliding_wind' in method.lower()
        self._cleaned_dist = None
        self._found_anomaly_indices = None

    def detect_and_clean(self, dist, wind_size, threshold):
        """
        Застосовує обраний метод детекції та корегує аномальні виміри, збирає дані про індекси аномалій
        для подальшої валідації
        :param dist: вибірка з аномалними значеннями як np.array
        :param wind_size: розмір вікна застосованого для порівняльних операцій по виявленню АВ
        :param threshold: калібраційний параметр для формул розрахунку довірчого інтервалу
        """

        if self._is_sliding_wind_method:
            moving_avg = np.convolve(dist, np.ones(wind_size) / wind_size, mode='valid')

            standard_deviations = np.empty(len(dist) - wind_size + 1)
            for i in range(len(standard_deviations)):
                standard_deviations[i] = np.std(dist[i:i + wind_size])

            anomalies = np.abs(dist[wind_size - 1:] - moving_avg) > threshold * standard_deviations
            self._found_anomaly_indices = np.flatnonzero(anomalies) + (wind_size - 1)

            dist[self._found_anomaly_indices] = moving_avg[anomalies]
            self._cleaned_dist = dist

        if self._is_custom_method:
            peaks, _ = find_peaks(dist)  # Визначаємо піки (максимальні значення) у вибірці
            sorted_peak_indices = np.argsort(dist[peaks])[::-1]  # Сортуємо індекси піків у порядку спадання їх значень

            mean_all = np.mean(dist)  # Обчислюємо середнє значення всієї вибірки
            std_error_all = np.std(dist) / np.sqrt(len(dist))  # Обчислюємо стандартну похибку середнього для всієї вибірки
            standard_score = norm.ppf(1 - (1 - threshold) / 2)  # Визначаємо z-значення для обраного довірчого рівня
            confidence_interval = mean_all + np.array([-1, 1]) * standard_score * std_error_all  # Обчислюємо довірчий інтервал на основі середнього значення

            anomaly_indices = []

            for idx in sorted_peak_indices:  # Проходимо по відсортованих індексах піків
                peak = peaks[idx]  # Отримуємо індекс поточного піка
                window_start = max(0, peak - wind_size)  # Визначаємо початок вікна, забезпечуючи його обмеження від нуля
                window_end = min(len(dist), peak + wind_size + 1)  # Визначаємо кінець вікна, обмежуючи його довжиною вибірки

                # Формуємо дані для вікна, виключаючи сам пік
                window_data = np.concatenate((dist[window_start:peak], dist[peak + 1:window_end]))
                mean_window = np.mean(window_data)  # Обчислюємо середнє значення у вікні

                # Перевіряємо, чи середнє значення вікна потрапляє в довірчий інтервал
                if confidence_interval[0] <= mean_window <= confidence_interval[1]:
                    correction_value = mean_window  # Якщо так, визначаємо значення для корекції
                    anomaly_indices.append(idx)

                    # Проходимо по всіх індексах у вікні та коригуємо значення
                    for j in range(window_start, window_end):
                        dist[j] = (dist[j] + correction_value) / 2  # Замінюємо значення вікна на середнє між поточним та коригувальним значенням

            self._found_anomaly_indices = anomaly_indices
            self._cleaned_dist = dist

    def detection_score(self, true_anomaly_indices):
        """
        Вирахуовує параметри точності виявлення аномалій на основі списку фактичних індексів аномалій переданих ззовні
        :param true_anomaly_indices: список індексів фактичних аномалій вибірки
        """
        found_true_anomalies = [i for i in self._found_anomaly_indices if i in true_anomaly_indices]
        accuracy = len(found_true_anomalies) / len(true_anomaly_indices) * 100
        precision = len(found_true_anomalies) / len(self._found_anomaly_indices) * 100
        print(f'Точність виявлення: {accuracy}\nПрецизійність  виявлення: {precision}')

    def get_distribution(self):
        """
        Повертає очищену від аномальних значень вибірку
        :return: вибірка у форматі np.array
        """
        return self._cleaned_dist
