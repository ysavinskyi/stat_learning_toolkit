import numpy as np
from bs4 import BeautifulSoup
import requests
import chardet
import pandas as pd


class WebScrapper:

    def __init__(self, url):
        """
        Виконує запит по зазначеному посиланню, декодує текст відповіді
        """
        response = requests.get(url)

        if response.status_code == 200:
            encoding = chardet.detect(response.content)['encoding']
            self._html_content = response.content.decode(encoding)
        else:
            raise Exception(f'Response code: {response.status_code}')

    def get_table(self):
        """
        Виокремлює дані таблиці з HTML коду відповіді та формує її як Pandas Dataframe
        :return: Pandas Dataframe об'єкт з даними таблиці на сайті
        """
        data = {}

        soup = BeautifulSoup(self._html_content, 'html.parser')
        table = soup.find('table', {'class': 'pxtable'})
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 7:
                continue
            data[cells[0].text] = [float(cell.text.replace(',','.')) for cell in cells[1:]]

        dataframe = pd.DataFrame(data).transpose()
        dataframe.columns = ['Народжуваність загальна', 'Смертність загальна',
                             'Народжуваність в міській місцевості', 'Смертність в міській місцевості',
                             'Народжуваність в сільській місцевості', 'Смертність в сільській місцевості']

        return dataframe
