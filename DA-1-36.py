import pandas as pd
#библиотека для работы с таблицами данных
import matplotlib.pyplot as plt
#биб-ка для построения графиков
from sklearn import datasets
#оттуды мы берем готовый датасет для примера

#Взял датасет ирисов Фишера
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
"""
data - это числовые данные 
target - это как сказать бы, целевая переменная, вид ириса
feature_names - названия колонок для числовых данных
"""

#добавили колонку с названием вида ириса к нашей таблице
df['species'] = iris.target
#заменямем цифры на названия видов
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df.head())#посмотрим на первые 5 строк нашей таблицы для корректности 

#ПУНКТ 1.1 из задания (Анализ пропорций)
#Нужно посчитать, сколько каждого вида ирисов
#представлено в датасете и перевести это в доли

#value_counts(normalize=True)- волшебство творит, практически всю работу 
#Она получается считает количество каждого уникального значения,
#а normalize=True преобразует counts в доли

proportions = df['species'].value_counts(normalize=True)
#выводим результат, чтобы посмотретьь что все супер
print(proportions)
print("\nСумма всех долей равна единицк: ",
proportions.sum()) 


#ПУНКТ 1.2 и 1.3 Построение круговой диаграммы 

#Создаем оси и фигуру для графика
plt.figure(figsize=(9, 6))#задали размер фигуры

#теперь строим PLE CHART
plt.pie(proportions, labels=proportions.index, autopct='%1.1f%%', startangle=90)
#proportions - рассчитанные доли в пунтке 1.1
#labels=proportions.index - подписи к секторам, т.е названия видов
#autopct=(...) - это еще одно волшебство которое подписывает проценты на секторах
#startangle - начинает первый сектор с 90 градусов(сверзу)

#Добавляем заголовок
plt.title('Доли различных видов ирисов в датасете')
plt.axis('equal') #чтобы диаграмма была круглой, а не сплюснутой
plt.show()#показываю красоту
#REFERENCES
#1) https://pythonru.com/primery/sklearn-datasets
#2) https://skillbox.ru/media/code/rabotaem-s-pandas-osnovnye-ponyatiya-i-realnye-dannye/
#3) https://skillbox.ru/media/code/biblioteka-matplotlib-dlya-postroeniya-grafikov/