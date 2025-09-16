from sklearn import datasets

def load_and_prepare_data():
    """
    Загружает и подготавливает данные ирисов Фишера
    Returns:
        DataFrame: подготовленный датафрейм с данными ирисов
    """
    try:
        # Загружаем датасет ирисов
        iris = datasets.load_iris()
        
        # Создаем DataFrame с числовыми данными
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        
        # Добавляем колонку с видом ириса
        df['species'] = iris.target
        
        # Заменяем цифры на названия видов
        species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        df['species'] = df['species'].map(species_mapping)
        
        return df
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def calculate_proportions(df, column_name='species'):
    """
    Вычисляет пропорции значений в указанной колонке
    
    Args:
        df (DataFrame): исходный датафрейм
        column_name (str): название колонки для анализа
        
    Returns:
        Series: пропорции значений
    """
    try:
        if column_name not in df.columns:
            raise ValueError(f"Колонка '{column_name}' не найдена в датафрейме")
        
        proportions = df[column_name].value_counts(normalize=True)
        return proportions
        
    except Exception as e:
        print(f"Ошибка при вычислении пропорций: {e}")
        return None

def create_pie_chart(proportions, title='Диаграмма пропорций', figsize=(9, 6)):
    """
    Создает круговую диаграмму на основе пропорций
    
    Args:
        proportions (Series): пропорции для отображения
        title (str): заголовок диаграммы
        figsize (tuple): размер фигуры
    """
    try:
        if proportions is None or proportions.empty:
            raise ValueError("Нет данных для построения диаграммы")
        
        # Создаем фигуру и оси
        plt.figure(figsize=figsize)
        
        # Строим круговую диаграмму
        plt.pie(proportions, 
                labels=proportions.index, 
                autopct='%1.1f%%', 
                startangle=90)
        
        # Добавляем заголовок и настраиваем внешний вид
        plt.title(title)
        plt.axis('equal')  # Делаем диаграмму круглой
        
        # Показываем диаграмму
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при построении диаграммы: {e}")

def main():
    """
    Основная функция для выполнения анализа
    """
    #Загружаем и подготавливаем данные
    df = load_and_prepare_data()
    
    if df is None:
        print("Не удалось загрузить данные. Программа завершена.")
        return
    
    #показываем первые строки данных
    print("Первые 5 строк данных:")
    print(df.head())
    print("\n" + "="*50 + "\n")
    
    #вычисляем пропорции видов ирисов
    proportions = calculate_proportions(df, 'species')
    
    if proportions is not None:
        #Выводим результаты вычислений
        print("Пропорции видов ирисов:")
        print(proportions)
        print(f"\nСумма всех долей: {proportions.sum():.2f}")
        print("\n" + "="*50 + "\n")
        
        #Строим круговую диаграмму
        create_pie_chart(proportions, 
                        title='Доли различных видов ирисов в датасете',
                        figsize=(9, 6))

if __name__ == "__main__":
    main()