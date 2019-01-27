Установка через pip:

pip install matplotlib 

pip install scikit-learn

pip install -i https://test.pypi.org/simple/ b64detector-dzh==0.2

Функции:

isb64(s)
Проверяет является ли s (string) b64.
Возвращает True если является, False иначе.

pick_out(s)
Делит s (string) на b64 и не b64.
Возвращает 2 списка - список с b64 и список с остальными словами

pick_out42(s):
Делит s (string) на фрагменты длинной 42 символа и проверяет их. 
Возвращает лист с индексами b64 фрагментов.

delete_clfs()
Удаляет файлы с обученными классификаторами

Обученные классификаторы располагаются в ./trained_clf
