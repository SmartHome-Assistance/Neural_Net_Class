Neural_Net_Class

new_1.py - нейронная сеть - классификатор;
record_words.py - инструмент для записи аудио в формате .wav длиной 2с.
Запись аудио

Необходима для решения задчи создания датасета. До запуска record_words.py в коде нужно указать будущий путь к файлу записи.


WAVE_OUTPUT_FILENAME = 'your_filename.wav'

Также, необходимо убедиться, что у вас установлены все необходимые библиотеки. Если это не так, установить их с помощью команды
pip install имя_библиотеки.
После успешного завершения всех предыдущих шагов можно переходить к запуску.

После запуска программы дождитесь появления текста *record, которое означает, что идёт запись с микрофона. Процесс записи аудио
будет продолжаться до тех пор, пока не появятся слова *done recording.
Использование датасета при обучении нейронной сети и проверка на файле из обучающей выборки

Перед запуском нужно произвести аналогичные записи аудио манипуляции: убедиться в наличии всех необходимых библиотек и указать путь
к файлу проверки.


x_final_vector = wav2mfcc(your_filename_from_dataset.wav)

Также необходимо указать путь к папке с содержимым датасета.


DATA_PATH = "C:/.../dataset_folder/"

Файл new_1.py готов к запуску.
