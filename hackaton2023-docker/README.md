# Docker проект для Хакатона 2023

Для решения задачи отборочного этапа Хакатона 2023 необходимо собрать Docker контейнер с программой. В качестве входных данных файл `input.webm`, содержащий 3 сцены путешествия по дороге. Алгоритм должен выполнить бинарную сегментацию дорожного полотна (определить все пиксели, содержащие дорогу), представить их в виде чёрно-белой маски, где белому цвету соответствует дорога, а чёрному — всё остальное и записать в CSV таблицу с именем файла `output.csv`, содержащий столбцы:

| ImageID | EncodedPixels |
|---|---|
|input.webm_1|255 8 269 5 279 5 438 3 477 3...|
|input.webm_2|265 1 473 2 145 2 154 1 234 1...|

- где ImageID - название видео и номер кадра в виде `input.webm_1234`,
- EncodedPixels - маска дороги, закодированная в RLE (Run-length encoding).

# Установка WORk_DIR
В файле `Makefile` в переменной `WORk_DIR` необходимо указать полный путь из корневой папки до папки, содержащей `input.webm`.
```bash
WORK_DIR := /home/user/hackaton2023
```
Файл с выходными данными будет создан в этой же папке.

## Команды сборки и запуска
Сборка:
```bash
make build
```

Запуск:
```bash
make run
```

Запуск оболочки Bash:
```bash
make shell
```
