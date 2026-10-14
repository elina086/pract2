import os
import glob
from ultralytics import YOLO
from PIL import Image

# Загружаем модели для детекции и сегментации
print("Загрузка моделей...")
model_det = YOLO('yolov8n.pt')      # Модель для детекции (рамки)
model_seg = YOLO('yolov8n-seg.pt')  # Модель для сегментации (маски)

# Создаем папки для результатов инференса по каждому кадру
os.makedirs('output_det', exist_ok=True)
os.makedirs('output_seg', exist_ok=True)

# Ищем наши нарезанные кадры
frame_paths = sorted(glob.glob('frames/*.jpg'))

if not frame_paths:
    print("Ошибка: кадры не найдены в папке frames!")
else:
    det_images = []
    seg_images = []

    print("Обработка кадров...")
    for path in frame_paths:
        filename = os.path.basename(path)
        
        # 1. Детекция объектов
        res_det = model_det(path, verbose=False)[0]
        img_det = Image.fromarray(res_det.plot()[..., ::-1])
        img_det.save(os.path.join('output_det', filename))
        det_images.append(img_det)

        # 2. Сегментация
        res_seg = model_seg(path, verbose=False)[0]
        img_seg = Image.fromarray(res_seg.plot()[..., ::-1])
        img_seg.save(os.path.join('output_seg', filename))
        seg_images.append(img_seg)

    # Собираем GIF №1 (Детекция)
    if det_images:
        det_images[0].save('detection_result.gif', save_all=True, append_images=det_images[1:], duration=330, loop=0)
        print("Создан файл: detection_result.gif")

    # Собираем GIF №2 (Сегментация)
    if seg_images:
        seg_images[0].save('segmentation_result.gif', save_all=True, append_images=seg_images[1:], duration=330, loop=0)
        print("Создан файл: segmentation_result.gif")

    print("Готово! Все задания выполнены.")