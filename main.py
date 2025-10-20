import cv2
from ultralytics import YOLO
import os


def main():
    """
    Основная функция для определения колва людей на видео.
    Результат выводится в виде готового видеоролика
    и статистики в консоли
    """
    video_file = "crowd.mp4"
    if not os.path.exists(video_file):
        print(f"Файл {video_file} не найден")
        print("Положите видео crowd.mp4 в ту же папку, что и main.py")
        return

    print("Загружаем модель YOLO")
    try:
        model = YOLO('yolov8n.pt')
        print("Модель загружена")
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Не получилось открыть видео")
        return

    # параметры исходника
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = "result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print("Начинаем обработку видео...")
    frame_count = 0
    total_people = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break  # видео закончилось

        results = model.track(
            frame,
            persist=True,
            classes=[0],  # 0 эьо класс человека в YOLO
            conf=0.5,
            verbose=False
        )

        people_in_frame = 0

        # Если найдены какие-то объекты
        if results[0].boxes is not None and results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # Проходимся по всем объектам
            for i in range(len(boxes)):
                class_id = class_ids[i]
                confidence = confidences[i]
                track_id = track_ids[i]
                box = boxes[i]

                if class_id == 0:
                    people_in_frame += 1

                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = (
                        f"People ID: {track_id}"
                        f"conf: {confidence: .2f}"
                    )

                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    text_width, text_height = text_size[0]
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1), (0, 255, 0), -1)

                    cv2.putText(
                        frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                    )

        # поменял на английский из-за ошибки с кодировкой
        cv2.putText(
            frame, f"people in frame: {people_in_frame}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        out.write(frame)

        total_people += people_in_frame
        frame_count += 1

        # прогресс каждые 50 кадров
        if frame_count % 50 == 0:
            print(f"Обработано кадров: {frame_count}")

    cap.release()
    out.release()

    print("\nКонец обработки")
    print(f"Всего обработано кадров: {frame_count}")
    print(f"Всего найдено людей: {total_people}")
    print(f"Результат сохранен в: {output_file}")


if __name__ == "__main__":
    main()
