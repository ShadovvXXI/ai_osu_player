import cv2

def draw_image_with_circle(image, center):
    # радиус круга в пикселях
    radius = 10

    # цвет в формате BGR (чёрный)
    color = (0, 0, 0)

    # залитый круг
    thickness = -1

    cv2.circle(image, center, radius, color, thickness)

    success = cv2.imwrite("result.jpg", image)
    if not success:
        raise IOError("Не удалось сохранить изображение")

def osu_cords_to_window_pos(cords, scale, offset):
    return int(cords[0] * scale + offset[0]), int(cords[1] * scale + offset[1])

def window_pos_to_train_pos(resolution, pos, width, height):
    return int(pos[0] / resolution[0] * width), int(pos[1] / resolution[1] * height)

def approach_time_ms(ar):
    if ar < 5:
        return int(1800 - 120 * ar)
    else:
        return int(1200 - 150 * (ar - 5))