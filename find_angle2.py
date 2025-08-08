import cv2
import numpy as np
from loguru import logger


class CircleDetector:
    def __init__(self):
        # Параметры детектирования окружностей
        self.circle_params = {
            'min_dist': 20,
            'param1': 50,
            'param2_small': 10,
            'param2_large': 30,
            'min_radius_small': 20,
            'max_radius_small': 100,
            'min_radius_large': 50,
            'max_radius_large': 300,
            'max_small_circles': 1,
            'max_large_circles': 1
        }

        # Параметры для определения угла
        self.angle_params = {
            'crop_margin': 0.5,
            'text_threshold': 0.3,
            'num_detail': 1,
            'eps': 5  # Допустимая погрешность угла
        }

        self.center = None
        self.min_angle = None
        self.small_circles = []
        self.large_circles = []

        # Эталонные данные для углов текста по номеру детали (как в коде 1)
        self.reference_text_angles = {
            1: [
                {"name": "text_brand", "center_angle": 153.75},
                {"name": "text_model", "center_angle": 267.75},
                {"name": "text_number", "center_angle": 26.5}
            ],
            2: [
                {"name": "text_brand", "center_angle": 153.75},
                {"name": "text_model", "center_angle": 267.75},
                {"name": "text_number", "center_angle": 26.5}
            ],
            3: [
                {"name": "text_brand", "center_angle": 46.25},
                {"name": "text_model", "center_angle": 219.75}
            ],
            4: [],
            5: []
        }

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def detect_all_circles(self, gray_img):
        # Детектирование маленьких окружностей
        small_circles = cv2.HoughCircles(
            gray_img,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.circle_params['min_dist'],
            param1=self.circle_params['param1'],
            param2=self.circle_params['param2_small'],
            minRadius=self.circle_params['min_radius_small'],
            maxRadius=self.circle_params['max_radius_small']
        )

        # Детектирование больших окружностей
        large_circles = cv2.HoughCircles(
            gray_img,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.circle_params['min_dist'],
            param1=self.circle_params['param1'],
            param2=self.circle_params['param2_large'],
            minRadius=self.circle_params['min_radius_large'],
            maxRadius=self.circle_params['max_radius_large']
        )

        # Фильтрация и ограничение количества кругов
        self.small_circles = small_circles[0][
                             :self.circle_params['max_small_circles']] if small_circles is not None else []
        self.large_circles = large_circles[0][
                             :self.circle_params['max_large_circles']] if large_circles is not None else []

        return self.small_circles, self.large_circles

    def draw_circles(self, img, circles, color, thickness=2):
        if circles is None or len(circles) == 0:
            return img

        for (x, y, r) in circles:
            center = (int(x), int(y))
            radius = int(r)
            cv2.circle(img, center, radius, color, thickness)
            cv2.circle(img, center, 3, (0, 0, 255), -1)
        return img

    def find_main_contour(self, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def auto_crop_image(self, img, contour):
        x, y, w, h = cv2.boundingRect(contour)
        margin = int(max(w, h) * self.angle_params['crop_margin'])
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        return img[y:y + h, x:x + w]

    def calculate_rotation_angle(self, img, contour):
        try:
            cropped = self.auto_crop_image(img, contour)

            (cx, cy), _ = cv2.minEnclosingCircle(contour)
            self.center = (int(cx), int(cy))

            maxR = int(cropped.shape[0] * 0.7)
            polar = cv2.linearPolar(cropped, self.center, maxR, cv2.INTER_LINEAR)
            polar_rotated = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

            gray_polar = cv2.cvtColor(polar_rotated, cv2.COLOR_BGR2GRAY)
            _, text_mask = cv2.threshold(gray_polar, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel_h = np.ones((1, 15), np.uint8)
            clean_text = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel_h)

            y_proj = np.sum(clean_text, axis=0)
            threshold = np.max(y_proj) * self.angle_params['text_threshold']
            text_regions_idx = np.where(y_proj > threshold)[0]

            if len(text_regions_idx) < 2:
                logger.warning("Мало текстовых регионов для определения угла, возвращаем 0")
                # Вернём бинарное изображение, убедившись что оно 1-канальное
                _, bin_img = cv2.threshold(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return 0.0, bin_img, clean_text, polar_rotated

            edges = []
            start = text_regions_idx[0]
            for i in range(1, len(text_regions_idx)):
                if text_regions_idx[i] != text_regions_idx[i - 1] + 1:
                    edges.append((start, text_regions_idx[i - 1]))
                    start = text_regions_idx[i]
            edges.append((start, text_regions_idx[-1]))

            centers_px = [(start + end) / 2 for start, end in edges]
            lengths = [end - start for start, end in edges]
            max_len_idx = np.argmax(lengths)
            center_px = centers_px[max_len_idx]

            one_pixel_degree = 360 / clean_text.shape[1]
            detected_angle = center_px * one_pixel_degree

            ref_texts = self.reference_text_angles.get(self.angle_params['num_detail'], [])
            if not ref_texts:
                logger.warning("Эталонные углы текста для данного num_detail не определены")
                _, bin_img = cv2.threshold(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return detected_angle, bin_img, clean_text, polar_rotated

            ref_angle = ref_texts[0]['center_angle']
            angle_diff = ref_angle - detected_angle
            if angle_diff < 0:
                angle_diff += 360
            elif angle_diff >= 360:
                angle_diff -= 360
            self.min_angle = round(angle_diff, 2)

            logger.info(f"Рассчитанный угол поворота: {self.min_angle}")

            # Возврат именно 1-канального бинарного изображения
            _, bin_img = cv2.threshold(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return self.min_angle, bin_img, clean_text, polar_rotated

        except Exception as e:
            logger.error(f"Ошибка определения угла: {e}")
            # Возвращаем 1-канальное чёрно-белое нулевое изображение при ошибке
            bin_empty = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            return 0.0, bin_empty, bin_empty, bin_empty

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            logger.error("Не удалось загрузить изображение")
            return None, 0.0, None, None, None, None, None

        # Предварительная обработка
        binary = self.preprocess_image(img)

        # Поиск всех окружностей
        small_circles, large_circles = self.detect_all_circles(binary)

        # Поиск основного контура
        contour = self.find_main_contour(binary)
        if contour is None:
            logger.warning("Не найден основной контур детали")
            return img, 0.0, binary, None, None, small_circles, large_circles

        # Определение угла
        angle, binary_img, text_img, polar_img = self.calculate_rotation_angle(img, contour)

        # Визуализация
        result = img.copy()
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

        # Рисуем окружности разными цветами
        result = self.draw_circles(result, small_circles, (255, 0, 0))  # Синий для малых
        result = self.draw_circles(result, large_circles, (0, 255, 0))  # Зеленый для больших

        if self.center:
            cv2.circle(result, self.center, 5, (0, 0, 255), -1)
        cv2.putText(result, f"Angle: {angle}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return result, angle, binary_img, text_img, polar_img, small_circles, large_circles


def main():
    detector = CircleDetector()
    image_path = r"Ваш путь к изображению"  # Укажите путь к изображению

    result_img, angle, binary_img, text_img, polar_img, small_circles, large_circles = detector.process_image(image_path)

    if result_img is not None:
        cv2.namedWindow("Original with Circles", cv2.WINDOW_NORMAL)
        cv2.imshow("Original with Circles", result_img)

        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Binary Image", binary_img)

        if text_img is not None:
            cv2.namedWindow("Text Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Text Detection", text_img)

        if polar_img is not None:
            cv2.namedWindow("Polar Transform", cv2.WINDOW_NORMAL)
            cv2.imshow("Polar Transform", polar_img)

        print(f"Найдено малых окружностей: {len(small_circles)}")
        print(f"Найдено больших окружностей: {len(large_circles)}")
        print(f"Определенный угол поворота: {angle}")

        cv2.imwrite("result_original.jpg", result_img)
        cv2.imwrite("result_binary.jpg", binary_img)
        if text_img is not None:
            cv2.imwrite("result_text.jpg", text_img)
        if polar_img is not None:
            cv2.imwrite("result_polar.jpg", polar_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Не удалось обработать изображение")


if __name__ == "__main__":
    main()

