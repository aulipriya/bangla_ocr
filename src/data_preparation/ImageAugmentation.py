import cv2
import numpy as np
import os
import random
import cv2
from skimage.util import random_noise


class ImageAugmentation:
    def __init__(self, prob_of_drop_pixel, prob_of_rotation, prob_of_scaling, prob_of_noise, images):
        self.cwd = os.getcwd()
        config_path = os.path.join(self.cwd, '../asset/config.txt')
        with open(config_path, 'r+') as conf_file:
            for lines in conf_file.readlines():
                if lines.split('=')[0] == 'max_drop_pixel_height':
                    self.max_drop_pixel_height = int(lines.split('=')[1].strip())
                elif lines.split('=')[0] == 'max_drop_pixel_width':
                    self.max_drop_pixel_width = int(lines.split('=')[1].strip())
                elif lines.split('=')[0] == 'no_of_drops':
                    self.no_of_drops = int(lines.split('=')[1].strip())
                elif lines.split('=')[0] == 'angle_low_limit':
                    self.angle_low_limit = int(lines.split('=')[1].strip())
                elif lines.split('=')[0] == 'angle_high_limit':
                    self.angle_high_limit = int(lines.split('=')[1].strip())
                elif lines.split('=')[0] == 'scaling_low_limit':
                    self.scaling_low_limit = float(lines.split('=')[1].strip())
                elif lines.split('=')[0] == 'scaling_high_limit':
                    self.scaling_high_limit = float(lines.split('=')[1].strip())
                elif lines.split('=')[0] == 'noise_low_limit':
                    self.noise_low_limit = float(lines.split('=')[1].strip())
                elif lines.split('=')[0] == 'noise_high_limit':
                    self.noise_high_limit = float(lines.split('=')[1].strip())

            conf_file.close()

        self.prob_of_drop_pixel = prob_of_drop_pixel
        self.prob_of_rotation = prob_of_rotation
        self.prob_of_scaling = prob_of_scaling
        self.prob_of_noise = prob_of_noise
        self.images = images

    def select_image(self):
        prob_of_rotation = self.prob_of_rotation
        prob_of_scaling = self.prob_of_scaling
        prob_of_noise = self.prob_of_noise
        prob_of_drop_pixel = self.prob_of_drop_pixel
        images = self.images

        aug_images = []
        for i, image in enumerate(images):
            if image is None:
                aug_images.append(image)
                continue

            store_image = image

            blank_image = np.zeros((store_image.shape[0], store_image.shape[1]), np.uint8)

            for ii in range(0, store_image.shape[0]):
                for jj in range(0, store_image.shape[1]):
                    blank_image[ii][jj] = store_image[ii][jj]

            image = blank_image
            rand = random.random()
            if rand < prob_of_rotation:
                image = self.augmentor_rotate_image(image)
            rand = random.random()
            if rand < prob_of_scaling:
                image = self.augmentor_scale_image(image)
            rand = random.random()
            if rand < prob_of_drop_pixel:
                image = self.create_faulty_image_rect(image)
            rand = random.random()
            if rand < prob_of_noise:
                image = self.augmentor_noisy_image(image)
            aug_images.append(image)

        return aug_images

    def augmentor_rotate_image(self, image):
        angle_low_limit = self.angle_low_limit
        angle_high_limit = self.angle_high_limit

        angle = random.uniform(angle_low_limit, angle_high_limit)
        (img_h, img_w) = image.shape[:2]
        (img_cX, img_cY) = (img_w // 2, img_h // 2)
        M = cv2.getRotationMatrix2D((img_cX, img_cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        updated_img_w = int((img_h * sin) + (img_w * cos))
        updated_img_h = int((img_h * cos) + (img_w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (updated_img_w / 2) - img_cX
        M[1, 2] += (updated_img_h / 2) - img_cY
        rotated_image = cv2.warpAffine(image, M, (updated_img_w, updated_img_h))
        rotated_image = cv2.resize(rotated_image, (img_w, img_h))
        return rotated_image

    def augmentor_scale_image(self, image):
        scaling_high_limit = self.scaling_high_limit
        scaling_low_limit = self.scaling_low_limit

        resize_scale_x = random.uniform(scaling_low_limit, scaling_high_limit)
        resize_scale_y = random.uniform(scaling_low_limit, scaling_high_limit)
        resize_scale_x += 1
        resize_scale_y += 1
        img_shape = image.shape
        image = cv2.resize(image, None, fx=resize_scale_x, fy=resize_scale_y)
        canvas = np.zeros(img_shape, dtype=np.uint8)
        y_lim = int(min(resize_scale_y, 1) * img_shape[0])
        x_lim = int(min(resize_scale_x, 1) * img_shape[1])
        canvas[:y_lim, :x_lim] = image[:y_lim, :x_lim]
        image = canvas
        return image

    def augmentor_noisy_image(self, image):
        # print(image)
        noise_low_limit = self.noise_low_limit
        noise_high_limit = self.noise_high_limit

        noise_amount = random.uniform(noise_low_limit, noise_high_limit)
        image = random_noise(image, mode='s&p', amount=noise_amount)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        image = np.array(255 * image, dtype='uint8')
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('noisy_image', image)
        # cv2.waitKey(0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    # function for pixel drop
    def create_faulty_image_rect(self, image):
        max_drop_pixel_height = random.randrange(2, self.max_drop_pixel_height)
        max_drop_pixel_width = random.randrange(2, self.max_drop_pixel_width)
        no_of_drops = random.randrange(3, self.no_of_drops)
        cnt = 0

        while no_of_drops > 0 and cnt < 1000:
            img_w = image.shape[1]
            img_h = image.shape[0]
            rect_x = random.randint(max_drop_pixel_width, img_w - max_drop_pixel_width)
            rect_y = random.randint(max_drop_pixel_height, img_h - max_drop_pixel_height)
            rect_x1 = rect_x - max_drop_pixel_width // 2
            rect_x2 = rect_x + max_drop_pixel_width // 2
            rect_y1 = rect_y - max_drop_pixel_height // 2
            rect_y2 = rect_y + max_drop_pixel_height // 2
            rectangle_crop_area = np.count_nonzero(image[rect_y1: rect_y2, rect_x1: rect_x2])
            if rectangle_crop_area > 0:
                image[rect_y1:rect_y2, rect_x1:rect_x2] = 0 * max_drop_pixel_height
                no_of_drops -= 1
            cnt -= 1
        return image


# if __name__ == "__main__":
#     input_images = []
#     imgAug = ImageAugmentation(0.2, 0.2, 0.2, 0.2, input_images)
#     augmented_images = imgAug.select_image()
