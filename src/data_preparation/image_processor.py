
import cv2 as cv
import os



input_path='/home/aulipriya/Desktop/new_modifiers'
roi_path='/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/modifier-processed'
tot_pixel_thresh=8
continuous_wht_thresh=7


def space_removal(img):


    '''
    #perform errosion....
    kernel = np.ones((2,2), np.uint8)
    img = cv.erode(img, kernel, iterations=3)
    #cv.imshow("erosion",img)
    #cv.waitKey(0)

    '''
    # perform binarization...
    retval, threshold = cv.threshold(img, 10, 255, cv.THRESH_BINARY)

    image = threshold

    row = image.shape[0]
    col = image.shape[1]
    left_side_col = -1
    right_side_col = -1



    # finding left side column
    continuous_white=0

    for i in range(0, col):
        tot_pixel = 0
        for j in range(0, row):
            if image[j, i] != 0:
                tot_pixel += 1

        if tot_pixel >=tot_pixel_thresh:
            continuous_white+=1
        if continuous_white>=continuous_wht_thresh:
            left_side_col=i-5
            break


    continuous_white=0
    # finding right side column
    for i in range(col - 1, 0, -1):
        tot_pixel = 0;
        for j in range(0,row):
            if image[j, i] != 0:
                tot_pixel += 1

        if tot_pixel >=tot_pixel_thresh:
            continuous_white+=1
        if continuous_white>=continuous_wht_thresh:
            right_side_col=i+15
            break



    #cropping the image.......
    x =left_side_col
    y = 0
    h = row
    w = right_side_col - left_side_col
    roi = image[y:y + h, x:x + w]



    #cv.imshow('result',roi)
    #cv.waitKey(0)

    return roi


if __name__== "__main__":
   image_list=os.listdir(input_path)

   for i in range(0, len(image_list)):
       folder = image_list[i]
       path = os.path.join(input_path, folder)
       images = os.listdir(path)
       output_path = os.path.join(roi_path, folder)
       os.makedirs(output_path)
       for img_name in images:
           img = cv.imread(os.path.join(path, img_name))
           # convetr the image to binary image...
           try:
               img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
               # print('shape before space removal {}'.format(img.shape))
               roi = space_removal(img)
               # print('shape after space removal {}'.format(roi.shape))

               # rotated the image about 90 degee angle ..
               roi = cv.rotate(roi, cv.ROTATE_90_CLOCKWISE)
               roi = space_removal(roi)

               # rotated back to main position
               roi = cv.rotate(roi, cv.ROTATE_90_COUNTERCLOCKWISE)

               # image_name = 'output' + str(i) + '.jpg'

               image_name = os.path.join(output_path, 'processed.png')
               cv.imwrite(image_name, roi)
           except:
               continue
