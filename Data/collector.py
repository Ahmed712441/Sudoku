import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json
def get_table(img):
    width = 500
    height = 500
    resized = cv2.resize(img,(width,height))
    gray_img = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_img,50,50)
    contours2, hierarchy2 = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area>90000 and area < 200000:
            epsilon = 0.012*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            x,y,w,h = cv2.boundingRect(approx)
            return cv2.resize(resized[x:x+w,y:y+w],(width,height))
    return resized
i2 = 0
width = 50
height = 50
images = ['s1.png','s2.png','s3.png','s4.jpg','s5.png','s6.png','s7.png','s8.png','s9.png','s10.png','s11.png']
while i2 < 11:
    img = cv2.imread(images[i2])
    img2 = get_table(img)
    gray_img = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 220, 255, 0)
    contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    places = []
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area > 1000 and area < 3000:
            # cv2.drawContours(thresh, cnt, -1, (0,255,0), 3)
            epsilon = 0.012*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            x,y,w,h = cv2.boundingRect(approx)
            i = y//50
            j = x//50
            places.append([(x,y,w,h),(i,j)])
    print(len(places))
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    for place in places :
        cropped_img = gray_img[place[0][1]:place[0][1]+place[0][3],place[0][0]:place[0][0]+place[0][2]]
        pts1= np.float32([[place[0][0],place[0][1]],[place[0][0]+place[0][2],place[0][1]],[place[0][0],place[0][1]+place[0][3]],[place[0][0]+place[0][2],place[0][1]+place[0][3]]])
        pts2= np.float32([[0,0],[width,0],[0,height],[width,height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(thresh, matrix, (width, height))
        prediction = np.argmax(loaded_model.predict(warped.reshape(1,50,50,1)/255))
        path = f'D:/ml projects/{prediction}'
        cv2.imwrite(os.path.join(path , f'waka13{i2}{place[1][1]}{place[1][0]}.jpg'), warped)
        # cv2.imshow(f'prediction:{prediction},{place[1][1]}{place[1][0]}.jpg', warped)
    i2+=1
