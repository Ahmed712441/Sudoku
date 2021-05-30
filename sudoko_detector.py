import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json
from solver import solver
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
            return cv2.resize(resized[y:y+h,x:x+w],(width,height))
    return resized

img_name = input('image name : ')
img = cv2.imread(img_name)
img2 = get_table(img)
cv2.waitKey(0)
img3 = img2.copy()
gray_img = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_img, 230, 255, 0) #220
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
places = []
for cnt in contours2:
    area = cv2.contourArea(cnt)
    if area > 1000 and area < 3000:
        cv2.drawContours(img3, cnt, -1, (0,255,0), 3)
        epsilon = 0.012*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        x,y,w,h = cv2.boundingRect(approx)
        i = y//50
        j = x//50
        places.append([(x,y,w,h),(i,j)])
if len(places) > 81 or len(places) < 70:
    print(len(places))
    print("the numbers in the photo is not detected correctly")
    os._exit(0)
json_file = open('digits.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("digits.h5")
width = 50
height = 50
empty = []
suduko_array = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
for place in places :
    cropped_img = gray_img[place[0][1]:place[0][1]+place[0][3],place[0][0]:place[0][0]+place[0][2]]
    pts1= np.float32([[place[0][0],place[0][1]],[place[0][0]+place[0][2],place[0][1]],[place[0][0],place[0][1]+place[0][3]],[place[0][0]+place[0][2],place[0][1]+place[0][3]]])
    pts2= np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(thresh, matrix, (width, height))
    prediction = np.argmax(loaded_model.predict(warped.reshape(1,50,50,1)/255))
    if(prediction == 0):
        empty.append((place[1][0],place[1][1]))
    suduko_array[place[1][0]][place[1][1]] = prediction

sd = solver(suduko_array)
sd.solve()
solved = sd.board
for x,y in empty:
    for place in places:
        if((x==place[1][0]) & (y==place[1][1])):
            x2,y2,w,h = place[0]
            break
    cv2.putText(img2,str(solved[x][y]),(x2+w//2,y2+h//2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1,cv2.LINE_AA)
cv2.imshow('solved',img2)
cv2.waitKey(0)
