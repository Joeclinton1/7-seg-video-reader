import cv2
import numpy as np
import random
import scipy
from math import floor
import csv
import time
title_csv = "mass data"
debug = 1 # set to 1 for debugging
blur_block_size = 5 # odd
threshold_block_size =55 # odd
threshold_constant = 2.4
number_of_digits = 5
SSD_location = [0.20, 0.2, 0.59, 0.7]
minArea= 120
minArea2=250
maxArea= 800
dist_thresh = 3.7
seg_seperation_dist =20
digit_slant_rot = -6
white_spot_fill = 150
max_compactness = 0.75
min_compactness = 0.23
max_y_deviation = 100
time_interval = 0.5 #seconds

dot    = scipy.dot
sin    = scipy.sin
cos    = scipy.cos
ar     = scipy.array

def get_image(image):
    img = cv2.resize(image,(1000,600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    _, angle = lines[0][0]
    rows, cols, o = img.shape
    Mtrx = cv2.getRotationMatrix2D((cols / 2, rows / 2), (angle*180/np.pi-90), 1)
    rotated = cv2.warpAffine(img, Mtrx, (cols, rows))

    SSDx1 = int(SSD_location[0] * cols)
    SSDy1 = int(SSD_location[1] * rows)
    SSDx2 = int(SSDx1 + SSD_location[2] * cols)
    SSDy2 = int(SSDy1 + SSD_location[3] * rows)
    SSD = rotated[SSDy1:SSDy2, SSDx1:SSDx2]

    return SSD

def process_image(ssd):
    h=np.shape(ssd)[0]
    mask = np.logical_and(np.logical_and(ssd[:,:,2]>100,ssd[:,:,1]<80),ssd[:,:,0]<80)
    mask[round(h/2):]=[False]
    ssd[mask]=[0,0,0]
    gray = cv2.cvtColor(ssd, cv2.COLOR_BGR2GRAY)
    """mask =cv2.threshold(gray,140,255,cv2.THRESH_BINARY)[1]
    gray= cv2.inpaint(gray,mask,8,cv2.INPAINT_TELEA)"""
    gray[gray > white_spot_fill] = white_spot_fill-12
    blur = cv2.GaussianBlur(gray, (blur_block_size, blur_block_size), 0)
    #blur = cv2.fastNlMeansDenoising(blur, None, 5, 11, 7)
    thresh = cv2.adaptiveThreshold(blur, 255,
          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
          threshold_block_size, threshold_constant)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (15, 15))
    return thresh

def get_contours(SSD):
    contours = cv2.findContours(SSD.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [cv2.approxPolyDP(x, 0.8, False) for x in contours if maxArea>cv2.contourArea(x)>minArea and min_compactness<cv2.contourArea(x)*4*np.pi/(cv2.arcLength(x,1)**2)<max_compactness]
    blank = np.zeros((SSD.shape[0], SSD.shape[1],3),np.uint8)
    for cnt in contours:
        cv2.drawContours(blank, [cnt], -1, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), -1)
    return contours, blank

def filter_segments(SSD, contours):
    contour_groups = []
    while len(contours)>0:
        matching_cnts = [contours[0]]
        matching_cnts_i = [0]
        area1=cv2.contourArea(contours[0])
        for i,cnt2 in enumerate(contours[1:]):
            dist = cv2.matchShapes(contours[0],cnt2,cv2.CONTOURS_MATCH_I1,0)
            if dist<dist_thresh:
                matching_cnts.append(cnt2)
                matching_cnts_i.append(i+1)
        tmp_contours = []
        for (i,cnt) in enumerate(contours):
            if i not in matching_cnts_i:
                tmp_contours.append(cnt)
        contours = tmp_contours.copy()
        contour_groups.append(matching_cnts)
    contours = np.array(max(contour_groups,key=len))

    return contours

def n(segment_values):
    """Return the displayed value of the SSD digit."""
    return {
       '1110111': 0,
        '0110111': 0,
        '1010111': 0,
        '0000111':0,
       '0010010': 1,
        '0101000':1,
        '0011000':1,
        '0001000':1,
       '0010001': 1,
       '0100001': 1,
        '0100100': 1,
        '0010100': 1,
       '0111101': 2,
        '0011101': 2,
        '1111101':2,
        '0001101':2,
        '0100100':2,
       '0111011': 3,
        '0011011':3,
       '1011010': 4,
       '1101011': 5,
        '1001011':5,
       '1101111': 6,
        '1001111':6,
       '0110010': 7,
       '1111111': 8,
        '1011111': 8,
        '0011111':8,
       '1111011': 9,
        '1011011': 9,
    }.get(segment_values,0)

"""
Seven-segment display segment regions:

c1  c2        c3    c4
       -------        r1
      |   1   |      
 ---   -------   ---  r2
|   |           |   |
| 0 |           | 2 |
|   |           |   |
 ---   -------   ---  r3
      |   3   |      
 ---   -------   ---  r4
|   |           |   |
| 4 |           | 5 |
|   |           |   |
 ---   -------   ---  r5
      |   6   |      
       -------        r6
"""

def find_digit_groups(contours):
    segments = []
    allContours = np.concatenate(contours)
    centerLine = np.mean(allContours,axis=0)[0][1]
    for i,c in enumerate(contours):
        centerX = np.mean(c,axis=0)[0][0]
        centerY = np.mean(c, axis=0)[0][1]
        x,y,w,h= cv2.boundingRect(c)
        if abs(centerY-centerLine)<max_y_deviation:
            #print(abs(centerY-centerLine))
            if cv2.contourArea(c)>minArea2 or centerLine-centerY>(max_y_deviation-40):
                segments.append((i,centerX,x,x+w,centerY))
    segments.sort(key=lambda x:x[0])
    segments.sort(key=lambda x: x[1])
    x1=1000
    digits = []
    last_i = 0
    for i, seg in enumerate(segments):
        x2=seg[1]
        if x2-x1>seg_seperation_dist or i==len(segments)-1:
            if i==len(segments)-1 and x2-x1<seg_seperation_dist:
                i+=1
            digits.append([s[0] for s in segments[last_i:i]])
            last_i =i
        x1=seg[3]

    return digits
def get_seg_number(x,y):
    #print(x,y)
    #regions dimenions: center-x,center-y,width,height
    regions = [
        (0.166, 0.2875,0.333,0.275),
        (0.5, 0.075,1,0.15),
        (0.833, 0.2875,0.333,0.275),
        (0.5, 0.5,1,0.15),
        (0.166,0.7125,0.333,0.275),
        (0.833, 0.7125,0.333,0.275),
        (0.5, 0.925,1,0.15),
        (0.5, 0.5, 0.333, 0.7)
    ]
    for i,r in enumerate(regions):
        w=r[2]/2
        h=r[3]/2
        x1=r[0]-w
        x2=r[0]+w
        y1 = r[1] - h
        y2 = r[1] + h
        if x1<=x<x2 and y1<=y<y2:
            if i<7:
                return i
            else:
                return 3
    return -1
def multi_bounding_rect(contours):
    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
    return min_x,min_y,max_x-min_x,max_y-min_y


def rotateContours(cnt,contours,ang):
    tmp_contours=[]
    for contour in contours:
            tmp_contours.append(np.array(dot(contour - cnt, ar([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]])) + cnt).astype(int))
    return tmp_contours
def find_number(segments,digit_num):
    segs = ['0']*7
    _, _, bw, bh = multi_bounding_rect(contours[np.concatenate(digits)])
    minWidth=40
    minHeight=bh*0.9

    seg_contours = [contours[s] for s in segments]
    bx, by, bw, bh = multi_bounding_rect(seg_contours)
    angle=(digit_slant_rot-(digit_num-round(number_of_digits/2))*2)*np.pi/180
    seg_contours = rotateContours((bx+bw/2,by+bh/2),seg_contours,angle)
    bx, by, bw, bh = multi_bounding_rect(seg_contours)

    """if bw <minWidth:
        bx-=(minWidth-bw)/2
        bw=minWidth"""
    if bh<minHeight:
        by -= minHeight-bh
        bh = minHeight


    for seg in segments:
        pos = np.mean(contours[seg], axis=0)[0]
        x = (pos[0] - bx) / bw
        y = (pos[1] - by) / bh
        #print(x,y)
        num = get_seg_number(x,y)
        if num != -1:
            segs[num] = '1'
    if segs.count('1')==1:
        return 1, seg_contours
    else:
        return n("".join(segs)), seg_contours

def read_digits(digits):
    reading = 0
    blank = np.zeros((img2.shape[0], img2.shape[1], 3), np.uint8)
    for i, segments in enumerate(digits):
        digit_value, digit_contours = find_number(segments, i)
        reading += digit_value * 10 ** (number_of_digits - 1 - i)
        if debug:
            colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for seg in digit_contours:
                cv2.drawContours(blank, [seg], -1, colour, -1)
    return floor(reading), blank

data = [["time","mass"]]
cap = cv2.VideoCapture("C:/Users/Joe/Pictures/2019-10/IMG_6694.mov")
success, image = cap.read()
frame=0
while success:
    img = get_image(image)
    if debug:
        cv2.imshow("frame", img)
        cv2.waitKey(0)
    img2=process_image(img)
    if debug:
        cv2.imshow("frame", img2)
        cv2.waitKey(0)
    contours, img3=get_contours(img2)
    contours=filter_segments(img3, contours)
    digits = find_digit_groups(contours)
    reading, blank = read_digits(digits)
    print(str(reading)[:2]+"."+str(reading)[2:])
    data.append([frame/30*time_interval,reading/1000])
    if debug:
        cv2.imshow("frame", blank)
        cv2.waitKey(0)
    for i  in range(round(30*time_interval)):
        success, image = cap.read()
        frame +=1

data =np.array(data)
mean= data.mean(axis=0)[1]
np.delete(data,data[np.logical_or(data>mean+2,data<mean-2)])
with open(title_csv+'.csv', 'w') as writeFile:
    writer = csv.writer(writeFile,lineterminator = '\n')
    writer.writerows(data)


