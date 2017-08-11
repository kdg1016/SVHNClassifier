#!/usr/bin/env python
# -*- coding:utf-8 -*-

########################################################################################################################
#
#
#   Image crop(box info) & save .csv file
#
#   기존 csv 파일에 추출된 정보(box 좌표 및 label)를 추가하는 루틴
#
#
########################################################################################################################
#
# 1. 사용자가 이미지를 확인하여 crop 하는 코드(각각의 숫자의 좌표정보(height,width,left,top)와 label 지정

# 2. 입력된 정보를 .csv 파일에 누적

import sys
import cv2
import numpy as np
import HoonUtils as mu
import argparse
import csv
import os
import imutils


########################################################################################################################
#   SYSTEM DEFINITION

g_img_full = []
g_img_crop = []
g_img_zoom = []
g_img_canvas = []

g_cropping  = False
g_cropped   = False
g_selecting = False
g_selected  = False
g_first     = False

g_crop_box = []
g_select_box = []

g_scale_full = 0.
g_scale_crop = 0.

xi, yi = -1, -1
xe, ye = -1, -1

MOUSE_OFFSET = (0, 0)

VALID_IMG_EXTs = (".jpg", ".png", ".bmp")


########################################################################################################################
#   FUNCTION

def zoom_and_crop(event, x, y, flags, param):

    global g_img_full, g_img_crop, g_img_zoom, g_img_canvas
    global g_cropping, g_cropped, g_selecting, g_selected
    global g_crop_box, g_select_box
    global g_scale_full, g_scale_crop
    global xi, yi, xe, ye

    if not g_cropped:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_cropping = True
            xi, yi = x, y
            xi -= MOUSE_OFFSET[0]
            yi -= MOUSE_OFFSET[1]
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_cropping is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), mu.RED, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            xe -= MOUSE_OFFSET[0]
            ye -= MOUSE_OFFSET[1]
            g_cropping = False
            g_cropped  = True
            xi = int(xi / g_scale_full)
            yi = int(yi / g_scale_full)
            xe = int(xe / g_scale_full)
            ye = int(ye / g_scale_full)
            dim = g_img_full.shape
            if 0 <= xi < xe < dim[1] and 0 <= yi < ye < dim[0]:
                g_img_crop = g_img_full[yi:ye,xi:xe]
                g_crop_box = [xi, yi, xe, ye]
                g_img_zoom, g_scale_crop = mu.imresize_full(g_img_crop)
                g_img_canvas = np.copy(g_img_zoom)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_selecting = False
                g_selected  = False
                g_select_box = []
            else:
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped  = False
                g_selected = False
    elif g_cropped and not g_selected:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_selecting = True
            xi, yi = x, y
            xi -= MOUSE_OFFSET[0]
            yi -= MOUSE_OFFSET[1]
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_selecting is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), mu.BLUE, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            xe -= MOUSE_OFFSET[0]
            ye -= MOUSE_OFFSET[1]
            g_selecting = False
            g_selected  = True
            xi -= 4
            yi -= 4
            xe += 4
            ye += 4
            g_img_canvas = cv2.rectangle(g_img_canvas, (xi, yi), (xe, ye), mu.GREEN, 2)
            dim = g_img_canvas.shape
            cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
            xi = int(xi / g_scale_crop)
            yi = int(yi / g_scale_crop)
            xe = int(xe / g_scale_crop)
            ye = int(ye / g_scale_crop)
            g_select_box = [xi, yi, xe, ye]
            g_img_zoom = g_img_canvas


# ----------------------------------------------------------------------------------------------------------------------
def specify_roi(img, win_loc=(10,10), color_fmt='RGB'):

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    if isinstance(img, str):
        img = mu.imread_safe(img)

    cv2.namedWindow('zoom_and_crop')
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('zoom_and_crop', zoom_and_crop)
    roi = []
    g_img_full = np.copy(img)

    print(" # Zoom and crop...")
    disp_img = np.copy(g_img_full)
    g_img_zoom, g_scale_full = mu.imresize_full(disp_img)
    g_img_canvas = g_img_zoom

    first = True
    while True:
        if color_fmt == 'RGB':
            cv2.imshow('zoom_and_crop', cv2.cvtColor(g_img_canvas, cv2.COLOR_RGB2BGR))
        else:
            cv2.imshow('zoom_and_crop', g_img_canvas)
        if first:
            cv2.moveWindow('zoom_and_crop', win_loc[0], win_loc[1])
            first = False
        in_key = cv2.waitKey(1) & 0xFF
        if (in_key == 27):
            print(" @ ESCAPE !! Something wrong...\n")
            sys.exit()
        elif g_selected:
            if in_key == ord('n'):
                disp_img = np.copy(g_img_full)
                g_img_zoom, g_scale_full = mu.imresize_full(disp_img)
                g_img_canvas = g_img_zoom
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped  = False
                g_selected = False
            if in_key == ord('y'):
                roi = [g_select_box[0]+g_crop_box[0], g_select_box[1]+g_crop_box[1],
                       g_select_box[2]+g_crop_box[0], g_select_box[3]+g_crop_box[1]]
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped  = False
                g_selected = False
                break
            else:
                dim = g_img_canvas.shape
                g_img_canvas = cv2.putText(g_img_canvas, "Press \'y\' for yes or \'n\' for no", (10, dim[0]-50),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, mu.RED, 4)
    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)
    return roi, img[roi[1]:roi[3],roi[0]:roi[2]]


# ----------------------------------------------------------------------------------------------------------------------
def specify_roi_line(img):

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    cv2.namedWindow('zoom_and_crop')
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('zoom_and_crop', zoom_and_crop_line)
    roi = []
    g_img_full = np.copy(img)

    print(" # Zoom and crop...")
    disp_img = np.copy(g_img_full)
    g_img_zoom, g_scale_full = mu.imresize_full(g_img_full)
    g_img_canvas = g_img_zoom

    while True:
        cv2.imshow('zoom_and_crop', g_img_canvas)
        in_key = cv2.waitKey(1) & 0xFF
        if (in_key == 27) or (in_key == ord('x')):
            print(" @ Something wrong ? Bye...\n\n")
            return False
        elif in_key == ord('n'):
            g_img_zoom, g_scale_full = mu.imresize_full(np.copy(g_img_full))
            g_img_canvas = g_img_zoom
            # dim = g_img_canvas.shape
            # cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
            g_cropped = False
            g_selected = False
        elif g_selected:
            if in_key == ord('y'):
                for k in range(len(g_select_box)):
                    roi.append([g_select_box[k][0]+g_crop_box[0], g_select_box[k][1]+g_crop_box[1],
                                g_select_box[k][2]+g_crop_box[0], g_select_box[k][3]+g_crop_box[1]])
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped = False
                g_selected = 0
                break
            else:
                dim = g_img_canvas
                g_img_canvas = cv2.putText(g_img_canvas, "Press \'y\' for yes or \'n\' for no", (10, dim[0]-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 2, mu.RED, 8)
    cv2.destroyAllWindows()
    return roi

# ----------------------------------------------------------------------------------------------------------------------
def specify_rois(img, roi_num, win_loc=(10,10)):

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    cv2.namedWindow('zoom_and_crop')
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('zoom_and_crop', zoom_and_crop)
    roi = []
    g_img_full = np.copy(img)

    for idx in range(roi_num):

        print(" # Zoom and crop {:d}...".format(idx+1))
        disp_img = np.copy(g_img_full)
        for k1 in range(idx):
            disp_img = cv2.rectangle(disp_img, tuple(roi[k1][:2]), tuple(roi[k1][2:]), mu.RED, -1)
        g_img_zoom, g_scale_full = mu.imresize_full(disp_img)
        g_img_canvas = g_img_zoom

        first = True
        while True:
            cv2.imshow('zoom_and_crop', g_img_canvas)
            if first:
                cv2.moveWindow('zoom_and_crop', win_loc[0], win_loc[1])
                first = False
            in_key = cv2.waitKey(1) & 0xFF
            if (in_key == 27) or (in_key == ord('x')):
                print(" @ Something wrong ? Bye...\n\n")
                return False
            elif in_key == ord('n'):
                disp_img = np.copy(g_img_full)
                for k1 in range(idx):
                    disp_img = cv2.rectangle(disp_img, tuple(roi[k1][:2]), tuple(roi[k1][2:]), mu.RED, -1)
                g_img_zoom, g_scale_full = mu.imresize_full(disp_img)
                g_img_canvas = g_img_zoom
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped  = False
                g_selected = False
            elif in_key == ord('y'):
                for k in range(len(g_select_box)):
                    roi.append([g_select_box[k][0]+g_crop_box[0], g_select_box[k][1]+g_crop_box[1],
                                g_select_box[k][2]+g_crop_box[0], g_select_box[k][3]+g_crop_box[1]])
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped = False
                g_selected = 0
                break
            else:
                dim = g_img_canvas.shape
                g_img_canvas = cv2.putText(g_img_canvas, "Press \'y\' for yes or \'n\' for no", (10, dim[0]-50),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, mu.RED, 4)
    cv2.destroyAllWindows()
    return roi


# ----------------------------------------------------------------------------------------------------------------------
def specify_roi_line(img, win_loc=(10,10)):

    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    cv2.namedWindow('zoom_and_crop')
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_NORMAL)
    cv2.namedWindow('zoom_and_crop', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('zoom_and_crop', zoom_and_crop_line)
    roi = []
    g_img_full = np.copy(img)

    print(" # Zoom and crop...")
    disp_img = np.copy(g_img_full)
    g_img_zoom, g_scale_full = mu.imresize_full(g_img_full)
    g_img_canvas = g_img_zoom

    first = True
    while True:
        cv2.imshow('zoom_and_crop', g_img_canvas)
        if first:
            cv2.moveWindow('zoom_and_crop', win_loc[0], win_loc[1])
            first = False
        in_key = cv2.waitKey(1) & 0xFF
        if (in_key == 27) or (in_key == ord('x')):
            print(" @ Something wrong ? Bye...\n\n")
            return False
        elif in_key == ord('n'):
            g_img_zoom, g_scale_full = mu.imresize_full(np.copy(g_img_full))
            g_img_canvas = g_img_zoom
            # dim = g_img_canvas.shape
            # cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
            g_cropped = False
            g_selected = False
        elif g_selected:
            if in_key == ord('y'):
                for k in range(len(g_select_box)):
                    roi.append([g_select_box[k][0]+g_crop_box[0], g_select_box[k][1]+g_crop_box[1],
                                g_select_box[k][2]+g_crop_box[0], g_select_box[k][3]+g_crop_box[1]])
                g_img_canvas = np.copy(g_img_full)
                dim = g_img_canvas.shape
                cv2.resizeWindow('zoom_and_crop', dim[1], dim[0])
                g_cropped = False
                g_selected = 0
                break
            else:
                dim = g_img_canvas
                g_img_canvas = cv2.putText(g_img_canvas, "Press \'y\' for yes or \'n\' for no", (10, dim[0]-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 2, mu.RED, 8)
    cv2.destroyAllWindows()
    return roi



def zoom_and_pick(event, x, y, flags, param):

    global g_img_full, g_img_crop, g_img_zoom, g_img_canvas
    global g_cropping, g_cropped, g_selecting, g_selected
    global g_crop_box, g_select_box
    global g_scale_full, g_scale_crop
    global xi, yi, xe, ye

    if not g_cropped:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_cropping = True
            xi, yi = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_cropping is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), mu.RED, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            g_cropping = False
            g_cropped = True
            xi = int(xi / g_scale_full)
            yi = int(yi / g_scale_full)
            xe = int(xe / g_scale_full)
            ye = int(ye / g_scale_full)
            g_img_crop = g_img_full[yi:ye,xi:xe]
            g_crop_box = [xi, yi, xe, ye]
            g_img_zoom, g_scale_crop = mu.imresize_full(g_img_crop)
            g_img_canvas = np.copy(g_img_zoom)
            g_selecting = False
            g_selected  = 0
            g_select_box = []
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            g_selecting = True
            xi, yi = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if g_selecting is True:
                g_img_canvas = cv2.rectangle(np.copy(g_img_zoom), (xi, yi), (x, y), mu.BLUE, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            xe, ye = x, y
            g_selecting = False
            g_selected += 1
            xi -= 4
            yi -= 4
            xe += 4
            ye += 4
            g_img_canvas = cv2.rectangle(g_img_canvas, (xi, yi), (xe, ye), mu.GREEN, 2)
            xi = int(xi / g_scale_crop)
            yi = int(yi / g_scale_crop)
            xe = int(xe / g_scale_crop)
            ye = int(ye / g_scale_crop)
            g_select_box.append([xi, yi, xe, ye])
            g_img_zoom = g_img_canvas


def specify_ref_box_roi(ref_img, group_num, check_num):
    """
    Define boxes in an image based on the given information.

    :param ref_img:
    :param group_num:
    :param check_num:
    :return:
    """


    global g_img_full, g_img_zoom, g_img_canvas
    global g_cropped, g_selected
    global g_scale_full, g_scale_crop

    zoom_and_pick_str = 'zoom_and_pick(각각의 숫자를 드래그하여 추출해 주세요. 종료를 원하시면 x를 눌러주세요.)'
    cv2.namedWindow(zoom_and_pick_str)
    cv2.namedWindow(zoom_and_pick_str, cv2.WINDOW_NORMAL)
    cv2.namedWindow(zoom_and_pick_str, cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(zoom_and_pick_str, zoom_and_pick)

    ref_box_roi = []
    g_img_full = np.copy(ref_img)


    for idx in range(group_num):

        print(" # Zoom and pick the check-boxes in {:d}-th group...".format(idx+1))
        disp_img = np.copy(g_img_full)
        for k1 in range(idx):
            for k2 in range(check_num[idx]):
                # disp_img = cv2.rectangle(disp_img, tuple(ref_box_roi[k1][k2][:2]), tuple(ref_box_roi[k1][k2][2:]), mu.RED, -1)
                disp_img = mu.add_box_overlay(disp_img, ref_box_roi[k1][k2], mu.RED, 0.5)
        g_img_zoom, g_scale_full = mu.imresize_full(disp_img)
        ref_box_roi.append([])
        g_img_canvas = g_img_zoom

        while True:
            cv2.imshow(zoom_and_pick_str, cv2.cvtColor(g_img_canvas, cv2.COLOR_RGB2BGR))
            in_key = cv2.waitKey(1) & 0xFF
            if (in_key == 27) or (in_key == ord('x')):
                print(" @ Something wrong ? Bye...\n\n")
                return False
            elif in_key == ord('n'):
                disp_img = np.copy(g_img_full)
                for k1 in range(idx):
                    for k2 in range(check_num[idx]):
                        # disp_img = cv2.rectangle(disp_img, tuple(ref_box_roi[k1][k2][:2]), tuple(ref_box_roi[k1][k2][2:]),
                        # mu.RED, -1)
                        disp_img = mu.add_box_overlay(disp_img, ref_box_roi[k1][k2], mu.RED, 0.5)
                g_img_zoom, g_scale_full = mu.imresize_full(disp_img)
                g_img_canvas = g_img_zoom
                g_cropped = False
                g_selected = 0
            elif g_selected == check_num[idx]:
                if in_key == ord('y'):
                    for k in range(len(g_select_box)):
                        ref_box_roi[idx].append([g_select_box[k][0]+g_crop_box[0], g_select_box[k][1]+g_crop_box[1],
                                                 g_select_box[k][2]+g_crop_box[0], g_select_box[k][3]+g_crop_box[1]])
                    g_img_canvas = np.copy(g_img_full)
                    g_cropped = False
                    g_selected = 0
                    break
                else:
                    g_img_canvas = cv2.putText(g_img_canvas, "Press \'y\' for the next step or  \'n\' for previous step",
                                               (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, mu.RED, 3)
    cv2.destroyAllWindows()

    return ref_box_roi




def crop_and_save_image(args):

    img = mu.imread_safe(args.img_filename)

    while True:

        roi = specify_ref_box_roi(img, 1, [1])
        if roi is False:
            print(" # Bye...\n")
            sys.exit()
        ans = input(" ? Enter cropped image filename : ")
        mu.imwrite_safe(ans, mu.crop_box_from_img(img, roi[0][0]))



def verify_num_window(ref_img):
    """
    Define boxes in an image based on the given information.

    :param ref_img:
    :return:
    """

    # length와 label 입력을 위한 Reference img 출력
    ref_str = '화면에 출력된 숫자의 자릿수와 내용을 확인해주세요. ex) 4 7890 (확인후 "y" 입력, 종료는 "x")'
    cv2.namedWindow(ref_str)
    cv2.namedWindow(ref_str, cv2.WINDOW_NORMAL)
    cv2.namedWindow(ref_str, cv2.WINDOW_KEEPRATIO)

    img = mu.imread_safe(ref_img)

    g_img_zoom, g_scale_full = mu.imresize_full(img)



    while True:
        cv2.imshow(ref_str, cv2.cvtColor(g_img_zoom, cv2.COLOR_RGB2BGR))
        in_key = cv2.waitKey(0) & 0xFF

        if (in_key == 27) or (in_key == ord('y')):
            print(" @ Verify Complete...\n\n")
            cv2.destroyAllWindows()
            break
        elif (in_key == ord('x')):
            print(" # Bye...\n")
            sys.exit()




def crop_and_save_box_info(img_file, csv_file):

    img = mu.imread_safe(img_file)
    acc_roi = []


    # 로딩된 이미지의 길이와 내용을 입력
    length_list, label = raw_input("확인된 숫자 길이와 내용을 입력해 주세요. ex) 4 7890 \n: ").split()
    length = int(length_list)

    if(length is None):
        print("이미지 Crop을 위한 숫자 정보가 없습니다. 숫자의 길이와 내용을 확인해 주세요.")

    elif(length == 0):
        print("이미지 Crop을 위한 숫자가 없습니다. 숫자의 길이와 내용을 확인해 주세요.")

    elif(length > 5):
        print("이미지 Crop을 위한 숫자가 너무 많습니다. 다른 영상을 선택해 주세요.")

    else:

        for idx in range(length):
            roi = specify_ref_box_roi(img, 1, [1]) # [1] 박스 하나 crop

            if roi is False:
                print(" # Bye...\n")
                sys.exit()

            # crop 된 box info 누적 저장
            acc_roi.append(roi[0][0])

            # 숫자 길이만큼 roi가 추출되면 반환
            if idx == length-1:

                # 추출된 각각의 숫자 crop info를 csv 파일에 추가 저장하는 루틴
                #############################################################################################
                #with open('./user_train/user_train.csv', 'a') as csvfile:
                with open(csv_file, 'a') as csvfile:

                    fieldnames = ['name', 'length', 'height(1)', 'left(1)', 'top(1)', 'width(1)', 'label(1)',
                                                    'height(2)', 'left(2)', 'top(2)', 'width(2)', 'label(2)',
                                                    'height(3)', 'left(3)', 'top(3)', 'width(3)', 'label(3)',
                                                    'height(4)', 'left(4)', 'top(4)', 'width(4)', 'label(4)',
                                                    'height(5)', 'left(5)', 'top(5)', 'width(5)', 'label(5)']

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    if(length == 1):
                        writer.writerow(
                            {'name': img_file[11:], 'length': length,
                             'height(1)': acc_roi[0][3], 'left(1)': acc_roi[0][0], 'top(1)': acc_roi[0][1], 'width(1)': acc_roi[0][2], 'label(1)': label[0]})
                    elif(length == 2):
                        writer.writerow(
                            {'name': img_file[11:], 'length': length,
                             'height(1)': acc_roi[0][3], 'left(1)': acc_roi[0][0], 'top(1)': acc_roi[0][1], 'width(1)': acc_roi[0][2], 'label(1)': label[0],
                             'height(2)': acc_roi[1][3], 'left(2)': acc_roi[1][0], 'top(2)': acc_roi[1][1], 'width(2)': acc_roi[1][2], 'label(2)': label[1]})
                    elif(length == 3):
                        writer.writerow(
                            {'name': img_file[11:], 'length': length,
                             'height(1)': acc_roi[0][3], 'left(1)': acc_roi[0][0], 'top(1)': acc_roi[0][1], 'width(1)': acc_roi[0][2], 'label(1)': label[0],
                             'height(2)': acc_roi[1][3], 'left(2)': acc_roi[1][0], 'top(2)': acc_roi[1][1], 'width(2)': acc_roi[1][2], 'label(2)': label[1],
                             'height(3)': acc_roi[2][3], 'left(3)': acc_roi[2][0], 'top(3)': acc_roi[2][1], 'width(3)': acc_roi[2][2], 'label(3)': label[2]})
                    elif(length == 4):
                        writer.writerow(
                            {'name': img_file[11:], 'length': length,
                             'height(1)': acc_roi[0][3], 'left(1)': acc_roi[0][0], 'top(1)': acc_roi[0][1], 'width(1)': acc_roi[0][2], 'label(1)': label[0],
                             'height(2)': acc_roi[1][3], 'left(2)': acc_roi[1][0], 'top(2)': acc_roi[1][1], 'width(2)': acc_roi[1][2], 'label(2)': label[1],
                             'height(3)': acc_roi[2][3], 'left(3)': acc_roi[2][0], 'top(3)': acc_roi[2][1], 'width(3)': acc_roi[2][2], 'label(3)': label[2],
                             'height(4)': acc_roi[3][3], 'left(4)': acc_roi[3][0], 'top(4)': acc_roi[3][1], 'width(4)': acc_roi[3][2], 'label(4)': label[3]})
                    else :
                        writer.writerow(
                            {'name': img_file[11:], 'length': length,
                             'height(1)': acc_roi[0][3], 'left(1)': acc_roi[0][0], 'top(1)': acc_roi[0][1], 'width(1)': acc_roi[0][2], 'label(1)': label[0],
                             'height(2)': acc_roi[1][3], 'left(2)': acc_roi[1][0], 'top(2)': acc_roi[1][1], 'width(2)': acc_roi[1][2], 'label(2)': label[1],
                             'height(3)': acc_roi[2][3], 'left(3)': acc_roi[2][0], 'top(3)': acc_roi[2][1], 'width(3)': acc_roi[2][2], 'label(3)': label[2],
                             'height(4)': acc_roi[3][3], 'left(4)': acc_roi[3][0], 'top(4)': acc_roi[3][1], 'width(4)': acc_roi[3][2], 'label(4)': label[3],
                             'height(5)': acc_roi[4][3], 'left(5)': acc_roi[4][0], 'top(5)': acc_roi[4][1], 'width(5)': acc_roi[4][2], 'label(5)': label[4]})
                    #############################################################################################
                print("user_train.csv 파일에 이미지(%s) box 추출 정보가 저장되었습니다." %(img_file))




def main_data_collect():

    parser = argparse.ArgumentParser(description="Data Collection GUI for SVHNClassifier")
    parser.add_argument("-v", "--verify", required=True, help="Method to be verified")
    parser.add_argument("-i", "--in_img_file", required=True, help="Input image file name or directory")
    parser.add_argument("-c", "--in_csv_file", required=True, help="Input csv file name or directory")
    args = parser.parse_args()


    # 이미지 파일이 dir에 존재하는지 확인
    if os.path.isdir(args.in_img_file):
        img_list = list(imutils.paths.list_files(args.in_img_file, validExts=VALID_IMG_EXTs, contains=None))
    elif os.path.isfile(args.in_img_file):
        img_list = [args.in_img_file]
    else:
        print(" @ Error: image directory or file not found {}".format(args.in_img_file))
        sys.exit()



    # 이미지 리스트에서 이미지 로딩
    for img_file in img_list:

        print(" # Processing {}...".format(img_file))

        verify_num_window(img_file)

        crop_and_save_box_info(img_file, args.in_csv_file)

    print("이미지 처리가 완료되었습니다.")





if __name__ == "__main__":

    if len(sys.argv) == 1:

        sys.argv.extend(["--verify", "crop_and_save_image", "--in_img_file",  "user_train",
                         "--in_csv_file", "user_train/user_train.csv"])

    main_data_collect()





















