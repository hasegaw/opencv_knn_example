#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  K近傍法による画像加工のサンプル
#  =================================
#  Copyright (C) 2015 Takeshi HASEGAWA
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


import argparse
import sys

import cv2
import numpy as np

# font size
fx=17
fy=17

# KNN samples
knn = cv2.ml.KNearest_create()
samples = None
responses = []
fonts = {}

def add_sample(response, img):
    global samples
    global responses

    sample = img.reshape((1, img.shape[0] * img.shape[1]))
    sample = np.array(sample, np.float32)

    if samples is None:
        samples = np.empty((0, img.shape[0] * img.shape[1]))
        responses = []

    samples = np.append(samples, sample, 0)
    responses.append(response)


def learn():
    print('Loading samples...')
    for i in range(6915):
        img = cv2.imread('kanji%d.png' % i)
        if img is None:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = img_gray.shape[0]
        w = img_gray.shape[1]

        img_norm = np.zeros((fx, fy), dtype=np.uint8)
        img_norm[:, :] = 255
        img_norm[0: fy, 0: fx] = img_gray[0: fy, 0: fx]

        # cv2.imshow('kanji', img_norm)
        # cv2.waitKey(1)

        fonts[i] = img_norm
        add_sample(i, img_norm)

    print('(サンプル数, 次元数) = ', samples.shape)
    knn.train(
        np.array(samples, np.float32),
        cv2.ml.ROW_SAMPLE,
        np.array(responses, np.float32),
    )


learn()

def draw(image_filename):
    img = cv2.imread(image_filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    count_y = int(img_gray.shape[0] / fy)
    count_x = int(img_gray.shape[1] / fx)
    cv2.imshow('Fuji', img_gray)

    skip = False
    for x in range(count_x):
        for y in range(count_y):
            img_part = img_gray[y * fy: (y + 1) * fy, x * fx: (x + 1) * fx]

            sample = img_part.reshape((1, img_part.shape[0] * img_part.shape[1]))
            sample = np.array(sample, np.float32)

            retval, results, neigh_rep, dists = knn.findNearest(sample, 1)

            img_gray[y * fy: (y + 1) * fy, x * fx: (x + 1) * fx] = \
                fonts[int(retval)]
            cv2.imshow('Fuji', img_gray)

            if skip or cv2.waitKey(1) == 27:
                skip = True

    cv2.imshow('Fuji', img_gray)
    cv2.imwrite('output_%s' % image_filename, img_gray)
    cv2.waitKey()

draw(sys.argv[1])


