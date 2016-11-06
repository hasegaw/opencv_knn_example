#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  K近傍法による画像仕分けのサンプル
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
import copy
import random

import cv2
import numpy as np

# image spec
w = 64
h = 64

# KNN samples
knn = cv2.ml.KNearest_create()
samples = None
samples = np.empty((0, 64 * 64))
responses = []


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--randomness',
        type=int,
        default=0,
        metavar='N',
        help='生成する問題画像のばらつき(0 < N < 10)'
    )
    parser.add_argument(
        '--learn-randomness',
        type=int,
        default=0,
        metavar='N',
        help='学習する正解画像のばらつき(0 < N < 10)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        metavar='N',
        help='学習する正解画像の数（図形あたり） (0 < N)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        metavar='N',
        help='最近傍の図形の必要数 (num_samples <= N)'
    )
    return parser


def new_image():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def modify_shape(points, r):
    for point in points:
        point[0] = point[0] + (random.random() - 0.5) * (w * 0.2) * r
        point[1] = point[1] + (random.random() - 0.5) * (w * 0.2) * r
    return points


def draw_triangle(r=0):
    points = modify_shape(
        np.array([
            [w * 0.5, h * 0.2],
            [w * 0.2, h * 0.8],
            [w * 0.8, h * 0.8],
        ], dtype=np.int32),
        r
    )

    img = new_image()
    cv2.polylines(img, [points], 1, (255, 255, 255), 3)
    return img


def draw_rectangle(r=0):
    points = modify_shape(np.array(
        [[w * 0.2, h * 0.2], [w * 0.8, h * 0.8]], dtype=np.int32), r)

    img = new_image()
    cv2.rectangle(img, tuple(points[0]), tuple(points[1]), (255, 255, 255), 3)
    return img


def draw_circle(r=0):
    points = modify_shape(
        np.array([[w * 0.5, h * 0.5], [w * 0.3, 0]], dtype=np.int32), r)
    rad = points[1][0]

    img = new_image()
    cv2.circle(img, tuple(points[0]), rad, (255, 255, 255), 3)
    return img


def add_sample(response, img):
    global samples
    global responses

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sample = img.reshape((1, img.shape[0] * img.shape[1]))
    sample = np.array(sample, np.float32)

    if samples is None:
        samples = np.empty((0, img.shape[0] * img.shape[1]))
        responses = []

    samples = np.append(samples, sample, 0)
    responses.append(response)


def learn(num_samples, learn_randomness):
    # ランダム性がない画像を学習
    add_sample(1, draw_circle())
    add_sample(2, draw_triangle())
    add_sample(3, draw_rectangle())

    # ランダム性がある画像を学習
    for i in range(num_samples - 1):
        add_sample(1, draw_circle(learn_randomness))
        add_sample(2, draw_triangle(learn_randomness))
        add_sample(3, draw_rectangle(learn_randomness))

    print('(サンプル数, 次元数) = ', samples.shape)
    knn.train(
        np.array(samples, np.float32),
        cv2.ml.ROW_SAMPLE,
        np.array(responses, np.float32),
    )


def main():
    # 引数のパース
    parser = create_parser()
    args = parser.parse_args()
    print(args)

    assert 0 < args.num_samples
    assert 0 < args.k <= args.num_samples
    assert 0 <= args.learn_randomness <= 10
    assert 0 <= args.randomness <= 10

    # 指定されたパラメータで学習
    learn(args.num_samples, args.learn_randomness / 10)

    # 問題画像の生成を使うパラメータ
    randomness = args.randomness / 10
    k = args.k

    # ESCキーが押されるまでひたすら問題を解く
    keycode = 0
    while keycode != 27:
        # 三角形、四角形、円のどれかを描画する
        draw_funcs = [draw_triangle, draw_rectangle, draw_circle, ]
        img = draw_funcs[int(random.random() * 0.99 *
                             len(draw_funcs))](randomness)

        # K近傍法で最も近いサンプルを調べる
        img2 = cv2.cvtColor(copy.deepcopy(img), cv2.COLOR_BGR2GRAY)
        sample = img2.reshape((1, img2.shape[0] * img2.shape[1]))
        sample = np.array(sample, np.float32)
        retval, results, neigh_rep, dists = knn.findNearest(sample, k)

        # 調べたサンプルのレスポンスから元の画像名を生成
        num_answer = int(results.ravel()[0])
        answer = {1: 'circle', 2: 'triangle', 3: 'rectangle'}[num_answer]

        # 元の画像名のウインドウに表示
        cv2.imshow('source', cv2.resize(img, (128, 128)))

        cv2.moveWindow('source', 80, 20)

        for l in (255, 192, 128, 64, 0):
            img_ = cv2.resize(img, (128, 128))
            img_[img_ < l] = l
            cv2.imshow(answer, img_)
            cv2.moveWindow(answer, 80 + 240 * (num_answer - 1), 240)
            keycode = cv2.waitKey(30)


main()
