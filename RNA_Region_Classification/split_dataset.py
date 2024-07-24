# -*- coding: utf-8 -*-
# file: test.py
# time: 00:48 16/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import os.path
import random

import findfile

for f in findfile.find_cwd_files(".txt"):
    print(f)
    with open(f, "r") as file:
        lines = file.readlines()
        random.shuffle(lines)

    with open(os.path.dirname(f) + "/train.json", "w") as file:
        for line in lines[: int(len(lines) * 0.8)]:
            file.write(line)

    with open(os.path.dirname(f) + "/test.json", "w") as file:
        for line in lines[int(len(lines) * 0.8) : int(len(lines) * 0.9)]:
            file.write(line)

    with open(os.path.dirname(f) + "/valid.json", "w") as file:
        for line in lines[int(len(lines) * 0.9) :]:
            file.write(line)
