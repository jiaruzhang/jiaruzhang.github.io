#!/usr/bin/env python3
# 重新生成整个discrete_math

import os
import sys
import json
import hashlib

import generate_dism_page


if __name__ != "__main__":
    exit(0)


UPDATE_MAP_FILE = "./discrete_math/map.json"
TEMPLATE_FILE = "./discrete_math/template.html"

update_map = {}
update_all = False


if "--force" in sys.argv:
    update_all = True
    print("(info) Force to update all pages.")

if os.path.exists(UPDATE_MAP_FILE):
    with open(UPDATE_MAP_FILE) as fp:
        update_map = json.load(fp)

def check_template_update(token, current_time):
    global update_map
    global update_all

    if token not in update_map or update_map[token] != current_time:
        update_map[token] = current_time
        update_all = True
        print("(info) Template (%s) changed." % token[:8])

def hash(obj):
    obj = str(obj).encode("ascii")
    return hashlib.md5(obj).hexdigest()

check_template_update(
    hash(TEMPLATE_FILE),
    hash(int(os.path.getmtime(TEMPLATE_FILE)))
)


def real_generate(filepath):
    try:
        print("(info) Generating {}...".format(filepath))
        return generate_dism_page.generate(filepath)
    except Exception as e:
        print("(error) Failed to generate {}.\n{}".format(
            filepath, str(e)
        ))


def generate(root, name):
    global update_map
    global update_all

    global pool

    path = os.path.join(root, name)

    path_token = hash(path)
    time_token = hash(int(os.path.getmtime(path)))

    if not update_all:
        if path_token in update_map:
            if update_map[path_token] == time_token:
                return
        else:
            update_map[path_token] = time_token

    data = real_generate(path)
    update_map[path_token] = time_token

    if data is None:
        return



if __name__ == "__main__":

    for root, dirs, files in os.walk(os.path.abspath("./discrete_math/")):
        for name in files:
            if name.endswith(".md") or name.endswith(".markdown"):
                generate(root, name)

    with open(UPDATE_MAP_FILE, "w") as fp:
        json.dump(update_map, fp, indent=4, sort_keys=True)
