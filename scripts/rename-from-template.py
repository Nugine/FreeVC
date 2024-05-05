#!/usr/bin/env python

import os
import sys
import re


def replace_in_file(path, project_name):
    with open(path, "r") as f:
        content = f.read()

    # https://github.com/Nugine/dl-template
    content = re.sub(r"(?<!Nugine/)dl-template", project_name, content)

    with open(path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    project_name = sys.argv[1]

    os.rename("dl-template", project_name)
    replace_in_file("run", project_name)

    print("Done!")
