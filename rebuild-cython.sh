#!/bin/bash

# 更改了 `.pyx` 文件后，使用此脚本重新编译 Cython 文件。
# 确保新的编译文件能正常工作，而不会受到旧文件的影响。

if [[ "$OSTYPE" == "darwin"* ]]; then
  find -E zipline tests -regex '.*\.(c|so)' -exec rm {} +
else
  find src/zipline tests -regex '.*\.\(c\|so\|html\)' -exec rm {} +
fi
python setup.py build_ext --inplace
