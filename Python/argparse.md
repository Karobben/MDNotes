---
title: "argparse lib in Python | Writing python scripts"
ytitle: "Python argparse 參數請求庫"
description: "argparse examples for python"
url: argparse2
date: 2020/01/22
toc: true
excerpt: "argparse for python"
tags: [Python, Script]
category: [Python, Scripting, Module]
cover: 'https://miro.medium.com/v2/resize:fit:720/format:webp/1*PIpjPTlcrDyXLl2fDv34bA.png'
covercopy: '<a href="https://towardsdatascience.com/python-libraries-for-natural-language-processing-be0e5a35dd64">© Claire D. Costa</a>'
thumbnail: 'https://tse4-mm.cn.bing.net/th/id/OIP.uTOM2B_iUkko5GTxOa3c-wAAAA'
priority: 10000
---

## 1. sys

```python
sys.argv[1]
```

## 2. argparse
### 1. Quick Start

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input')
parser.add_argument('-o','-U','--output')

args = parser.parse_args()
INPUT = args.input
RANGE = args.output
```

```bash
python3 test.py -i inputfile -o outpufile
```

### 2. Important arguments

```python
with type and default
parser.add_argument(
  '--width',
  dest='num_hands',
  type = int,
  default = 80,
  help='Max number of hands to detect.')
```

### 3. Reading *.png

#### 3.1 nargs="+" (One/More)

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input',nargs='+')

args = parser.parse_args()
INPUT = args.input

print(INPUT)
```

```bash
python3.7 test.py  -i Ms*
['Msg', 'Msg2']
```

#### 3.2 nargs="?" (None/One)

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input',  default='a', nargs='?')

args = parser.parse_args()
INPUT = args.input

print(INPUT)
```

```bash
python3.7 test.py
a
python3.7 test.py -i
None
python3.7 test.py -i b
b
```
