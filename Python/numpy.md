---
title: "General Skills for Numpy"
description: "General Skills for Numpy"
url: numpy
date: 2020/09/12
toc: true
excerpt: "Basic grammar of numpy"
tags: [Python, Matrix]
category: [Python, Data, Matrix]
cover: 'https://miro.medium.com/v2/resize:fit:720/format:webp/1*PIpjPTlcrDyXLl2fDv34bA.png'
covercopy: '<a href="https://towardsdatascience.com/python-libraries-for-natural-language-processing-be0e5a35dd64">© Claire D. Costa</a>'
thumbnail: 'https://tse4-mm.cn.bing.net/th/id/OIP.uTOM2B_iUkko5GTxOa3c-wAAAA'
priority: 10000
---



##  np arry to list

```python
import numpy as np

List = arry.tolist()
```

## Create a list

```python
print(np.linspace(0, 100, 51))
```
<pre>
[  0.   2.   4.   6.   8.  10.  12.  14.  16.  18.  20.  22.  24.  26.
  28.  30.  32.  34.  36.  38.  40.  42.  44.  46.  48.  50.  52.  54.
  56.  58.  60.  62.  64.  66.  68.  70.  72.  74.  76.  78.  80.  82.
  84.  86.  88.  90.  92.  94.  96.  98. 100.]
</pre>


```python
# np.arrary sum()
np.sum(array1-array2)
np.mean()
```


##  append

```python
np.append(np1, np2,axis=0)
```

## Reduce Dimension

```python
x = np.array([[1, 2],[3, 4]])
print(np.ravel(x))      
print(np.ravel(x,'F'))  
```

## Locating (argwhere)

```python
arr = np.random.randint(0,10, (3,4))  
index = np.argwhere(arr < 5)
index2 = np.where(arr < 5)

# quick way to find the max and min:
arr.argmax()
arr.argmin()
```

## Axis

## Axis flip / swap

```python
frame2 = frame.swapaxes(0,1)

frame.shape
frame2.shape
```
<pre>
(360, 480, 3)
(480, 360, 3)
</pre>

## Replace


```python
arr[arr > 255] = x
```

<br />
<br />
<br />
<br />ummmm... I am not really like to using numpy because I thing data.frame() in R is much better.
