---
layout: post
title: Numpy Manual Page
date:   2016-09-23 23:47:39
categories: others
---
Hi, This is `Numpy Manual Directory`. I made this due to maintaining my Numpy Skills and preparing for creating Deep Learning Source Code.

### Python

This is code snippets:

{% highlight python %}
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) / 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print quicksort([3,6,8,10,1,2,1])
# Prints "[1, 1, 2, 3, 6, 8, 10]"
{% endhighlight %}
