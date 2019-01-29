"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow


def Graph(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/Graph .
    """
    return tensorflow.Graph(*args, **kwargs)

def Session(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/Session .
    """
    return tensorflow.Session(*args, **kwargs)

def map_fn(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/map_fn .
    """
    return tensorflow.map_fn(*args, **kwargs)


def pad(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/pad .
    """
    return tensorflow.pad(*args, **kwargs)


def top_k(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/top_k .
    """
    return tensorflow.nn.top_k(*args, **kwargs)


def clip_by_value(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/clip_by_value .
    """
    return tensorflow.clip_by_value(*args, **kwargs)


def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest' : tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tensorflow.image.ResizeMethod.BICUBIC,
        'area'    : tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.image.resize_images(images, size, methods[method], align_corners)


def non_max_suppression(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/non_max_suppression .
    """
    return tensorflow.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/range .
    """
    return tensorflow.range(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/scatter_nd .
    """
    return tensorflow.scatter_nd(*args, **kwargs)


def gather_nd(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/gather_nd .
    """
    return tensorflow.gather_nd(*args, **kwargs)

def gather(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/gather .
    """
    return tensorflow.gather(*args, **kwargs)


def meshgrid(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/meshgrid .
    """
    return tensorflow.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/where .
    """
    return tensorflow.where(*args, **kwargs)

def matmul(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/matmul .
    """
    return tensorflow.matmul(*args, **kwargs)

def multiply(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/multiply .
    """
    return tensorflow.multiply(*args, **kwargs)

def divide(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/divide .
    """
    return tensorflow.divide(*args, **kwargs)

def norm(*args, **kwargs):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/norm .
    """
    return tensorflow.norm(*args, **kwargs)

def cross(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/cross .
    """
    return tensorflow.cross(*args, **kwargs)

def one_hot(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/one_hot .
    """
    return tensorflow.one_hot(*args, **kwargs)

def random_brightness(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/image/random_brightness .
    """
    return tensorflow.image.random_brightness(*args, **kwargs)

def random_contrast(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/image/random_contrast .
    """
    return tensorflow.image.random_contrast(*args, **kwargs)

def random_hue(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/image/random_hue .
    """
    return tensorflow.image.random_hue(*args, **kwargs)

def random_saturation(*args, **kwargs):
    """ See https://www.tensorflow.org/api_docs/python/tf/image/random_saturation .
    """
    return tensorflow.image.random_saturation(*args, **kwargs)