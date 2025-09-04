"""A collection of functions and classes used across multiple modules."""

import math
import queue
import cv2
import threading
import numpy as np
from src.common import config, settings
from random import random


def run_if_enabled(function):
    """
    Decorator for functions that should only run if the bot is enabled.
    :param function:    The function to decorate.
    :return:            The decorated function.
    """

    def helper(*args, **kwargs):
        if config.enabled:
            return function(*args, **kwargs)
    return helper


def run_if_disabled(message=''):
    """
    Decorator for functions that should only run while the bot is disabled. If MESSAGE
    is not empty, it will also print that message if its function attempts to run when
    it is not supposed to.
    """

    def decorator(function):
        def helper(*args, **kwargs):
            if not config.enabled:
                return function(*args, **kwargs)
            elif message:
                print(message)
        return helper
    return decorator


def distance(a, b):
    """
    Applies the distance formula to two points.
    :param a:   The first point.
    :param b:   The second point.
    :return:    The distance between the two points.
    """

    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def separate_args(arguments):
    """
    Separates a given array ARGUMENTS into an array of normal arguments and a
    dictionary of keyword arguments.
    :param arguments:    The array of arguments to separate.
    :return:             An array of normal arguments and a dictionary of keyword arguments.
    """

    args = []
    kwargs = {}
    for a in arguments:
        a = a.strip()
        index = a.find('=')
        if index > -1:
            key = a[:index].strip()
            value = a[index+1:].strip()
            kwargs[key] = value
        else:
            args.append(a)
    return args, kwargs


def single_match(frame, template, scales=None):
    """
    Finds the best match within FRAME, optionally trying multiple scales.
    :param frame:       The image in which to search for TEMPLATE.
    :param template:    The template to match with.
    :param scales:      List of scales to try (e.g., [0.5, 1.0, 1.5, 2.0]).
    :return:            The top-left and bottom-right positions of the best match.
    """
    
    if scales is None:
        # Try multiple scales for high-resolution displays
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_match = None
    best_val = -float('inf')
    best_scale = 1.0
    
    for scale in scales:
        # Scale the template
        if scale != 1.0:
            new_width = int(template.shape[1] * scale)
            new_height = int(template.shape[0] * scale)
            if new_width < 1 or new_height < 1:
                continue
            if new_width > gray.shape[1] or new_height > gray.shape[0]:
                continue
            scaled_template = cv2.resize(template, (new_width, new_height))
        else:
            scaled_template = template
            
        # Skip if template is larger than frame
        if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
            continue
            
        result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_match = max_loc
            best_scale = scale
    
    if best_match is None:
        # Fallback to original method
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
        _, _, _, top_left = cv2.minMaxLoc(result)
        w, h = template.shape[::-1]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return top_left, bottom_right
    
    # Calculate dimensions with best scale
    w = int(template.shape[1] * best_scale)
    h = int(template.shape[0] * best_scale)
    top_left = best_match
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    print(f'[~] Template matched at scale {best_scale:.2f}x with confidence {best_val:.3f}')
    return top_left, bottom_right


def multi_match(frame, template, threshold=0.95, scales=None):
    """
    Finds all matches in FRAME that are similar to TEMPLATE by at least THRESHOLD.
    :param frame:       The image in which to search.
    :param template:    The template to match with.
    :param threshold:   The minimum percentage of TEMPLATE that each result must match.
    :param scales:      List of scales to try for multi-scale matching.
    :return:            An array of matches that exceed THRESHOLD.
    """

    if scales is None:
        # For multi_match, use fewer scales to avoid too many false positives
        scales = [0.75, 1.0, 1.25, 1.5, 2.0]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_results = []
    
    for scale in scales:
        # Scale the template
        if scale != 1.0:
            new_width = int(template.shape[1] * scale)
            new_height = int(template.shape[0] * scale)
            if new_width < 1 or new_height < 1:
                continue
            if new_width > gray.shape[1] or new_height > gray.shape[0]:
                continue
            scaled_template = cv2.resize(template, (new_width, new_height))
        else:
            scaled_template = template
            
        # Skip if template is larger than frame
        if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
            continue
            
        result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))
        
        for p in locations:
            x = int(round(p[0] + scaled_template.shape[1] / 2))
            y = int(round(p[1] + scaled_template.shape[0] / 2))
            all_results.append((x, y))
    
    # Remove duplicate detections that are too close to each other
    if len(all_results) > 1:
        filtered_results = []
        for result in all_results:
            is_duplicate = False
            for existing in filtered_results:
                if distance(result, existing) < 20:  # 20 pixel threshold
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_results.append(result)
        return filtered_results
    
    return all_results


def convert_to_relative(point, frame):
    """
    Converts POINT into relative coordinates in the range [0, 1] based on FRAME.
    Normalizes the units of the vertical axis to equal those of the horizontal
    axis by using config.mm_ratio.
    :param point:   The point in absolute coordinates.
    :param frame:   The image to use as a reference.
    :return:        The given point in relative coordinates.
    """

    x = point[0] / frame.shape[1]
    y = point[1] / config.capture.minimap_ratio / frame.shape[0]
    return x, y


def convert_to_absolute(point, frame):
    """
    Converts POINT into absolute coordinates (in pixels) based on FRAME.
    Normalizes the units of the vertical axis to equal those of the horizontal
    axis by using config.mm_ratio.
    :param point:   The point in relative coordinates.
    :param frame:   The image to use as a reference.
    :return:        The given point in absolute coordinates.
    """

    x = int(round(point[0] * frame.shape[1]))
    y = int(round(point[1] * config.capture.minimap_ratio * frame.shape[0]))
    return x, y


def filter_color(img, ranges):
    """
    Returns a filtered copy of IMG that only contains pixels within the given RANGES.
    on the HSV scale.
    :param img:     The image to filter.
    :param ranges:  A list of tuples, each of which is a pair upper and lower HSV bounds.
    :return:        A filtered copy of IMG.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
    for i in range(1, len(ranges)):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, ranges[i][0], ranges[i][1]))

    # Mask the image
    color_mask = mask > 0
    result = np.zeros_like(img, np.uint8)
    result[color_mask] = img[color_mask]
    return result


def draw_location(minimap, pos, color):
    """
    Draws a visual representation of POINT onto MINIMAP. The radius of the circle represents
    the allowed error when moving towards POINT.
    :param minimap:     The image on which to draw.
    :param pos:         The location (as a tuple) to depict.
    :param color:       The color of the circle.
    :return:            None
    """

    center = convert_to_absolute(pos, minimap)
    cv2.circle(minimap,
               center,
               round(minimap.shape[1] * settings.move_tolerance),
               color,
               1)


def print_separator():
    """Prints a 3 blank lines for visual clarity."""

    print('\n\n')


def print_state():
    """Prints whether Auto Maple is currently enabled or disabled."""

    print_separator()
    print('#' * 18)
    print(f"#    {'ENABLED ' if config.enabled else 'DISABLED'}    #")
    print('#' * 18)


def closest_point(points, target):
    """
    Returns the point in POINTS that is closest to TARGET.
    :param points:      A list of points to check.
    :param target:      The point to check against.
    :return:            The point closest to TARGET, otherwise None if POINTS is empty.
    """

    if points:
        points.sort(key=lambda p: distance(p, target))
        return points[0]


def bernoulli(p):
    """
    Returns the value of a Bernoulli random variable with probability P.
    :param p:   The random variable's probability of being True.
    :return:    True or False.
    """

    return random() < p


def rand_float(start, end):
    """Returns a random float value in the interval [START, END)."""

    assert start < end, 'START must be less than END'
    return (end - start) * random() + start


##########################
#       Threading        #
##########################
class Async(threading.Thread):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.queue = queue.Queue()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.function(*self.args, **self.kwargs)
        self.queue.put('x')

    def process_queue(self, root):
        def f():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                root.after(100, self.process_queue(root))
        return f


def async_callback(context, function, *args, **kwargs):
    """Returns a callback function that can be run asynchronously by the GUI."""

    def f():
        task = Async(function, *args, **kwargs)
        task.start()
        context.after(100, task.process_queue(context))
    return f
