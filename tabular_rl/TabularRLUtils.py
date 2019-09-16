# -*- coding: future_fstrings -*-
import copy
import logging
import sys
from logging import critical as CRITICAL
from logging import debug as DEBUG
from logging import error as ERROR
from logging import info as INFO
from logging import warn as WARN

import numpy as np


def PadList(list_to_pad, req_length, padding_token):
    diff = req_length - len(list_to_pad)
    if diff < 0:
        raise ValueError("list to pad already longer than req_length")

    new_list = copy.deepcopy(list_to_pad)

    for _ in range(diff):
        new_list.append(padding_token)

    return new_list

def ToTuple(input_list):
    # Try to convert lists, ndarrays, etc to tuple
    try:
        return tuple(input_list)

    # If we get type error, just return the val without conversion
    except TypeError:
        return input_list