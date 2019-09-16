# -*- coding: future_fstrings -*-
import sys
import numpy as np
import copy

import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import warn as WARN
from logging import error as ERROR
from logging import critical as CRITICAL

def PadList(list_to_pad, req_length, padding_token):
    diff = req_length - len(list_to_pad)
    if diff < 0:
        raise ValueError("list to pad already longer than req_length")

    new_list = copy.deepcopy(list_to_pad)

    for _ in range(diff):
        new_list.append(padding_token)

    return new_list