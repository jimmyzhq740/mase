from functools import partial

import torch
import torch.nn.functional as F

# from ....graph.mase_tracer import mark_as_leaf_func
from chop.nn.quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
    ternary_quantizer,
)

def leaky_relu_integer(x, negative_slope, inplace=False, config=None):
    bypass = config.get("bypass", False)
    if bypass:
        return F.leaky_relu(x, inplace=inplace)
    else:
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        # print (ca)
        print ('leaky_relu_integer form leaky_relu.py' )
        print (x_frac_width)
        # print (x_quantizer(x))
        x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width, is_signed=True
        )
        # neg_slope_quantizer = partial(
        #     integer_quantizer, width=x_width, frac_width=x_frac_width, is_signed=False
        # )
        # neg_slope_quantizer_value=neg_slope_quantizer(negative_slope)
        # print ("Original Slope Value: ", negative_slope)
        # print (neg_slope_quantizer_value)
        return F.leaky_relu(x_quantizer(x), negative_slope=negative_slope,inplace=inplace)