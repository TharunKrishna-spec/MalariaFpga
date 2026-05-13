#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> conv2d_3_weight_t;
typedef ap_fixed<16,6> conv2d_3_bias_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> conv2d_3_relu_table_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> conv2d_4_weight_t;
typedef ap_fixed<16,6> conv2d_4_bias_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<18,8> conv2d_4_relu_table_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> conv2d_5_weight_t;
typedef ap_fixed<16,6> conv2d_5_bias_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<18,8> conv2d_5_relu_table_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> layer12_t;
typedef ap_fixed<16,6> dense_2_weight_t;
typedef ap_fixed<16,6> dense_2_bias_t;
typedef ap_uint<1> layer12_index;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<18,8> dense_2_relu_table_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<16,6> dense_3_weight_t;
typedef ap_fixed<16,6> dense_3_bias_t;
typedef ap_uint<1> layer15_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> dense_3_sigmoid_table_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
