
#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_layer_1[64*64*3],
    result_t layer16_out[1]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_layer_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_layer_1,layer16_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv2d_3_weight_t, 432>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv2d_3_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv2d_4_weight_t, 4608>(w5, "w5.txt");
        nnet::load_weights_from_txt<conv2d_4_bias_t, 32>(b5, "b5.txt");
        nnet::load_weights_from_txt<conv2d_5_weight_t, 18432>(w8, "w8.txt");
        nnet::load_weights_from_txt<conv2d_5_bias_t, 64>(b8, "b8.txt");
        nnet::load_weights_from_txt<dense_2_weight_t, 262144>(w12, "w12.txt");
        nnet::load_weights_from_txt<dense_2_bias_t, 64>(b12, "b12.txt");
        nnet::load_weights_from_txt<dense_3_weight_t, 64>(w15, "w15.txt");
        nnet::load_weights_from_txt<dense_3_bias_t, 1>(b15, "b15.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[64*64*16];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[64*64*16];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[32*32*32];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[32*32*32];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer7_t layer7_out[16*16*32];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    layer8_t layer8_out[16*16*64];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer9_t layer9_out[16*16*64];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    layer10_t layer10_out[8*8*64];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    auto& layer11_out = layer10_out;
    layer12_t layer12_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    layer13_t layer13_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    layer15_t layer15_out[1];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    nnet::conv_2d_cl<input_t, layer2_t, config2>(input_layer_1, layer2_out, w2, b2); // conv2d_3

    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // conv2d_3_relu

    nnet::pooling2d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out); // max_pooling2d_3

    nnet::conv_2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5); // conv2d_4

    nnet::relu<layer5_t, layer6_t, relu_config6>(layer5_out, layer6_out); // conv2d_4_relu

    nnet::pooling2d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out); // max_pooling2d_4

    nnet::conv_2d_cl<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // conv2d_5

    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // conv2d_5_relu

    nnet::pooling2d_cl<layer9_t, layer10_t, config10>(layer9_out, layer10_out); // max_pooling2d_5

    nnet::dense<layer10_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // dense_2

    nnet::relu<layer12_t, layer13_t, relu_config13>(layer12_out, layer13_out); // dense_2_relu

    nnet::dense<layer13_t, layer15_t, config15>(layer13_out, layer15_out, w15, b15); // dense_3

    nnet::sigmoid<layer15_t, result_t, sigmoid_config16>(layer15_out, layer16_out); // dense_3_sigmoid

}

