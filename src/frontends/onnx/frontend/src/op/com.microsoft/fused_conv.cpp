// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/fused_conv.hpp"

#include <memory>
#include <vector>

#include "exceptions.hpp"
#include "op/conv.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector fused_conv(const Node& node) {
    auto conv_res = conv(node).at(0);

    if (node.get_ng_inputs().size() == 4) {  // Z input provided
        conv_res = std::make_shared<v1::Add>(conv_res, node.get_ng_inputs()[3]);
    }

    const auto activation_type = node.get_attribute_value<std::string>("activation");
    const auto activation_params = node.get_attribute_value<std::vector<float>>("activation_params", {});

    if (activation_type == "Relu") {
        return {std::make_shared<v0::Relu>(conv_res)};
    } else if (activation_type == "Tanh") {
        return {std::make_shared<v0::Tanh>(conv_res)};
    } else if (activation_type == "Sigmoid") {
        return {std::make_shared<v0::Sigmoid>(conv_res)};
    } else if (activation_type == "Clip") {
        CHECK_VALID_NODE(node,
                         activation_params.size() == 2,
                         "min and max attributes of Clip activation function were not provided");
        return {std::make_shared<v0::Clamp>(conv_res, activation_params[0], activation_params[1])};
    } else if (activation_type == "LeakyRelu") {
        CHECK_VALID_NODE(node,
                         activation_params.size() == 1,
                         "activation_alpha attribute of LeakyRelu activation function was not provided");
        const auto activation_alpha_node = v0::Constant::create(ov::element::f32, Shape{}, activation_params);
        return {std::make_shared<v0::PRelu>(conv_res, activation_alpha_node)};
    } else if (activation_type == "HardSigmoid") {
        CHECK_VALID_NODE(node,
                         activation_params.size() == 2,
                         "alpha and beta attributes of HardSigmoid activation function were not provided");
        const auto alpha = v0::Constant::create<float>(ov::element::f32, Shape{}, {activation_params[0]});
        const auto beta = v0::Constant::create<float>(ov::element::f32, Shape{}, {activation_params[1]});
        return {std::make_shared<v0::HardSigmoid>(conv_res, alpha, beta)};
    }
    CHECK_VALID_NODE(node,
                     !activation_type.empty(),
                     "Not supported: ",
                     activation_type,
                     " activation function was used");

    return {conv_res};
}

}  // namespace set_1

}  // namespace op

}  // namespace  onnx_import

}  // namespace  ngraph
