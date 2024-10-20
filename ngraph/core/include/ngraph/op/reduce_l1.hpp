// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v4
        {
            /// \brief Reduction operation using L1 norm: L1(x) = sum(abs(x)) if all dimensions are
            /// specified for the normalisation.
            ///
            /// Reduces the tensor, eliminating the specified reduction axes by taking the L1-norm.
            class NGRAPH_API ReduceL1 : public util::ArithmeticReductionKeepDims
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReduceL1", 4};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a reducet L1-norm operation.
                ReduceL1() = default;
                /// \brief Constructs a reduce L1-norm operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to true it holds axes that are used for reduction.
                ReduceL1(const Output<Node>& arg,
                         const Output<Node>& reduction_axes,
                         bool keep_dims = false);

                size_t get_version() const override { return 4; }
                /// \return The default value for Reduce.
                virtual std::shared_ptr<Node> get_default_value() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        }
    }
}
