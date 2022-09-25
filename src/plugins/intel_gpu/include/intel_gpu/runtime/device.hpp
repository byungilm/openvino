// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device_info.hpp"
#include "memory_caps.hpp"
#include "layout.hpp"
#include "kernel.hpp"

#include <memory>

namespace cldnn {

const uint32_t INTEL_VENDOR_ID = 0x8086;

/// @brief Represents detected GPU device object. Use device_query to get list of available objects.
struct device {
public:
    using ptr = std::shared_ptr<device>;
    virtual device_info get_info() const = 0;
    virtual memory_capabilities get_mem_caps() const = 0;

    virtual bool is_same(const device::ptr other) = 0;
    virtual bool try_kernel_execution(kernel::ptr kernel) = 0;
    virtual int8_t get_subgroup_local_block_io_supported() = 0;
    virtual void set_subgroup_local_block_io_supported(bool support) = 0;

    float get_gops(cldnn::data_types dt) const;

    virtual ~device() = default;
};

}  // namespace cldnn
