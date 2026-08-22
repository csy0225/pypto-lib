/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * Please refer to the LICENSE file in the root of the software repository.
 */

#ifndef PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP
#define PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP

#include "tensor.h"

// Simpler's address-free Buffer ABI added task_interface/buffer.h and renamed
// the device-side Tensor descriptor to ChipTensor.  Keep this external kernel
// source compilable with both the preceding PyPTO runtime and the new ABI so
// the pypto-lib compatibility change can land before PyPTO updates its pin.
#if __has_include("task_interface/buffer.h")
using PyPTORuntimeTensor = ChipTensor;
#else
using PyPTORuntimeTensor = Tensor;
#endif

#endif  // PYPTO_QWEN_RUNTIME_TENSOR_COMPAT_HPP
