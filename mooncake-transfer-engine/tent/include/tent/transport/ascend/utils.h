// Copyright 2026 Huawei Technologies Co., Ltd
// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ASCEND_DIRECT_UTILS_H_
#define ASCEND_DIRECT_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include <acl/acl.h>
#include <hixl/hixl.h>

#include "tent/common/status.h"
#include "tent/transport/ascend/ascend_direct_transport.h"

namespace mooncake {
namespace tent {

std::string hixlTransferStatusToString(hixl::TransferStatus status);

uint16_t findListenPort(int32_t dev_id);

std::string getHostIpFromSegmentName(const std::string &segment_name);

void initializeHixlTask(HixlTask &task, const Request &request);

hixl::TransferOpDesc buildTransferOpDesc(const HixlTask &task);

Status getHixlMemType(const BufferDesc &desc, hixl::MemType &mem_type);

Status getMemcpyKind(Request::OpCode opcode, const Request &request,
                     aclrtMemcpyKind &kind);

aclrtMemcpyBatchAttr buildMemcpyBatchAttr(aclrtMemcpyKind kind,
                                          int32_t device_logic_id);

}  // namespace tent
}  // namespace mooncake

#endif  // ASCEND_DIRECT_UTILS_H_
