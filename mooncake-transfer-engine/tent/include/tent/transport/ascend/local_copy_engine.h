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

#ifndef ASCEND_DIRECT_LOCAL_COPY_ENGINE_H
#define ASCEND_DIRECT_LOCAL_COPY_ENGINE_H

#include <vector>

#include <acl/acl.h>

#include "tent/transport/ascend/ascend_direct_transport.h"

namespace mooncake {
namespace tent {

class LocalCopyEngine {
   public:
    LocalCopyEngine();

    ~LocalCopyEngine();

    Status initialize(uint64_t transfer_timeout, int32_t device_logic_id);

    void finalize();

    void copy(Request::OpCode opcode, const std::vector<HixlTask *> &tasks);

   private:
    aclError copyWithBatch(Request::OpCode opcode,
                           const std::vector<HixlTask *> &tasks,
                           aclrtMemcpyKind kind, size_t batch_num,
                           size_t task_index) const;

    static void copyWithSync(Request::OpCode opcode,
                             const std::vector<HixlTask *> &tasks,
                             aclrtMemcpyKind kind);

    void copyWithAsync(Request::OpCode opcode,
                       const std::vector<HixlTask *> &tasks,
                       aclrtMemcpyKind kind);

   private:
    int32_t device_logic_id_{};
    uint64_t transfer_timeout_{};
    aclrtStream stream_{nullptr};
    bool initialized_{false};
};

}  // namespace tent
}  // namespace mooncake

#endif  // ASCEND_DIRECT_LOCAL_COPY_ENGINE_H
