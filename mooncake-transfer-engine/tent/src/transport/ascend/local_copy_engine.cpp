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

#include "tent/transport/ascend/local_copy_engine.h"

#include <algorithm>

#include <glog/logging.h>

#include "tent/transport/ascend/utils.h"

namespace mooncake {
namespace tent {
namespace {
constexpr size_t kMemcpyBatchLimit = 4096U;
constexpr uint32_t kStreamFlags =
    ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC;
}  // namespace

LocalCopyEngine::LocalCopyEngine() = default;

LocalCopyEngine::~LocalCopyEngine() { finalize(); }

Status LocalCopyEngine::initialize(uint64_t transfer_timeout,
                                   int32_t device_logic_id) {
    transfer_timeout_ = transfer_timeout;
    device_logic_id_ = device_logic_id;
    auto ret = aclrtCreateStreamWithConfig(&stream_, 0, kStreamFlags);
    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtCreateStreamWithConfig failed, ret:" << ret
                   << ", errmsg:" << aclGetRecentErrMsg();
        return Status::InternalError("Create stream failed.");
    }
    initialized_ = true;
    return Status::OK();
}

void LocalCopyEngine::finalize() {
    if (initialized_ && stream_ != nullptr) {
        (void)aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    initialized_ = false;
}

void LocalCopyEngine::copy(Request::OpCode opcode,
                           const std::vector<HixlTask *> &tasks) {
    if (tasks.empty()) {
        return;
    }
    if (!initialized_) {
        LOG(ERROR) << "LocalCopyEngine not initialized";
        for (auto &task : tasks) {
            task->status_word = TransferStatusEnum::FAILED;
        }
        return;
    }

    aclrtMemcpyKind kind;
    auto kind_status = getMemcpyKind(opcode, tasks[0]->request, kind);
    if (!kind_status.ok()) {
        for (auto &task : tasks) {
            task->status_word = TransferStatusEnum::FAILED;
        }
        return;
    }

    if (kind == ACL_MEMCPY_HOST_TO_HOST) {
        copyWithSync(opcode, tasks, kind);
        return;
    }
    if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
        copyWithAsync(opcode, tasks, kind);
        return;
    }

    auto left_num = tasks.size();
    size_t task_index = 0;
    while (left_num > 0) {
        auto batch_num = std::min(left_num, kMemcpyBatchLimit);
        auto ret = copyWithBatch(opcode, tasks, kind, batch_num, task_index);
        if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
            copyWithAsync(opcode, tasks, kind);
            return;
        }
        left_num -= batch_num;
        task_index += batch_num;
    }
}

aclError LocalCopyEngine::copyWithBatch(
    Request::OpCode opcode, const std::vector<HixlTask *> &tasks,
    aclrtMemcpyKind kind, size_t batch_num, size_t task_index) const {
    std::vector<void *> void_remote_addrs(batch_num);
    std::vector<void *> void_local_addrs(batch_num);
    std::vector<aclrtMemcpyBatchAttr> attrs(batch_num);
    std::vector<size_t> attrs_ids(batch_num);
    std::vector<size_t> sizes(batch_num);
    size_t idx = 0;
    for (size_t i = 0; i < batch_num; i++) {
        attrs[i] = buildMemcpyBatchAttr(kind, device_logic_id_);
        attrs_ids[i] = idx++;
        auto &task = tasks[task_index + i];
        void_local_addrs[i] = task->request.source;
        void_remote_addrs[i] =
            reinterpret_cast<void *>(task->request.target_offset);
        sizes[i] = task->request.length;
    }

    size_t fail_idx;
    aclError ret;
    if (opcode == Request::OpCode::WRITE) {
        ret = aclrtMemcpyBatch(void_remote_addrs.data(), sizes.data(),
                               void_local_addrs.data(), sizes.data(),
                               sizes.size(), attrs.data(), attrs_ids.data(),
                               attrs.size(), &fail_idx);
    } else {
        ret = aclrtMemcpyBatch(void_local_addrs.data(), sizes.data(),
                               void_remote_addrs.data(), sizes.data(),
                               sizes.size(), attrs.data(), attrs_ids.data(),
                               attrs.size(), &fail_idx);
    }

    if (ret != ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        if (ret == ACL_ERROR_NONE) {
            VLOG(1) << "Copy with aclrtMemcpyBatch suc.";
            for (size_t i = 0; i < batch_num; i++) {
                tasks[task_index + i]->status_word =
                    TransferStatusEnum::COMPLETED;
            }
        } else {
            for (size_t i = 0; i < batch_num; i++) {
                tasks[task_index + i]->status_word =
                    TransferStatusEnum::FAILED;
            }
        }
    }
    return ret;
}

void LocalCopyEngine::copyWithSync(Request::OpCode opcode,
                                   const std::vector<HixlTask *> &tasks,
                                   aclrtMemcpyKind kind) {
    for (auto &task : tasks) {
        auto local_ptr = task->request.source;
        auto remote_ptr = reinterpret_cast<void *>(task->request.target_offset);
        auto len = task->request.length;
        aclError ret;
        if (opcode == Request::OpCode::WRITE) {
            ret = aclrtMemcpy(remote_ptr, len, local_ptr, len, kind);
        } else {
            ret = aclrtMemcpy(local_ptr, len, remote_ptr, len, kind);
        }
        if (ret == ACL_ERROR_NONE) {
            VLOG(1) << "Copy with aclrtMemcpy suc.";
            task->status_word = TransferStatusEnum::COMPLETED;
        } else {
            LOG(ERROR) << "aclrtMemcpy failed, ret:" << ret;
            task->status_word = TransferStatusEnum::FAILED;
        }
    }
}

void LocalCopyEngine::copyWithAsync(Request::OpCode opcode,
                                    const std::vector<HixlTask *> &tasks,
                                    aclrtMemcpyKind kind) {
    std::vector<HixlTask *> async_list;
    for (auto &task : tasks) {
        auto local_ptr = task->request.source;
        auto remote_ptr = reinterpret_cast<void *>(task->request.target_offset);
        auto len = task->request.length;
        aclError ret;
        if (opcode == Request::OpCode::WRITE) {
            ret = aclrtMemcpyAsync(remote_ptr, len, local_ptr, len, kind,
                                   stream_);
        } else {
            ret = aclrtMemcpyAsync(local_ptr, len, remote_ptr, len, kind,
                                   stream_);
        }
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "aclrtMemcpyAsync failed, ret:" << ret;
            task->status_word = TransferStatusEnum::FAILED;
            continue;
        }
        async_list.emplace_back(task);
    }
    auto ret = aclrtSynchronizeStreamWithTimeout(
        stream_, static_cast<int32_t>(transfer_timeout_));
    if (ret == ACL_ERROR_NONE) {
        VLOG(1) << "Copy with aclrtMemcpyAsync suc.";
        for (auto &task : async_list) {
            task->status_word = TransferStatusEnum::COMPLETED;
        }
    } else {
        LOG(ERROR) << "Memory copy failed.";
        (void)aclrtStreamAbort(stream_);
        for (auto &task : async_list) {
            task->status_word = TransferStatusEnum::FAILED;
        }
    }
}

}  // namespace tent
}  // namespace mooncake
