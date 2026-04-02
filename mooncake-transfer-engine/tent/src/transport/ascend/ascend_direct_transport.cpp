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

#include "tent/transport/ascend/ascend_direct_transport.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <random>
#include <string>

#include <bits/stdint-uintn.h>
#include <glog/logging.h>

#include "tent/common/status.h"
#include "tent/runtime/slab.h"
#include "tent/runtime/control_plane.h"
#include "tent/transport/ascend/local_copy_engine.h"
#include "tent/transport/ascend/utils.h"

namespace mooncake {
namespace tent {
AscendDirectTransport::AscendDirectTransport() : installed_(false) {}

AscendDirectTransport::~AscendDirectTransport() {
    if (local_copy_engine_) {
        local_copy_engine_->finalize();
        local_copy_engine_.reset();
    }
    hixl_->Finalize();
    uninstall();
}

Status AscendDirectTransport::install(std::string &local_segment_name,
                                      std::shared_ptr<ControlService> metadata,
                                      std::shared_ptr<Topology> local_topology,
                                      std::shared_ptr<Config> conf) {
    if (installed_) {
        return Status::InvalidArgument(
            "Hixl transport has been installed" LOC_MARK);
    }

    metadata_ = metadata;
    local_segment_name_ = local_segment_name;
    local_topology_ = local_topology;
    installed_ = true;
    caps.dram_to_dram = true;
    caps.dram_to_gpu = true;
    caps.gpu_to_dram = true;
    caps.gpu_to_gpu = true;
    transfer_timeout_ =
        conf->get("transports/ascend_direct/transfer_timeout_ns", 3000000UL);
    connect_timeout_ =
        conf->get("transports/ascend_direct/connect_timeout_ns", 3000000UL);
    return initHixl(conf);
}

Status AscendDirectTransport::initHixl(const std::shared_ptr<Config> &conf) {
    auto ret = aclrtGetDevice(&device_logic_id_);
    if (ret) {
        LOG(ERROR) << "Call aclrtGetDevice failed, ret: " << ret;
        return Status::InvalidArgument(
            "Get device id failed, device id may be not set.");
    }
    ret = aclrtGetCurrentContext(&rt_context_);
    if (ret) {
        LOG(ERROR) << "Call aclrtGetCurrentContext failed, ret: " << ret;
        return Status::InternalError("Get rt context failed.");
    }
    auto host_ip = getHostIpFromSegmentName(local_segment_name_);
    auto port = findListenPort(device_logic_id_);
    auto hixl_name = host_ip + ":" + std::to_string(port);
    local_hixl_name_ = hixl_name;

    auto segment = metadata_->segmentManager().getLocal();
    auto &detail = std::get<MemorySegmentDesc>(segment->detail);
    detail.device_attrs["hixl_name"] = hixl_name;

    hixl_ = std::make_unique<hixl::Hixl>();
    if (!hixl_) return Status::InternalError("Create hixl failed.");
    std::map<hixl::AscendString, hixl::AscendString> options;
    std::string rdma_tc = conf->get("transports/ascend_direct/rdma_tc", "");
    if (!rdma_tc.empty()) {
        options["RdmaTrafficClass"] = rdma_tc.c_str();
        LOG(INFO) << "Set RdmaTrafficClass to:" << rdma_tc;
    } else {
        auto rdma_tc_env = std::getenv("HCCL_RDMA_TC");
        if (rdma_tc_env) {
            options["RdmaTrafficClass"] = rdma_tc_env;
            LOG(INFO) << "Set RdmaTrafficClass to:" << rdma_tc_env;
        }
    }
    std::string rdma_sl = conf->get("transports/ascend_direct/rdma_sl", "");
    if (!rdma_sl.empty()) {
        options["RdmaServiceLevel"] = rdma_sl.c_str();
        LOG(INFO) << "Set RdmaServiceLevel to:" << rdma_sl;
    } else {
        auto rdma_sl_env = std::getenv("HCCL_RDMA_SL");
        if (rdma_sl_env) {
            options["RdmaServiceLevel"] = rdma_sl_env;
            LOG(INFO) << "Set RdmaServiceLevel to:" << rdma_sl_env;
        }
    }
    std::string auto_connect =
        conf->get("transports/ascend_direct/auto_connect", "");
    if (!auto_connect.empty()) {
        auto_connect_ = (auto_connect == "1");
        options["AutoConnect"] = auto_connect_ ? "1" : "0";
        LOG(INFO) << "Set AutoConnect to: " << auto_connect;
    } else {
        options["AutoConnect"] = "1";
        auto_connect_ = true;
        LOG(INFO) << "Set AutoConnect to default: 1";
    }
    auto status =
        hixl_->Initialize(hixl::AscendString(hixl_name.c_str()), options);
    if (status != hixl::SUCCESS) {
        LOG(ERROR) << "Failed to initialize AdxlEngine, status: " << status
                   << ", errmsg: " << aclGetRecentErrMsg();
        return Status::InternalError("Initialize hixl failed.");
    }
    local_copy_engine_ = std::make_unique<LocalCopyEngine>();
    if (!local_copy_engine_) {
        return Status::InternalError("Create local copy engine failed.");
    }
    auto local_copy_status =
        local_copy_engine_->initialize(transfer_timeout_, device_logic_id_);
    if (!local_copy_status.ok()) {
        return local_copy_status;
    }
    LOG(INFO) << "Success to initialize hixl engine:" << hixl_name
              << " with device_id:" << device_logic_id_;
    return Status::OK();
}

Status AscendDirectTransport::uninstall() {
    if (installed_) {
        metadata_.reset();
        installed_ = false;
    }
    return Status::OK();
}

Status AscendDirectTransport::allocateSubBatch(SubBatchRef &batch,
                                               size_t max_size) {
    auto hixl_batch = Slab<HixlSubBatch>::Get().allocate();
    if (!hixl_batch)
        return Status::InternalError("Unable to allocate TCP sub-batch");
    batch = hixl_batch;
    hixl_batch->task_list.reserve(max_size);
    hixl_batch->max_size = max_size;
    return Status::OK();
}

Status AscendDirectTransport::freeSubBatch(SubBatchRef &batch) {
    auto hixl_batch = dynamic_cast<HixlSubBatch *>(batch);
    if (!hixl_batch)
        return Status::InvalidArgument("Invalid TCP sub-batch" LOC_MARK);
    Slab<HixlSubBatch>::Get().deallocate(hixl_batch);
    batch = nullptr;
    return Status::OK();
}

Status AscendDirectTransport::submitTransferTasks(
    SubBatchRef batch, const std::vector<Request> &request_list) {
    auto hixl_batch = dynamic_cast<HixlSubBatch *>(batch);
    if (!hixl_batch)
        return Status::InvalidArgument("Invalid TCP sub-batch" LOC_MARK);
    if (request_list.size() + hixl_batch->task_list.size() >
        hixl_batch->max_size)
        return Status::TooManyRequests("Exceed batch capacity" LOC_MARK);
    std::map<SegmentID, std::map<Request::OpCode, std::vector<HixlTask *>>>
        seg_to_tasks;
    for (auto &request : request_list) {
        hixl_batch->task_list.push_back(HixlTask{});
        auto &task = hixl_batch->task_list[hixl_batch->task_list.size() - 1];
        initializeHixlTask(task, request);
        seg_to_tasks[task.request.target_id][task.request.opcode].push_back(
            &task);
    }
    for (auto &[seg_id, op_to_tasks] : seg_to_tasks) {
        for (auto &[opcode, tasks] : op_to_tasks) {
            startTransfer(seg_id, opcode, tasks);
        }
    }
    return Status::OK();
}

Status AscendDirectTransport::checkAndConnect(const std::string &remote_hixl) {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    auto it = connected_segments_.find(remote_hixl);
    if (it != connected_segments_.end()) {
        VLOG(1) << "Already connected to target hixl engine: " << remote_hixl;
    } else {
        auto status = hixl_->Connect(
            remote_hixl.c_str(), static_cast<int32_t>(connect_timeout_ / 1000));
        if (status == hixl::TIMEOUT) {
            LOG(ERROR) << "Connect timeout to: " << remote_hixl
                       << ", you can increase the timeout duration to reduce "
                          "the failure rate by configuring "
                          "the ASCEND_CONNECT_TIMEOUT environment variable"
                       << ", errmsg: " << aclGetRecentErrMsg();
            return Status::InternalError("Connect to target timed out.");
        } else if (status != hixl::SUCCESS) {
            LOG(ERROR) << "Failed to connect to target: " << remote_hixl
                       << ", status: " << status
                       << ", errmsg: " << aclGetRecentErrMsg();
            return Status::InternalError("Connect to target failed.");
        }
        connected_segments_.emplace(remote_hixl);
        LOG(INFO) << "Connected to segment: " << remote_hixl;
    }
    return Status::OK();
}

void AscendDirectTransport::startTransfer(
    SegmentID target_id, Request::OpCode opcode,
    const std::vector<HixlTask *> &tasks) {
    std::string rpc_server_addr;
    SegmentDesc *desc = nullptr;
    auto status = metadata_->segmentManager().getRemoteCached(desc, target_id);
    if (!status.ok()) {
        for (auto &task : tasks) {
            task->status_word = TransferStatusEnum::FAILED;
        }
        return;
    }
    auto &detail = std::get<MemorySegmentDesc>(desc->detail);
    auto remote_hixl = detail.device_attrs["hixl_name"];
    if (remote_hixl == local_hixl_name_) {
        VLOG(1) << "Target is local.";
        if (!local_copy_engine_) {
            LOG(ERROR) << "Local copy engine is not initialized.";
            for (auto &task : tasks) {
                task->status_word = TransferStatusEnum::FAILED;
            }
            return;
        }
        auto start = std::chrono::steady_clock::now();
        local_copy_engine_->copy(opcode, tasks);
        LOG(INFO) << "Local copy cost: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count()
                  << " us.";
        return;
    } else {
        if (!auto_connect_) {
            auto ret = checkAndConnect(remote_hixl);
            if (!ret.ok()) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(connection_mutex_);
                connected_segments_.emplace(remote_hixl);
            }
        }
    }
    auto op = (opcode == Request::WRITE) ? hixl::WRITE : hixl::READ;
    std::vector<hixl::TransferOpDesc> op_descs;
    op_descs.reserve(tasks.size());
    for (auto &task : tasks) {
        op_descs.emplace_back(buildTransferOpDesc(*task));
    }
    hixl::Status hixl_ret;
    hixl::TransferReq req_handle;
    hixl_ret = hixl_->TransferAsync(hixl::AscendString(remote_hixl.c_str()), op,
                                    op_descs, hixl::TransferArgs(), req_handle);
    if (hixl_ret != hixl::SUCCESS) {
        LOG(ERROR) << "Failed to transfer to: " << remote_hixl
                   << ", status: " << hixl_ret
                   << ", errmsg: " << aclGetRecentErrMsg();
        // disconnect to remote when transfer fail
        disconnect(remote_hixl, 10);
        for (auto &task : tasks) {
            task->status_word = TransferStatusEnum::FAILED;
        }
        return;
    }
    for (auto &task : tasks) {
        task->start_time = getCurrentTimeInNano();
        task->req_handle = req_handle;
        task->batch_size = tasks.size();
        task->remote_hixl = remote_hixl;
        task->status_word = TransferStatusEnum::PENDING;
    }
}

Status AscendDirectTransport::getTransferStatus(SubBatchRef batch, int task_id,
                                                TransferStatus &status) {
    auto hixl_batch = dynamic_cast<HixlSubBatch *>(batch);
    if (task_id < 0 || task_id >= (int)hixl_batch->task_list.size()) {
        return Status::InvalidArgument("Invalid task id" LOC_MARK);
    }
    auto &task = hixl_batch->task_list[task_id];
    status = TransferStatus{task.status_word, task.transferred_bytes};
    if (task.status_word == TransferStatusEnum::PENDING) {
        if (task.req_handle == nullptr) {
            return Status::OK();
        }
        std::lock_guard<std::mutex> lock(req_mutex_);
        auto it = req_map_.find(task.req_handle);
        if (it != req_map_.end()) {
            auto xfer_status = it->second.first;
            task.status_word = xfer_status;
            if (xfer_status == TransferStatusEnum::COMPLETED) {
                task.transferred_bytes = task.request.length;
            }
            if (--req_map_[task.req_handle].second == 0) {
                req_map_.erase(task.req_handle);
            }
            return Status::OK();
        }
        uint64_t current_ts = getCurrentTimeInNano();
        if ((current_ts - task.start_time) > transfer_timeout_) {
            disconnect(task.remote_hixl, 10);
            task.status_word = TransferStatusEnum::TIMEOUT;
            if (task.batch_size > 1) {
                req_map_[task.req_handle] =
                    std::make_pair(task.status_word, task.batch_size - 1);
            }
            return Status::OK();
        }
        hixl::TransferStatus xfer_status;
        auto err = hixl_->GetTransferStatus(task.req_handle, xfer_status);
        if (err != hixl::SUCCESS) {
            // call failed equal to transfer failed;
            xfer_status = hixl::TransferStatus::FAILED;
        }
        if (xfer_status == hixl::TransferStatus::WAITING) {
            return Status::OK();
        }
        if (xfer_status == hixl::TransferStatus::COMPLETED) {
            task.transferred_bytes = task.request.length;
            task.status_word = TransferStatusEnum::COMPLETED;
        } else if (xfer_status == hixl::TransferStatus::FAILED) {
            LOG(ERROR) << "Get transfer status failed, ret: "
                       << hixlTransferStatusToString(xfer_status)
                       << ", errmsg: " << aclGetRecentErrMsg();
            disconnect(task.remote_hixl, 10);
            task.status_word = TransferStatusEnum::FAILED;
        }
        req_map_[task.req_handle] =
            std::make_pair(task.status_word, task.batch_size - 1);
    }
    return Status::OK();
}

void AscendDirectTransport::disconnect(const std::string &remote_hixl,
                                       int32_t timeout_in_millis) {
    if (auto_connect_) {
        auto status = hixl_->Disconnect(remote_hixl.c_str(), timeout_in_millis);
        if (status != hixl::SUCCESS) {
            LOG(ERROR) << "Failed to disconnect to: " << remote_hixl
                       << ", status: " << status
                       << ", errmsg: " << aclGetRecentErrMsg();
        }
        {
            std::lock_guard<std::mutex> lock(connection_mutex_);
            connected_segments_.erase(remote_hixl);
        }
        return;
    }
    std::lock_guard<std::mutex> lock(connection_mutex_);
    auto it = connected_segments_.find(remote_hixl);
    if (it == connected_segments_.end()) {
        LOG(INFO) << "Target hixl engine: " << remote_hixl
                  << " is not connected.";
    } else {
        auto status = hixl_->Disconnect(remote_hixl.c_str(), timeout_in_millis);
        connected_segments_.erase(remote_hixl);
        if (status != hixl::SUCCESS) {
            LOG(ERROR) << "Failed to disconnect to: " << remote_hixl
                       << ", status: " << status
                       << ", errmsg: " << aclGetRecentErrMsg();
        }
    }
}

Status AscendDirectTransport::addMemoryBuffer(BufferDesc &desc,
                                              const MemoryOptions &options) {
    desc.transports.push_back(TransportType::AscendDirect);
    hixl::MemDesc mem_desc{};
    mem_desc.addr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(desc.addr));
    mem_desc.len = desc.length;
    hixl::MemType mem_type;
    auto mem_type_status = getHixlMemType(desc, mem_type);
    if (!mem_type_status.ok()) {
        return mem_type_status;
    }
    hixl::MemHandle mem_handle;
    auto hixl_ret = hixl_->RegisterMem(mem_desc, mem_type, mem_handle);
    if (hixl_ret != hixl::SUCCESS) {
        LOG(ERROR) << "hixl_ret:" << hixl_ret
                   << ", errmsg: " << aclGetRecentErrMsg();
        return Status::InternalError("Register failed for addr:" +
                                     std::to_string(desc.addr));
    }
    LOG(INFO) << "AscendDirectTransport register mem addr:" << desc.addr
              << ", length:" << desc.length << ", location:" << desc.location
              << ", mem type:"
              << (mem_type == hixl::MEM_HOST ? "host" : "device");
    std::lock_guard<std::mutex> lock(mem_handle_mutex_);
    addr_to_mem_handle_[desc.addr] = mem_handle;
    return Status::OK();
}

Status AscendDirectTransport::removeMemoryBuffer(BufferDesc &desc) {
    std::vector<std::string> remote_hixls;
    {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        remote_hixls.assign(connected_segments_.begin(),
                            connected_segments_.end());
    }
    for (const auto &remote_hixl : remote_hixls) {
        disconnect(remote_hixl, connect_timeout_);
    }

    std::lock_guard<std::mutex> lock(mem_handle_mutex_);
    auto addr = desc.addr;
    if (addr_to_mem_handle_.find(addr) != addr_to_mem_handle_.end()) {
        (void)hixl_->DeregisterMem(addr_to_mem_handle_[addr]);
        addr_to_mem_handle_.erase(addr);
    }
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake
