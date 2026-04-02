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

#include "tent/transport/ascend/utils.h"

#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <glog/logging.h>

namespace mooncake {
namespace tent {

std::string hixlTransferStatusToString(hixl::TransferStatus status) {
    switch (status) {
        case hixl::TransferStatus::WAITING:
            return "WAITING";
        case hixl::TransferStatus::COMPLETED:
            return "COMPLETED";
        case hixl::TransferStatus::TIMEOUT:
            return "TIMEOUT";
        case hixl::TransferStatus::FAILED:
            return "FAILED";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int>(status)) + ")";
    }
}

uint16_t findListenPort(int32_t dev_id) {
    constexpr int base_port = 20000;

    char *rt_visible_devices = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    if (rt_visible_devices) {
        std::vector<std::string> device_list;
        std::stringstream ss(rt_visible_devices);
        std::string item;
        while (std::getline(ss, item, ',')) {
            device_list.push_back(item);
        }
        if (dev_id < static_cast<int32_t>(device_list.size())) {
            try {
                dev_id = std::stoi(device_list[dev_id]);
            } catch (const std::exception &e) {
                LOG(WARNING) << "ASCEND_RT_VISIBLE_DEVICES is not valid, value:"
                             << rt_visible_devices;
            }
        } else {
            LOG(WARNING) << "Device id is " << dev_id
                         << ", ASCEND_RT_VISIBLE_DEVICES is "
                         << rt_visible_devices << ", which is unexpected.";
        }
    }

    static std::random_device rand_gen;
    const int min_port = base_port + dev_id * 1000;
    const int max_port = base_port + (dev_id + 1) * 1000;
    LOG(INFO) << "Find available between " << min_port << " and " << max_port;

    constexpr int max_attempts = 500;
    std::uniform_int_distribution<> rand_dist(min_port, max_port);
    int sockfd;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int port = rand_dist(rand_gen);
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1) {
            continue;
        }
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                       sizeof(timeout))) {
            close(sockfd);
            sockfd = -1;
            continue;
        }
        sockaddr_in bind_address;
        memset(&bind_address, 0, sizeof(sockaddr_in));
        bind_address.sin_family = AF_INET;
        bind_address.sin_port = htons(port);
        bind_address.sin_addr.s_addr = INADDR_ANY;
        if (bind(sockfd, (sockaddr *)&bind_address, sizeof(sockaddr_in)) < 0) {
            close(sockfd);
            sockfd = -1;
            continue;
        }
        close(sockfd);
        return port;
    }
    return 0;
}

std::string getHostIpFromSegmentName(const std::string &segment_name) {
    const size_t colon_pos = segment_name.find(':');
    if (colon_pos == std::string::npos) {
        return segment_name;
    }
    return segment_name.substr(0, colon_pos);
}

void initializeHixlTask(HixlTask &task, const Request &request) {
    task.target_addr = request.target_offset;
    task.request = request;
    task.status_word = TransferStatusEnum::PENDING;
    task.transferred_bytes = 0;
}

hixl::TransferOpDesc buildTransferOpDesc(const HixlTask &task) {
    hixl::TransferOpDesc op_desc{};
    op_desc.local_addr = reinterpret_cast<uintptr_t>(task.request.source);
    op_desc.remote_addr =
        reinterpret_cast<uintptr_t>(task.request.target_offset);
    op_desc.len = task.request.length;
    return op_desc;
}

Status getHixlMemType(const BufferDesc &desc, hixl::MemType &mem_type) {
    if (desc.location.starts_with("cpu")) {
        mem_type = hixl::MEM_HOST;
        return Status::OK();
    }
    if (desc.location.starts_with("npu")) {
        mem_type = hixl::MEM_DEVICE;
        return Status::OK();
    }
    if (desc.location != kWildcardLocation) {
        LOG(ERROR) << "location:" << desc.location << " is not supported.";
        return Status::InvalidArgument("Invalid location of addr:" +
                                       std::to_string(desc.addr));
    }

    aclrtPtrAttributes attributes;
    auto ret = aclrtPointerGetAttributes(reinterpret_cast<void *>(desc.addr),
                                         &attributes);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "aclrtPointerGetAttributes failed, ret:" << ret;
        return Status::InvalidArgument("Invalid location of addr:" +
                                       std::to_string(desc.addr));
    }
    mem_type = attributes.location.type == ACL_MEM_LOCATION_TYPE_HOST
                   ? hixl::MEM_HOST
                   : hixl::MEM_DEVICE;
    return Status::OK();
}

Status getMemcpyKind(Request::OpCode opcode, const Request &request,
                     aclrtMemcpyKind &kind) {
    auto remote_ptr = reinterpret_cast<void *>(request.target_offset);

    aclrtPtrAttributes src_attributes;
    auto ret = aclrtPointerGetAttributes(request.source, &src_attributes);
    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtPointerGetAttributes failed, ret:" << ret;
        return Status::InternalError("Failed to get source pointer attributes.");
    }

    aclrtPtrAttributes dst_attributes;
    ret = aclrtPointerGetAttributes(remote_ptr, &dst_attributes);
    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtPointerGetAttributes failed, ret:" << ret;
        return Status::InternalError("Failed to get target pointer attributes.");
    }

    if (src_attributes.location.type != ACL_MEM_LOCATION_TYPE_HOST &&
        src_attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE) {
        LOG(ERROR) << "location of local addr is not supported.";
        return Status::InvalidArgument("Invalid location of local addr.");
    }
    if (dst_attributes.location.type != ACL_MEM_LOCATION_TYPE_HOST &&
        dst_attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE) {
        LOG(ERROR) << "location of remote addr is not supported.";
        return Status::InvalidArgument("Invalid location of remote addr.");
    }

    if (src_attributes.location.type == ACL_MEM_LOCATION_TYPE_HOST &&
        dst_attributes.location.type == ACL_MEM_LOCATION_TYPE_HOST) {
        kind = ACL_MEMCPY_HOST_TO_HOST;
    } else if (src_attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE &&
               dst_attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE) {
        kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    } else if (src_attributes.location.type == ACL_MEM_LOCATION_TYPE_HOST) {
        kind = opcode == Request::OpCode::WRITE ? ACL_MEMCPY_HOST_TO_DEVICE
                                                : ACL_MEMCPY_DEVICE_TO_HOST;
    } else {
        kind = opcode == Request::OpCode::WRITE ? ACL_MEMCPY_DEVICE_TO_HOST
                                                : ACL_MEMCPY_HOST_TO_DEVICE;
    }
    return Status::OK();
}

aclrtMemcpyBatchAttr buildMemcpyBatchAttr(aclrtMemcpyKind kind,
                                          int32_t device_logic_id) {
    auto device_loc = aclrtMemLocation{
        static_cast<uint32_t>(device_logic_id),
        aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE};
    auto host_loc =
        aclrtMemLocation{0, aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST};
    if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        return aclrtMemcpyBatchAttr{host_loc, device_loc, {}};
    }
    return aclrtMemcpyBatchAttr{device_loc, host_loc, {}};
}

}  // namespace tent
}  // namespace mooncake
