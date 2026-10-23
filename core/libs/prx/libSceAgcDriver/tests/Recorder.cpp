#include "prx/libSceAgcDriver/Execution/include/PipelineLinkOptimization.hpp"
#include "Triangle_frag_spv.h"
#include "Triangle_vert_spv.h"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/QueueGate.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawInputMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawTicket.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthLease.hpp"
#include "prx/libSceAgcDriver/Graphics/include/FieldCount.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/InputLease.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ParallelCompare.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TargetLease.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/BdaFeatures.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "ResidentPresent.hpp"
#include "ImageTableAbi.hpp"
#include "HostImportLifetime.hpp"
#include "SampleLod_spv.h"
#include "BufferUpdate_spv.h"
#include "SampleArray_spv.h"
#include "Cover_vert_spv.h"
#include "Cover_frag_spv.h"
#include "VertexRange_vert_spv.h"
#include "VertexRangeProducer_frag_spv.h"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "BdaAbi.hpp"
#include "ShaderCacheSourceVersion.hpp"
#include <SDL_loadso.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <functional>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelMprotect(const void*, std::size_t, int);
}

using namespace AgcDriver::Graphics;
using AgcDriver::GuestMemory::GpuMutex;

void* AllocateWatched(std::size_t bytes, std::size_t alignment) {
    if (!AgcDriver::GuestMemory::WriteWatched()) return nullptr;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, alignment);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    if (!AgcDriver::GuestMemory::Watched(reinterpret_cast<std::uint64_t>(block), bytes)) throw std::runtime_error("the watched block is not watched");
    return block;
}

void ReleaseWatched(void* block, std::size_t bytes) {
#ifdef _WIN32
    GuestArena::GuestArenaReset_nid_postfix(block, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(block, bytes);
#else
    munmap(block, bytes);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
#endif
}

bool collectMappingHistoryTests() {
    constexpr std::size_t bytes = 131072;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes, 65536));
    if (memory == nullptr) return false;
    GuestAllocations::Mutation().Add(memory, bytes, true, true, false);
    struct Cleanup {
        std::uint8_t* memory;
        ~Cleanup() {
            GuestAllocations::Mutation().Remove(memory);
            ReleaseWatched(memory, bytes);
        }
    } cleanup{memory};
    const auto address = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    const auto before = AgcDriver::GuestMemory::CollectWrites(address, bytes);
    Require(before != 0, "the mapping fixture has no complete write history");
    memory[4096] = 0x22;
    GuestAllocations::Mutation().Remove(memory);
    GuestAllocations::Mutation().Add(memory, bytes, true, true, false);
    Require(AgcDriver::GuestMemory::CollectWrites(address, bytes) != 0, "the replacement mapping lost write tracking");
    Require(!AgcDriver::GuestMemory::UnchangedSince(address, bytes, before), "a cached collect hid a CPU store across allocation replacement");
    Require(memory[4096] == 0x22, "collecting the replacement mapping changed its CPU bytes");
    return true;
}

bool partialWatchRestorationTests() {
#ifdef _WIN32
    constexpr std::size_t bytes = 131072;
    constexpr std::size_t page = 4096;
    constexpr std::size_t restoredOffset = 4 * page;
    constexpr std::size_t restoredBytes = 4 * page;
    if (!AgcDriver::GuestMemory::WriteWatched()) return false;
    auto* memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536));
    if (memory == nullptr) return false;
    GuestArena::GuestArenaCommit_nid_postfix(memory, bytes, PAGE_READWRITE, restoredBytes);
    struct Cleanup {
        std::uint8_t* memory;
        ~Cleanup() { ReleaseWatched(memory, bytes); }
    } cleanup{memory};
    const auto address = reinterpret_cast<std::uint64_t>(memory);
    AgcDriver::GuestMemory::Unwatch(address, 65536);
    Require(!AgcDriver::GuestMemory::Watched(address, page), "the partial remap fixture was not excluded");
    memory[page] = 0x22;
    AgcDriver::GuestMemory::MarkWritten(address + page, 1);
    Require(AgcDriver::GuestMemory::UntrackedHint(address + page, 1) != 0, "the excluded remap fixture has no driver hint");
    GuestArena::GuestArenaReset_nid_postfix(memory + restoredOffset, restoredBytes);
    GuestArena::GuestArenaCommit_nid_postfix(memory + restoredOffset, restoredBytes, PAGE_READWRITE, restoredBytes);
    Require(AgcDriver::GuestMemory::Watched(address + restoredOffset, restoredBytes), "a partial private remap did not restore its write watch");
    Require(!AgcDriver::GuestMemory::Watched(address, restoredOffset), "a partial remap restored an excluded prefix");
    Require(!AgcDriver::GuestMemory::Watched(address + restoredOffset + restoredBytes, page), "a partial remap restored an excluded suffix");
    Require(AgcDriver::GuestMemory::UntrackedHint(address + page, 1) == 0, "a partial remap retained a stale driver hint");
    std::memset(memory + restoredOffset, 0x33, restoredBytes);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    const auto generation = AgcDriver::GuestMemory::CollectWrites(address + restoredOffset, restoredBytes);
    Require(generation != 0, "the private remap has no collected CPU write history");
    memory[restoredOffset + page] = 0x44;
    AgcDriver::GuestMemory::BumpCollectEpoch();
    Require(AgcDriver::GuestMemory::CollectWrites(address + restoredOffset, restoredBytes) != 0, "the private remap stopped collecting CPU writes");
    Require(!AgcDriver::GuestMemory::UnchangedSince(address + restoredOffset, restoredBytes, generation), "the private remap hid a genuine CPU write");
    GuestArena::GuestArenaReset_nid_postfix(memory, restoredBytes);
    GuestArena::GuestArenaCommit_nid_postfix(memory, restoredBytes, PAGE_READWRITE, restoredBytes);
    Require(AgcDriver::GuestMemory::Watched(address, restoredBytes), "the first private pages did not regain coverage");
    Require(!AgcDriver::GuestMemory::Watched(address + 12 * page, page), "restoring the first pages restored an excluded tail");
    memory[12 * page] = 0x55;
    AgcDriver::GuestMemory::MarkWritten(address + 12 * page, 1);
    AgcDriver::GuestMemory::Unwatch(address + 12 * page, page);
    Require(AgcDriver::GuestMemory::UntrackedHint(address + 12 * page, 1) == 0, "excluding a partially watched block invented a write-watch hint");
    return true;
#else
    return false;
#endif
}

std::optional<std::size_t> selectDevice(std::span<const VkPhysicalDeviceProperties> devices, std::string_view request) {
    const auto rankDeviceType = [](VkPhysicalDeviceType type) {
        switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
            default: return 0;
        }
    };
    std::optional<std::size_t> selected;
    int selectedRank = -1;
    for (std::size_t index = 0; index < devices.size(); ++index) {
        const auto& candidate = devices[index];
        if (candidate.apiVersion < VK_API_VERSION_1_1) continue;
        const std::string_view name(candidate.deviceName);
        if (!request.empty() && std::search(name.begin(), name.end(), request.begin(), request.end(), [](char left, char right) {
            return std::tolower(static_cast<unsigned char>(left)) == std::tolower(static_cast<unsigned char>(right));
        }) == name.end()) continue;
        const int rank = request.empty() ? rankDeviceType(candidate.deviceType) : 4;
        if (rank <= selectedRank) continue;
        selected = index;
        selectedRank = rank;
    }
    return selected;
}

bool hintMappingHistoryTests() {
#ifdef _WIN32
    constexpr std::size_t bytes = 131072;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes, 65536));
    if (memory == nullptr) return false;
    GuestAllocations::Mutation().Add(memory, bytes, true, true, false);
    struct Cleanup {
        std::uint8_t* memory;
        ~Cleanup() {
            GuestAllocations::Mutation().Remove(memory);
            ReleaseWatched(memory, bytes);
        }
    } cleanup{memory};
    const auto address = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    AgcDriver::GuestMemory::BumpCollectEpoch();
    AgcDriver::GuestMemory::Unwatch(address, bytes);
    const auto before = AgcDriver::GuestMemory::UntrackedHint(address, bytes);
    Require(before != 0, "the hint mapping fixture has no write hint");
    Require(AgcDriver::GuestMemory::UntrackedHint(address, bytes) == before, "an unchanged hint range advanced its history");
    memory[4096] = 0x22;
    GuestAllocations::Mutation().Remove(memory);
    GuestAllocations::Mutation().Add(memory, bytes, true, true, false);
    const auto after = AgcDriver::GuestMemory::UntrackedHint(address, bytes);
    Require(after > before, "a cached hint walk hid a CPU store across allocation replacement");
    Require(memory[4096] == 0x22 && memory[bytes - 4096] == 0x11, "collecting replacement hints changed guest bytes");
    GuestAllocations::Mutation().Remove(memory);
    GuestAllocations::Mutation().Add(memory, bytes, true, true, false);
    Require(AgcDriver::GuestMemory::UntrackedHint(address, bytes) == after, "allocation replacement invented a CPU write hint");
    return true;
#else
    return false;
#endif
}

void deviceSelectionTests() {
    const auto properties = [](const char* name, VkPhysicalDeviceType type) {
        VkPhysicalDeviceProperties result{};
        result.apiVersion = VK_API_VERSION_1_1;
        result.deviceType = type;
        std::memcpy(result.deviceName, name, std::strlen(name) + 1u);
        return result;
    };
    std::array devices{
        properties("AMD Radeon Graphics", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
        properties("NVIDIA GeForce RTX 5080", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU),
        properties("Virtual GPU", VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU),
        properties("CPU", VK_PHYSICAL_DEVICE_TYPE_CPU),
    };
    Require(!selectDevice({}, {}).has_value(), "an empty device list selected a device");
    Require(selectDevice(devices, {}) == 1u, "default selection did not prefer the discrete GPU");
    Require(selectDevice(devices, "aMd rAdEoN") == 0u, "a requested integrated GPU lost to an unrelated discrete GPU");
    Require(selectDevice(devices, "rtx 5080") == 1u, "a partial GPU name did not match without regard to case");
    Require(selectDevice(devices, "Virtual") == 2u, "a requested virtual GPU was not selected");
    Require(selectDevice(devices, "cpu") == 3u, "a requested CPU device was not selected");
    Require(!selectDevice(devices, "missing").has_value(), "an unmatched request selected another GPU");
    Require(!selectDevice(devices, "NVIDIA GeForce RTX 5080 with a longer name").has_value(), "a request longer than the device name matched");
    devices[0].apiVersion = VK_API_VERSION_1_0;
    Require(!selectDevice(devices, "AMD").has_value(), "a requested device without Vulkan 1.1 was selected");
    Require(selectDevice(devices, {}) == 1u, "an unsupported device changed default selection");
    devices[0] = devices[1];
    Require(selectDevice(devices, {}) == 0u && selectDevice(devices, "NVIDIA") == 0u, "equally ranked matches did not retain enumeration order");
    for (auto& device : devices) device.apiVersion = VK_API_VERSION_1_0;
    Require(!selectDevice(devices, {}).has_value(), "a list without Vulkan 1.1 selected a device");
}

// A compute-capable device with host imports (VK_EXT_external_memory_host) when the host offers
// them, as the driver creates its own; the recorder's batches need a real queue and real fences.
class Device {
public:
    // `timeline`: VK_KHR_timeline_semaphore enabled when the device has it (the submit thread tests'
    // recorders wait on their timeline; Timeline() says whether it was).
    explicit Device(bool settleOnOtherFamily = false, bool timeline = false) {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        Require(library != nullptr, "cannot load Vulkan");
        try {
            instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
            Require(instanceProc != nullptr, "missing Vulkan instance resolver");
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &application;
            Check(function<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr, &instance), "vkCreateInstance");
            std::uint32_t count = 0;
            const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
            Check(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            Require(count != 0, "no Vulkan device");
            std::vector<VkPhysicalDevice> devices(count);
            Check(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            const auto physicalProperties = function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
            std::vector<VkPhysicalDeviceProperties> deviceProperties(devices.size());
            for (std::size_t index = 0; index < devices.size(); ++index) physicalProperties(devices[index], &deviceProperties[index]);
            const char* requestedName = std::getenv("ANYPS5_GPU");
            const auto selectedIndex = selectDevice(deviceProperties, requestedName != nullptr ? requestedName : "");
            Require(selectedIndex.has_value(), requestedName != nullptr && *requestedName != '\0' ? std::string("no Vulkan 1.1 device matching ANYPS5_GPU=\"") + requestedName + "\"" : "no Vulkan 1.1 device");
            context.physical = devices[*selectedIndex];
            const auto& selected = deviceProperties[*selectedIndex];
            std::cout << "Vulkan device: " << selected.deviceName << '\n';
            const auto extensions = function<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
            Check(extensions(context.physical, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
            std::vector<VkExtensionProperties> available(count);
            Check(extensions(context.physical, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
            const auto hasExtension = [&](const char* name) {
                for (const auto& extension : available) {
                    if (std::strcmp(extension.extensionName, name) == 0) return true;
                }
                return false;
            };
            auto bytes = AgcDriver::QueryBdaByteFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            auto address = AgcDriver::QueryBdaFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
            queues(context.physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            queues(context.physical, &count, families.data());
            std::uint32_t family = 0;
            while (family < count && (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) ++family;
            Require(family < count, "no Vulkan compute queue");
            const float priority = 1;
            const float priorities[2]{1, 1};
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            VkDeviceQueueCreateInfo settle{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            settle.queueCount = 1;
            settle.pQueuePriorities = &priority;
            constexpr std::uint32_t noSettleFamily = UINT32_MAX;
            std::uint32_t settleFamily = noSettleFamily;
#ifdef _WIN32
            if (!settleOnOtherFamily && families[family].queueCount >= 2) {
                queue.queueCount = 2;
                queue.pQueuePriorities = priorities;
                settleFamily = family;
            } else {
                const auto copies = [&](std::uint32_t index) { return index != family && families[index].queueCount != 0 && (families[index].queueFlags & (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) != 0; };
                for (std::uint32_t index = 0; index < families.size() && settleFamily == noSettleFamily; ++index) {
                    if (copies(index) && (families[index].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) == 0) settleFamily = index;
                }
                for (std::uint32_t index = 0; index < families.size() && settleFamily == noSettleFamily; ++index) {
                    if (copies(index)) settleFamily = index;
                }
                settle.queueFamilyIndex = settleFamily;
            }
#endif
            VkPhysicalDeviceFeatures enabled{};
            enabled.shaderInt64 = VK_TRUE;
            graphics = (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            VkPhysicalDeviceFeatures supported{};
            function<PFN_vkGetPhysicalDeviceFeatures>("vkGetPhysicalDeviceFeatures")(context.physical, &supported);
            enabled.occlusionQueryPrecise = supported.occlusionQueryPrecise;
            context.occlusionQueryPrecise = supported.occlusionQueryPrecise == VK_TRUE;
            enabled.shaderStorageImageReadWithoutFormat = supported.shaderStorageImageReadWithoutFormat;
            enabled.shaderStorageImageWriteWithoutFormat = supported.shaderStorageImageWriteWithoutFormat;
            enabled.shaderStorageImageMultisample = supported.shaderStorageImageMultisample;
            context.storageImageMultisample = supported.shaderStorageImageMultisample == VK_TRUE;
            context.singlePassStorage = supported.shaderStorageImageReadWithoutFormat == VK_TRUE && supported.shaderStorageImageWriteWithoutFormat == VK_TRUE;
            address.pNext = &bytes;
            std::vector<const char*> extensionsEnabled{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_8BIT_STORAGE_EXTENSION_NAME};
            VkPhysicalDeviceDriverProperties driverProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
            if (hasExtension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
                extensionsEnabled.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
                VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT, &driverProperties};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &hostProperties};
                function<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(context.physical, &properties);
                context.hostImportAlignment = hostProperties.minImportedHostPointerAlignment;
            }
            if (driverProperties.driverID != VK_DRIVER_ID_NVIDIA_PROPRIETARY && hasExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) && hasExtension(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)) {
                extensionsEnabled.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
                extensionsEnabled.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
                context.dmaBufImport = true;
            }
            VkPhysicalDeviceImageViewMinLodFeaturesEXT minLod{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
            if (hasExtension(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &minLod};
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
                context.imageViewMinLod = minLod.minLod == VK_TRUE;
            }
            minLod = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
            minLod.minLod = VK_TRUE;
            if (context.imageViewMinLod) {
                extensionsEnabled.push_back(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
                minLod.pNext = address.pNext;
                address.pNext = &minLod;
            }
            VkPhysicalDeviceDynamicRenderingFeaturesKHR rendering{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
            VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamicState{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, &rendering};
            VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT library{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, &dynamicState};
            const std::array<const char*, 6> libraryExtensions{VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME};
            if (std::getenv("APS5_NO_GPL") == nullptr && std::all_of(libraryExtensions.begin(), libraryExtensions.end(), hasExtension)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &library};
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
                VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT libraryProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &libraryProperties};
                function<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(context.physical, &properties);
                context.graphicsPipelineLibrary = library.graphicsPipelineLibrary == VK_TRUE && dynamicState.extendedDynamicState == VK_TRUE && rendering.dynamicRendering == VK_TRUE && libraryProperties.graphicsPipelineLibraryFastLinking == VK_TRUE && properties.properties.limits.maxPushConstantsSize >= PipelinePushConstantBytes;
                if (context.graphicsPipelineLibrary && hasExtension(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)) {
                    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
                    VkPhysicalDeviceProperties2 driverProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
                    function<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(context.physical, &driverProperties);
                    context.pipelineLinkTimeOptimization = AgcDriver::PipelineLinkTimeOptimizationUsable(driver.driverID);
                }
            }
            rendering = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR, address.pNext, VK_TRUE};
            library = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, &dynamicState, VK_TRUE};
            dynamicState = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, &rendering, VK_TRUE};
            if (context.graphicsPipelineLibrary) {
                extensionsEnabled.insert(extensionsEnabled.end(), libraryExtensions.begin(), libraryExtensions.end());
                address.pNext = &library;
            }
            VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
            if (hasExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &indexing};
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
                context.descriptorIndexing = indexing.shaderSampledImageArrayNonUniformIndexing == VK_TRUE && indexing.shaderStorageImageArrayNonUniformIndexing == VK_TRUE;
                context.partiallyBound = indexing.descriptorBindingPartiallyBound == VK_TRUE;
                context.variableDescriptorCount = indexing.descriptorBindingVariableDescriptorCount == VK_TRUE;
                context.runtimeDescriptorArray = indexing.runtimeDescriptorArray == VK_TRUE;
            }
            indexing = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
            if (context.descriptorIndexing && context.partiallyBound && context.variableDescriptorCount && context.runtimeDescriptorArray) {
                extensionsEnabled.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
                indexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
                indexing.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
                indexing.descriptorBindingPartiallyBound = VK_TRUE;
                indexing.descriptorBindingVariableDescriptorCount = VK_TRUE;
                indexing.runtimeDescriptorArray = VK_TRUE;
                indexing.pNext = address.pNext;
                address.pNext = &indexing;
            } else {
                context.descriptorIndexing = context.partiallyBound = context.variableDescriptorCount = context.runtimeDescriptorArray = false;
            }
            VkPhysicalDeviceRobustness2FeaturesEXT robustness{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
            if (hasExtension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &robustness};
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
                nullDescriptors = robustness.nullDescriptor == VK_TRUE;
            }
            robustness = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
            robustness.nullDescriptor = VK_TRUE;
            if (nullDescriptors) {
                extensionsEnabled.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
                robustness.pNext = address.pNext;
                address.pNext = &robustness;
            }
            VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timelineFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR};
            if (timeline && hasExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
                VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &timelineFeatures};
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(context.physical, &features);
                timelineEnabled = timelineFeatures.timelineSemaphore == VK_TRUE;
            }
            timelineFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR};
            timelineFeatures.timelineSemaphore = VK_TRUE;
            if (timelineEnabled) {
                extensionsEnabled.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
                timelineFeatures.pNext = address.pNext;
                address.pNext = &timelineFeatures;
            }
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &address};
            const std::array<VkDeviceQueueCreateInfo, 2> queueInfos{queue, settle};
            device.queueCreateInfoCount = settleFamily != noSettleFamily && settleFamily != family ? 2 : 1;
            device.pQueueCreateInfos = queueInfos.data();
            device.enabledExtensionCount = static_cast<std::uint32_t>(extensionsEnabled.size());
            device.ppEnabledExtensionNames = extensionsEnabled.data();
            device.pEnabledFeatures = &enabled;
            Check(function<PFN_vkCreateDevice>("vkCreateDevice")(context.physical, &device, nullptr, &context.device), "vkCreateDevice");
            context.deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(context.physical, &context.memory);
            VkPhysicalDeviceProperties properties{};
            function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &properties);
            std::cout << "Vulkan device: " << properties.deviceName << '\n';
            context.limits = properties.limits;
            context.bufferDeviceAddress = true;
            context.imageTableCapacity = ImageTableCapacity(context);
            context.formatProperties = function<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            context.imageFormatProperties = function<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
            context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, family, 0, &context.queue);
            context.queueFamily = family;
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = family;
            Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.pool), "vkCreateCommandPool");
#ifdef _WIN32
            if (settleFamily != noSettleFamily) {
                context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, settleFamily, settleFamily == family ? 1 : 0, &context.settleQueue);
                pool.queueFamilyIndex = settleFamily;
                Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.settlePool), "vkCreateCommandPool settle");
                context.settleFamily = settleFamily;
                context.settleQueueMutex = &settleMutex;
            }
#endif
        } catch (...) {
            release();
            throw;
        }
    }

    ~Device() { release(); }
    bool NullDescriptors() const { return nullDescriptors; }
    bool Timeline() const { return timelineEnabled; }
    const Context& GetContext() const { return context; }
    bool Graphics() const { return graphics; }
    bool Timestamps() const {
        std::uint32_t count = 0;
        const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
        queues(context.physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        queues(context.physical, &count, families.data());
        return families.at(context.queueFamily).timestampValidBits != 0;
    }
    VkPhysicalDeviceProperties Properties() const {
        VkPhysicalDeviceProperties properties{};
        function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &properties);
        return properties;
    }
    void WaitQueue() const { Check(context.Function<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(context.queue), "vkQueueWaitIdle"); }

private:
    template<typename TFunction>
    TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(instanceProc(instance, name));
        Require(result != nullptr, name);
        return result;
    }

    void release() noexcept {
        if (context.device != VK_NULL_HANDLE) ClearHostImports(context.device);
        if (context.settlePool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, context.settlePool, nullptr);
        if (context.pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, context.pool, nullptr);
        context.bufferPool.reset();
        if (context.device != VK_NULL_HANDLE) {
            DestroyShadows(context.device);
            function<PFN_vkDestroyDevice>("vkDestroyDevice")(context.device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        if (library != nullptr) SDL_UnloadObject(library);
    }

    void* library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    Context context{};
    bool graphics = false;
    std::mutex settleMutex;
    bool nullDescriptors = false;
    bool timelineEnabled = false;
};

using Kind = Recorder::ReadKind;

void copiedDirectOrderingTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    context.nullDescriptors = device.NullDescriptors();
    if (context.hostImportAlignment == 0 || !context.nullDescriptors || !AgcDriver::GuestMemory::WriteWatched()) {
        std::cout << "Host imports, null descriptors or write watching unavailable: copied/direct ordering test skipped\n";
        return;
    }
    recorder.Sync();
    const auto bytes = static_cast<std::size_t>(std::max<VkDeviceSize>(131072, context.hostImportAlignment));
    auto* block = static_cast<std::uint32_t*>(AllocateWatched(bytes, bytes));
    Require(block != nullptr, "allocate copied/direct ordering memory");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    std::vector<std::shared_ptr<ShaderResources>> writers;
    context.copiedWriters = &writers;
    struct Release {
        const Context& context;
        Recorder& recorder;
        std::uint32_t* block;
        std::size_t bytes;
        std::vector<std::shared_ptr<ShaderResources>>& writers;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkShaderModule module = VK_NULL_HANDLE;
        std::shared_ptr<ShaderResources> layoutOwner;
        ~Release() {
            recorder.Sync();
            writers.clear();
            DrawCopiedWriters()->clear();
            if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
            if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
            if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            static_cast<void>(HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes));
            recorder.Sync();
            ReleaseWatched(block, bytes);
        }
    } release{context, recorder, block, bytes, writers};
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    std::memset(block, 0, bytes);
    Require(HostImportFor(context, address, bytes) != nullptr, "import copied/direct ordering memory");
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, 1024, 0x31000000u};
    binding.bufferWritten = {true};
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    auto direct = std::make_shared<ShaderResources>(context, compute);
    release.layoutOwner = direct;
    Require(!direct->HasCopiedWrites(), "ordering consumer did not bind directly");
    const auto setLayout = direct->Layout();
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(std::uint32_t)};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &push;
    Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &release.layout), "ordering pipeline layout");
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = sizeof(BUFFER_UPDATE_SPV);
    moduleInfo.pCode = BUFFER_UPDATE_SPV;
    Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &release.module), "ordering shader module");
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, release.module, "main", nullptr};
    pipelineInfo.layout = release.layout;
    Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &release.pipeline), "ordering pipeline");
    const auto update = [&](const std::shared_ptr<ShaderResources>& resources, std::uint32_t delta, std::uint32_t count = 1) {
        const auto commands = recorder.Commands();
        recorder.Keep(resources);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, release.pipeline);
        resources->Bind(commands, VK_PIPELINE_BIND_POINT_COMPUTE, release.layout);
        context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, release.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(delta), &delta);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, count, 1, 1);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
        resources->MarkGpuWrites(recorder);
    };
    for (int mode = 0; mode < 3; ++mode) {
        *block = 10;
        auto copiedContext = context;
        copiedContext.hostImportAlignment = 0;
        copiedContext.dmaBufImport = false;
        auto copied = std::make_shared<ShaderResources>(copiedContext, compute);
        Require(copied->HasCopiedWrites(), "ordering producer did not take the copied path");
        auto& pending = mode == 2 ? *DrawCopiedWriters() : writers;
        pending.push_back(copied);
        update(copied, 1);
        recorder.OnComplete([copied, &pending] {
            copied->WriteBackBuffers();
            std::erase(pending, copied);
        });
        Require(!recorder.Idle() && *block == 10, "copied producer was completed before the consumer test");
        if (mode == 0) direct = std::make_shared<ShaderResources>(context, compute);
        else Require(direct->Revalidate(std::span(&compute, 1)), "the imported template could not be revalidated");
        update(direct, 2);
        recorder.Sync();
        Require(*block == 13, mode == 0 ? "fresh direct consumer ran before copied write-back" : "reused direct consumer ran before copied write-back");
    }
    *block = 20;
    block[1024] = 30;
    auto copiedContext = context;
    copiedContext.hostImportAlignment = 0;
    copiedContext.dmaBufImport = false;
    auto copied = std::make_shared<ShaderResources>(copiedContext, compute);
    writers.push_back(copied);
    update(copied, 1);
    recorder.OnComplete([copied, &writers] {
        copied->WriteBackBuffers();
        std::erase(writers, copied);
    });
    auto separateProgram = program;
    separateProgram.bindings[0].guestDescriptor[0] += 4096;
    const CompiledShader separateCompute{ShaderRecompiler::ShaderStage::Compute, &separateProgram, 0};
    auto separate = std::make_shared<ShaderResources>(context, separateCompute);
    Require(!writers.empty() && *block == 20, "a disjoint consumer waited for copied writes");
    update(separate, 2);
    recorder.Sync();
    Require(*block == 21 && block[1024] == 32, "disjoint copied and direct writes changed each other");
    const std::uint32_t label = 40;
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(address, std::as_bytes(std::span(&label, 1)), 1000, 0, false);
    Require(direct->Revalidate(std::span(&compute, 1)), "template reuse after a CPU completion label failed");
    update(direct, 2);
    recorder.Sync();
    Require(*block == 42, "direct consumer preceded an overlapping CPU completion label");
    update(direct, 1);
    const auto submissions = recorder.Submissions();
    Require(direct->Revalidate(std::span(&compute, 1)) && recorder.Submissions() == submissions, "ordinary direct work incurred a CPU write-back wait");
    update(direct, 2);
    recorder.Sync();
    Require(*block == 45, "direct consumers lost GPU-ordered updates");
    {
        std::array<std::uint32_t, 1024> unregistered{};
        struct Finish {
            Recorder& recorder;
            ~Finish() { recorder.Sync(); }
        } finish{recorder};
        const auto copiedAddress = reinterpret_cast<std::uint64_t>(unregistered.data());
        auto mixedProgram = program;
        auto& mixedBinding = mixedProgram.bindings[0];
        mixedBinding.count = 2;
        mixedBinding.guestDescriptor.insert(mixedBinding.guestDescriptor.end(), {static_cast<std::uint32_t>(copiedAddress), static_cast<std::uint32_t>(copiedAddress >> 32u) & 0xffffu, 1024, 0x31000000u});
        mixedBinding.bufferWritten = {true, true};
        const CompiledShader mixedCompute{ShaderRecompiler::ShaderStage::Compute, &mixedProgram, 0};
        for (const bool draw : {false, true}) {
            auto mixed = std::make_shared<ShaderResources>(context, mixedCompute);
            Require(mixed->HasCopiedWrites() && mixed->WritesOverlap(address, 4) && !mixed->CopiedWritesOverlap(address, 4) && mixed->CopiedWritesOverlap(copiedAddress, 4), "the mixed producer did not retain direct and CPU-copied writes");
            auto& pending = draw ? *DrawCopiedWriters() : writers;
            static_cast<void>(recorder.Commands());
            recorder.Keep(mixed);
            mixed->MarkGpuWrites(recorder);
            pending.push_back(mixed);
            auto completed = std::make_shared<unsigned>(0);
            recorder.OnComplete([mixed, &pending, completed] {
                mixed->WriteBackBuffers();
                std::erase(pending, mixed);
                ++*completed;
            });
            Require(!CompletePendingCpuWrites(context, address, 4), "a direct write in a mixed producer caused a CPU write-back wait");
            Require(*completed == 0 && !pending.empty(), "a disjoint copied write-back was completed while binding a direct buffer");
            Require(CompletePendingCpuWrites(context, copiedAddress, 4), "a CPU-copied write in a mixed producer was not completed");
            Require(*completed == 1 && pending.empty(), "the required CPU write-back did not release the mixed producer");
        }
    }
    {
        TextureDetiler detiler(context);
        auto imageContext = context;
        imageContext.detiler = &detiler;
        TextureCache textureCache(imageContext);
        imageContext.textureCache = &textureCache;
        Buffer readback(context, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        struct Clear {
            const Context& context;
            Recorder& recorder;
            ~Clear() { recorder.Sync(); ClearCachedTextures(context.device); }
        } clear{context, recorder};
        ShaderRecompiler::RecompileResult imageProgram;
        ShaderRecompiler::DescriptorBinding imageBinding;
        imageBinding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
        imageBinding.role = ShaderRecompiler::DescriptorRole::GuestImages;
        imageBinding.descriptorSet = 0;
        imageBinding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding;
        imageBinding.count = 1;
        imageBinding.imageSamplers = {0};
        imageBinding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
        imageBinding.guestDescriptor = {static_cast<std::uint32_t>(address >> 8), static_cast<std::uint32_t>(address >> 40) | (56u << 20) | (3u << 30), 15u | (63u << 14), 0x91b00fac, 0, 0, 0, 0};
        imageProgram.bindings.push_back(imageBinding);
        const CompiledShader imageCompute{ShaderRecompiler::ShaderStage::Compute, &imageProgram, 0};
        auto sampled = std::make_shared<ShaderResources>(imageContext, imageCompute);
        recorder.Sync();
        for (int reuse = 0; reuse < 2; ++reuse) {
            auto producer = std::make_shared<ShaderResources>(copiedContext, compute);
            writers.push_back(producer);
            update(producer, 1);
            recorder.OnComplete([producer, &writers] {
                producer->WriteBackBuffers();
                std::erase(writers, producer);
            });
            if (reuse == 0) sampled = std::make_shared<ShaderResources>(imageContext, imageCompute);
            else Require(sampled->Revalidate(std::span(&imageCompute, 1)), "sampled image template did not revalidate");
            const auto images = sampled->StorageImages();
            Require(images.size() == 1, "sampled image did not use imported texture storage");
            const auto commands = recorder.Commands();
            recorder.Keep(sampled);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {1, 1, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, images.front().first, VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Sync();
            std::uint32_t value = 0;
            std::memcpy(&value, readback.Bytes().data(), sizeof(value));
            Require(value == static_cast<std::uint32_t>(46 + reuse), "cached sampled image preceded a copied producer");
        }
        sampled.reset();
        ClearCachedTextures(context.device);
        const auto keys = address + 65536;
        const auto keyBytes = DccKeyBytes(65536);
        imageProgram.bindings[0].guestDescriptor[6] = (1u << 21) | (static_cast<std::uint32_t>((keys >> 8) & 0xffu) << 24);
        imageProgram.bindings[0].guestDescriptor[7] = static_cast<std::uint32_t>(keys >> 16);
        auto keyProgram = program;
        keyProgram.bindings[0].guestDescriptor = {static_cast<std::uint32_t>(keys), static_cast<std::uint32_t>(keys >> 32u) & 0xffffu, static_cast<std::uint32_t>(keyBytes), 0x31000000u};
        const CompiledShader keyCompute{ShaderRecompiler::ShaderStage::Compute, &keyProgram, 0};
        for (int reuse = 0; reuse < 2; ++reuse) {
            std::memset(reinterpret_cast<void*>(keys), 0, keyBytes);
            AgcDriver::GuestMemory::MarkWritten(keys, keyBytes);
            sampled = std::make_shared<ShaderResources>(imageContext, imageCompute);
            recorder.Sync();
            auto producer = std::make_shared<ShaderResources>(copiedContext, keyCompute);
            writers.push_back(producer);
            update(producer, 0x40404040, static_cast<std::uint32_t>(keyBytes / 4));
            recorder.OnComplete([producer, &writers] {
                producer->WriteBackBuffers();
                std::erase(writers, producer);
            });
            if (reuse == 0) sampled = std::make_shared<ShaderResources>(imageContext, imageCompute);
            else Require(sampled->Revalidate(std::span(&imageCompute, 1)), "metadata-only producer invalidated the sampled template");
            const auto images = sampled->StorageImages();
            Require(images.size() == 1, "metadata texture is not resident");
            const auto commands = recorder.Commands();
            recorder.Keep(sampled);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {1, 1, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, images.front().first, VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Sync();
            std::uint32_t value = 0;
            std::memcpy(&value, readback.Bytes().data(), sizeof(value));
            Require(value == 0xff000000, "cached sampled image ignored copied compression-metadata writes");
        }
    }
    std::cout << "Copied producers precede fresh and reused direct consumers\n";
}

void readTrackingTests(const Device& device, Recorder& recorder) {
    Require(Recorder::ReadTracking(), "read tracking is off (APS5_COPY_READ_TRACKING=0 set?)");
    Require(recorder.Idle() && !recorder.PendingReadOverlaps(0x10000, 16), "a fresh recorder reports a read");
    recorder.NotePendingRead(0x10000, 0x100, Kind::DispatchElement);
    Require(recorder.Recording(), "a read note did not open a batch");
    Require(recorder.PendingReadOverlaps(0x10080, 4) && recorder.PendingReadOverlaps(0xff00, 0x101) && recorder.PendingReadOverlaps(0x100ff, 1), "an open batch's read is not seen");
    Require(!recorder.PendingReadOverlaps(0x10100, 4) && !recorder.PendingReadOverlaps(0xff00, 0x100) && !recorder.PendingReadOverlaps(0x10000, 0), "a disjoint range is reported as read");
    const auto open = recorder.DescribePendingRead(0x10000, 4);
    Require(open.has_value() && open->open && !open->signaled && open->kind == Kind::DispatchElement && open->serial == recorder.Submissions() + 1, "the open batch's read is described wrongly");
    recorder.Submit();
    Require(!recorder.Recording() && !recorder.Idle(), "the batch is not in flight after Submit");
    Require(recorder.PendingReadOverlaps(0x10000, 4, false), "the raw scan does not see the in-flight batch's read");
    // An empty batch completes at once: once its fence signaled, its read is no reader any more,
    // although the batch stays in flight until it is reaped.
    device.WaitQueue();
    Require(!recorder.PendingReadOverlaps(0x10000, 4), "a signaled batch's read still refuses");
    Require(!recorder.Idle(), "an overlap query reaped the batch");
    const auto signaled = recorder.DescribePendingRead(0x10000, 4);
    Require(signaled.has_value() && !signaled->open && signaled->signaled && signaled->serial == recorder.Submissions(), "the signaled batch's read is described wrongly");
    const auto counts = Recorder::ReadCounts();
    Require(counts.noted == 1 && counts.staleIgnored >= 1 && counts.hits[static_cast<std::size_t>(Kind::DispatchElement)] >= 3 && counts.queries >= 6, "read counters are off");
    recorder.Sync();
    Require(recorder.Idle() && !recorder.PendingReadOverlaps(0x10000, 4, false) && !recorder.DescribePendingRead(0x10000, 4).has_value(), "a read outlived its batch");
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{{0x20000, 0x21000}, {0x30000, 0x30010}, {0x40000, 0x40000}};
    recorder.NotePendingReads(ranges, Kind::AddressBased);
    Require(recorder.PendingReadOverlaps(0x20fff, 1) && recorder.PendingReadOverlaps(0x30000, 16) && !recorder.PendingReadOverlaps(0x21000, 16) && !recorder.PendingReadOverlaps(0x40000, 16), "noted ranges are not seen as reads");
    const auto batch = recorder.DescribePendingRead(0x30008, 4);
    Require(batch.has_value() && batch->kind == Kind::AddressBased && batch->open, "a noted range has the wrong reader kind");
    Require(Recorder::ReadCounts().hits[static_cast<std::size_t>(Kind::AddressBased)] >= 2, "address-based hits are not counted");
    recorder.Sync();
    Require(recorder.Idle() && !recorder.PendingReadOverlaps(0x20000, 0x1000, false), "noted ranges outlived their batch");
}

Recorder::WriteRanges expectedWriteUnion(Recorder::WriteRanges ranges) {
    std::sort(ranges.begin(), ranges.end());
    Recorder::WriteRanges result;
    for (const auto& [begin, end] : ranges) {
        if (!result.empty() && begin <= result.back().second) result.back().second = std::max(result.back().second, end);
        else result.emplace_back(begin, end);
    }
    return result;
}

void pendingWriteSnapshotTests(const Device& device, Recorder& recorder) {
    recorder.Sync();
    Recorder::WriteRanges expected;
    const auto check = [&] {
        const auto snapshot = Recorder::PendingWriteSnapshot();
        Require(snapshot != nullptr && *snapshot == expectedWriteUnion(expected), "pending write snapshot differs from the full union");
    };
    const auto note = [&](std::uint64_t address, std::size_t bytes) {
        const auto previous = Recorder::PendingWriteSnapshot();
        const auto saved = previous == nullptr ? Recorder::WriteRanges{} : *previous;
        recorder.NotePendingWrite(address, bytes);
        if (bytes != 0) expected.emplace_back(address, address + bytes);
        check();
        Require(previous == nullptr || *previous == saved, "publishing a new write modified an older reader's snapshot");
    };
    note(0x10000, 16);
    note(0x10020, 16);
    note(0x10010, 16);
    note(0x10004, 4);
    note(0x10000, 0);
    note(0, 1);
    note(UINT64_MAX - 16, 16);
    const Recorder::WriteRanges additions{{0x20020, 0x20040}, {0x20000, 0x20020}, {0x20010, 0x20030}, {0x20040, 0x20040}, {0x20050, 0x20040}};
    recorder.NotePendingWrites(additions);
    for (const auto& range : additions) if (range.second > range.first) expected.push_back(range);
    check();
    recorder.NotePendingFill(0x30000, 16, 0xA5);
    expected.emplace_back(0x30000, 0x30010);
    check();
    alignas(64) static std::array<std::byte, 64> labels{};
    const auto labelAddress = reinterpret_cast<std::uint64_t>(labels.data());
    const std::array value{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    recorder.NoteLabel(labelAddress, value, 1, 0);
    expected.emplace_back(labelAddress, labelAddress + value.size());
    check();
    recorder.Submit();
    device.WaitQueue();
    recorder.AfterCompletions(labelAddress + 8, value, 2, 0, true);
    expected.emplace_back(labelAddress + 8, labelAddress + 8 + value.size());
    check();
    note(0x40000, 16);
    bool completing = false;
    recorder.OnComplete([&] {
        const auto snapshot = Recorder::PendingWriteSnapshot();
        Require(Recorder::SnapshotOverlaps(snapshot.get(), 0x40000, 16), "a finishing batch left the snapshot before its completion");
        completing = true;
    });
    recorder.Sync();
    Require(completing && Recorder::PendingWriteSnapshot()->empty(), "completed writes remained published");
    std::mt19937_64 random(0x41505335u);
    for (int trial = 0; trial < 16; ++trial) {
        expected.clear();
        for (int step = 0; step < 64; ++step) note(0x80000 + random() % 4096, 1 + random() % 128);
        recorder.Sync();
        Require(Recorder::PendingWriteSnapshot()->empty(), "random writes survived their GPU batch");
    }
    expected.clear();
    note(0x90000, 16);
    {
        std::atomic<bool> invalid{false};
        std::atomic<unsigned> ready{0};
        std::atomic<std::uint64_t> reads{0};
        std::array<std::jthread, 2> readers;
        for (auto& reader : readers) reader = std::jthread([&](std::stop_token stop) {
            ready.fetch_add(1, std::memory_order_release);
            while (!stop.stop_requested()) {
                const auto snapshot = Recorder::PendingWriteSnapshot();
                if (snapshot == nullptr || !Recorder::SnapshotOverlaps(snapshot.get(), 0x90000, 16)) invalid.store(true, std::memory_order_relaxed);
                if (snapshot != nullptr) {
                    for (std::size_t i = 0; i < snapshot->size(); ++i) {
                        const auto [begin, end] = (*snapshot)[i];
                        if (end <= begin || (i != 0 && (*snapshot)[i - 1].second >= begin)) invalid.store(true, std::memory_order_relaxed);
                    }
                }
                reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
        while (ready.load(std::memory_order_acquire) != readers.size()) std::this_thread::yield();
        for (std::uint64_t i = 0; i < 256; ++i) note(0x91000 + i * 128, 32);
        for (auto& reader : readers) reader.request_stop();
        for (auto& reader : readers) reader.join();
        Require(reads.load() != 0 && !invalid.load(), "concurrent snapshot readers observed an invalid or incomplete union");
    }
    recorder.Sync();
    expected.clear();
    note(UINT64_MAX - 3, 8);
    note(0x50000, 16);
    recorder.Sync();
    recorder.NotePendingWrite(0x58000, 16);
    {
        Recorder other(device.GetContext());
        other.Activate();
        recorder.NotePendingWrite(0x59000, 16);
        recorder.Activate();
        recorder.NotePendingWrite(0x5A000, 16);
        Require(*Recorder::PendingWriteSnapshot() == Recorder::WriteRanges{{0x58000, 0x58010}, {0x59000, 0x59010}, {0x5A000, 0x5A010}}, "reactivation missed writes recorded while inactive");
        other.Activate();
        recorder.Sync();
        recorder.Activate();
        recorder.NotePendingWrite(0x5B000, 16);
        Require(*Recorder::PendingWriteSnapshot() == Recorder::WriteRanges{{0x5B000, 0x5B010}}, "reactivation retained writes completed while inactive");
        recorder.Sync();
        other.Activate();
        other.NotePendingWrite(0x60000, 16);
        const auto otherSnapshot = Recorder::PendingWriteSnapshot();
        recorder.Activate();
        recorder.NotePendingWrite(0x70000, 16);
        const auto ownSnapshot = Recorder::PendingWriteSnapshot();
        Require(ownSnapshot != nullptr && *ownSnapshot == Recorder::WriteRanges{{0x70000, 0x70010}}, "a recorder extended another recorder's snapshot");
        Require(*otherSnapshot == Recorder::WriteRanges{{0x60000, 0x60010}}, "a replaced owner's retained snapshot changed");
        other.Sync();
        Require(Recorder::PendingWriteSnapshot() == ownSnapshot, "an inactive recorder replaced the active snapshot");
        recorder.Sync();
    }
    Require(Recorder::PendingWriteSnapshot()->empty(), "snapshot ownership tests left pending writes");
    std::cout << "Pending write snapshot differential, lifetime, completion and ownership tests passed\n";
}

void pendingWriteSnapshotBenchmark(Recorder& recorder) {
    for (std::uint64_t distinct : {8u, 32u, 512u}) {
        for (int duplicates : {0, 8, 64}) {
            std::array<double, 5> samples{};
            for (auto& sample : samples) {
                recorder.Sync();
                static_cast<void>(recorder.Commands());
                const auto start = std::chrono::steady_clock::now();
                for (std::uint64_t i = 0; i < distinct; ++i) {
                    const auto address = 0x100000 + i * 4096;
                    recorder.NotePendingWrite(address, 256);
                    for (int repeat = 0; repeat < duplicates; ++repeat) recorder.NotePendingWrite(address, 256);
                }
                sample = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                const auto snapshot = Recorder::PendingWriteSnapshot();
                Require(snapshot != nullptr && snapshot->size() == distinct, "snapshot benchmark produced another union");
            }
            std::sort(samples.begin(), samples.end());
            std::cout << "snapshot-note benchmark: " << distinct << " distinct, " << duplicates << " repeats, median " << samples[2] << " ms\n";
        }
    }
    recorder.Sync();
}

void readListTests(const Device& device, Recorder& recorder) {
    using Ranges = Recorder::ReadList;
    Require(Recorder::ValidReadList({{0x1000, 0x2000}, {0x2000, 0x2100}}) && Recorder::ValidReadList({}), "a sorted disjoint list is refused");
    Require(!Recorder::ValidReadList({{0x2000, 0x2100}, {0x1000, 0x2000}}) && !Recorder::ValidReadList({{0x1000, 0x2001}, {0x2000, 0x2100}}) && !Recorder::ValidReadList({{0x1000, 0x1000}}), "an unsorted, overlapping or empty list is accepted");
    std::uint64_t seed = 0x9e3779b97f4a7c15ull;
    const auto next = [&](std::uint64_t bound) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return (seed >> 33u) % bound;
    };
    const auto makeList = [&](std::uint64_t base, std::size_t count) {
        auto list = std::make_shared<Ranges>();
        auto at = base;
        for (std::size_t index = 0; index < count; ++index) {
            at += next(3) * 0x100;
            const auto bytes = (next(16) + 1) * 0x10;
            list->emplace_back(at, at + bytes);
            at += bytes;
        }
        return list;
    };
    const auto first = makeList(0x100000, 300);
    const auto second = makeList(0x100000, 120);
    const Ranges before{{0x100400, 0x100410}, {0x200000, 0x200100}};
    const Ranges between{{0x100800, 0x100900}, {0x300000, 0x300040}};
    const Ranges after{{0x100000, 0x180000}};
    const auto note = [&](bool listed) {
        recorder.NotePendingReads(before, Kind::DispatchElement);
        if (listed) recorder.NotePendingReadList(first, Kind::AddressBased);
        else recorder.NotePendingReads(*first, Kind::AddressBased);
        recorder.NotePendingReads(between, Kind::GpuCopy);
        if (listed) {
            recorder.NotePendingReadList(second, Kind::Indirect);
            recorder.NotePendingReadList(first, Kind::AddressBased);
        } else {
            recorder.NotePendingReads(*second, Kind::Indirect);
        }
        recorder.NotePendingReads(after, Kind::CopySource);
    };
    std::vector<std::pair<std::uint64_t, std::uint64_t>> queries;
    for (const auto* list : {first.get(), second.get()}) {
        for (const auto& [begin, end] : *list) {
            for (const auto at : {begin - 1, begin, end - 1, end}) queries.emplace_back(at, 1 + next(0x80));
        }
    }
    for (int query = 0; query < 2000; ++query) queries.emplace_back(0xff000 + next(0x300000), 1 + next(0x2000));
    const auto answers = [&] {
        std::vector<int> result;
        for (const auto& [address, bytes] : queries) {
            const auto described = recorder.DescribePendingRead(address, bytes);
            result.push_back(recorder.PendingReadOverlaps(address, bytes) ? 1 : 0);
            result.push_back(described.has_value() ? 10 + static_cast<int>(described->kind) : -1);
        }
        recorder.Submit();
        for (const auto& [address, bytes] : queries) result.push_back(recorder.PendingReadOverlaps(address, bytes, false) ? 1 : 0);
        recorder.Sync();
        Require(recorder.Idle() && !recorder.PendingReadOverlaps(0x100000, 0x200000, false), "listed reads outlived their batch");
        return result;
    };
    const auto notedBefore = Recorder::ReadCounts().noted;
    note(false);
    const auto plain = answers();
    const auto notedPlain = Recorder::ReadCounts().noted - notedBefore;
    note(true);
    const auto listed = answers();
    const auto notedListed = Recorder::ReadCounts().noted - notedBefore - notedPlain;
    Require(plain == listed, "reads noted by reference answer a query differently from the same ranges noted one by one");
    Require(notedListed == notedPlain + first->size(), "a list's ranges are not counted as noted");
    for (const auto kind : {Kind::DispatchElement, Kind::AddressBased, Kind::CopySource}) Require(std::count(plain.begin(), plain.end(), 10 + static_cast<int>(kind)) != 0, "the queries miss a reader kind: the comparison proves too little");
    recorder.NotePendingReadList(std::make_shared<Ranges>(), Kind::AddressBased);
    recorder.NotePendingReadList(nullptr, Kind::AddressBased);
    Require(!recorder.Recording(), "an empty list opened a batch");
    static_cast<void>(device);
}

struct TestRange {
    std::uint64_t begin;
    std::uint64_t end;
};

void overlappedRangeTests(Recorder& recorder) {
    Require(recorder.Idle(), "the recorder is busy before the overlap tests");
    recorder.NotePendingWrite(0x210000, 0x100);
    recorder.NotePendingWrite(0x210400, 0x10);
    recorder.Submit();
    recorder.NotePendingWrite(0x210800, 0x200);
    recorder.NotePendingWrite(0x210900, 0x200);
    const std::array<TestRange, 10> ranges{{{0x20f000, 0x210000}, {0x210000, 0x210080}, {0x210080, 0x210400}, {0x210400, 0x210410}, {0x210410, 0x210800}, {0x210800, 0x210801}, {0x210801, 0x210a00}, {0x210a00, 0x210b00}, {0x210b00, 0x211000}, {0x300000, 0x300010}}};
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    recorder.PendingWriteRanges(writes);
    Require(writes.size() == 4, "the pending write ranges miss a batch's writes");
    std::vector<std::size_t> indices;
    std::size_t matched = 0;
    for (const std::size_t first : {0u, 3u, 6u, 10u}) {
        OverlappedRanges(std::span<const TestRange>(ranges), first, writes, indices);
        Require(std::is_sorted(indices.begin(), indices.end()) && std::adjacent_find(indices.begin(), indices.end()) == indices.end(), "the overlapped ranges are not sorted and unique");
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            const bool listed = std::find(indices.begin(), indices.end(), index) != indices.end();
            const bool pending = index >= first && recorder.PendingWriteOverlaps(ranges[index].begin, static_cast<std::size_t>(ranges[index].end - ranges[index].begin));
            Require(listed == pending, "a range is listed as overlapped differently from PendingWriteOverlaps");
            matched += listed ? 1 : 0;
        }
    }
    Require(matched == 12, "the overlap comparison covered too few overlapped ranges");
    recorder.Sync();
    recorder.PendingWriteRanges(writes);
    Require(writes.empty(), "pending write ranges outlived their batches");
}

void spaceWriteSyncTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: address space pending writes not tested\n";
        return;
    }
    constexpr std::size_t bytes = 0x10000;
    constexpr std::size_t count = 4;
    void* block = ::operator new(2 * count * bytes, std::align_val_t{65536});
    std::memset(block, 0, 2 * count * bytes);
    auto* memory = static_cast<std::byte*>(block);
    const auto registry = [&](bool add) {
        GuestAllocations::Mutation mutation;
        for (std::size_t index = 0; index < count; ++index) {
            if (add) mutation.Add(memory + 2 * index * bytes, bytes, true, false, true);
            else mutation.Remove(memory + 2 * index * bytes);
        }
    };
    const auto rangeAt = [&](std::size_t index) { return reinterpret_cast<std::uint64_t>(memory + 2 * index * bytes); };
    const auto build = [&] {
        GuestBufferMemory leased(context);
        leased.AcquireRegistered();
        leased.AddReadable(rangeAt(0) + 64, 16);
        leased.Upload(true);
        const auto reads = leased.DeviceReads();
        leased.WriteBack();
        return reads.size();
    };
    registry(true);
    Require(recorder.Idle(), "the recorder is busy before the address space tests");
    const auto regions = build();
    if (regions < count) {
        registry(false);
        ::operator delete(block, std::align_val_t{65536});
        std::cout << "the registered ranges are not served in place: address space pending writes not tested\n";
        return;
    }
    const std::array<std::byte, 4> value{std::byte{0x5a}, std::byte{0xa5}, std::byte{0x5a}, std::byte{0xa5}};
    const auto landed = [&](std::uint64_t address) { return std::memcmp(reinterpret_cast<const void*>(address), value.data(), value.size()) == 0; };
    const auto inside = rangeAt(2) + 0x100;
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(inside, value, 7000, 0, false);
    Require(Recorder::PendingCompletionLabels() == 1 && !landed(inside), "the label inside a registered range is not pending");
    build();
    Require(landed(inside) && Recorder::PendingCompletionLabels() == 0, "a build over the address space did not complete the label pending in one of its ranges");
    recorder.Sync();
    const auto gap = rangeAt(2) + bytes + 0x100;
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(gap, value, 7001, 0, false);
    build();
    Require(!landed(gap) && Recorder::PendingCompletionLabels() == 1, "a build over the address space completed a label outside its ranges");
    recorder.Sync();
    Require(landed(gap) && Recorder::PendingCompletionLabels() == 0, "the label outside the ranges did not land");
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(rangeAt(1) + 0x40, value, 7002, 0, false);
    recorder.Submit();
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(rangeAt(3) + 0x40, value, 7003, 0, false);
    build();
    Require(landed(rangeAt(1) + 0x40) && landed(rangeAt(3) + 0x40) && Recorder::PendingCompletionLabels() == 0, "a build over the address space missed a label behind an earlier sync");
    recorder.Sync();
    static_cast<void>(recorder.Commands());
    recorder.AfterCompletions(rangeAt(1) + 0x80, value, 7004, 0, false);
    bool noted = false;
    recorder.OnComplete([&] {
        static_cast<void>(recorder.Commands());
        recorder.AfterCompletions(rangeAt(3) + 0x80, value, 7005, 0, false);
        noted = true;
    });
    build();
    Require(noted && landed(rangeAt(1) + 0x80) && landed(rangeAt(3) + 0x80) && Recorder::PendingCompletionLabels() == 0, "a build over the address space missed a label a completion noted during its sync");
    recorder.Sync();
    registry(false);
    ::operator delete(block, std::align_val_t{65536});
}

void writeSettledTests(const Device& device, Recorder& recorder) {
    Require(recorder.PendingWriteSettled(0x50000, 0x100), "an unwritten range is not settled");
    recorder.NotePendingWrite(0x50000, 0x100);
    Require(recorder.PendingWriteOverlaps(0x50000, 0x100) && !recorder.PendingWriteSettled(0x50000, 0x100) && recorder.PendingWriteSettled(0x50100, 0x100), "an open batch's write counts as settled");
    recorder.Submit();
    device.WaitQueue();
    Require(recorder.PendingWriteOverlaps(0x50000, 0x100) && recorder.PendingWriteSettled(0x50000, 0x100), "a signaled batch's write is not settled");
    recorder.NotePendingWrite(0x50080, 0x10);
    Require(!recorder.PendingWriteSettled(0x50000, 0x100) && recorder.PendingWriteSettled(0x50000, 0x80), "a later open write over the range still counts as settled");
    recorder.Sync();
    Require(recorder.Idle() && !recorder.PendingWriteOverlaps(0x50000, 0x100), "writes outlived their batches");
}

// Completion counting: a write-back completion (OnComplete) is pending until its batch finished;
// a completion label the GPU also stored (AfterCompletions storedOnGpu) is pending only once a CPU
// write-back overlapped it (NoteWrittenBack, once per label), one the GPU has no view of from its
// registration; both counts return to 0 when the batch finishes.
void completionCountTests(const Device& device, Recorder& recorder) {
    alignas(64) static std::uint32_t memory[64];
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const std::array<std::byte, 4> value{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    Require(recorder.Idle() && Recorder::PendingCompletionLabels() == 0 && Recorder::PendingWriteBackCompletions() == 0, "a fresh recorder reports pending completions");
    int ran = 0;
    recorder.OnComplete([&] { ++ran; });
    Require(Recorder::PendingWriteBackCompletions() == 1 && recorder.HasCompletions(), "a write-back completion is not pending");
    // APS5_COUNT_ALL_COMPLETION_LABELS=1 (or APS5_LABEL_STORE_ALWAYS=1) counts the GPU-stored
    // labels at registration, as before; the write-back then changes nothing.
    const bool countAll = std::getenv("APS5_COUNT_ALL_COMPLETION_LABELS") != nullptr || std::getenv("APS5_LABEL_STORE_ALWAYS") != nullptr;
    const std::uint64_t registered = countAll ? 2 : 0;
    recorder.AfterCompletions(base, value, 3, 0, true);
    recorder.AfterCompletions(base + 8, value, 4, 0, true);
    Require(Recorder::PendingCompletionLabels() == registered, "a GPU-stored completion label counts before any write-back");
    recorder.AfterCompletions(base + 0x40, value, 5, 0, false);
    Require(Recorder::PendingCompletionLabels() == registered + 1, "a label the GPU has no view of is not pending");
    // A late lookup (stamp <= afterStamp) of a label whose value reaches memory only through the
    // batch's completion action is refused as BehindCompletion (a poller reaps for it); a GPU-stored
    // one keeps the ordinary reason (unclosed group, or trust off) until a write-back overlaps it.
    using Refusal = Recorder::LabelRefusal;
    const Refusal ordinary = Recorder::LateTrust() ? Refusal::Unclosed : Refusal::TrustOff;
    Refusal refusal{};
    Require(!Recorder::LookupLabel(base + 0x40, 4, 5, &refusal).has_value() && refusal == Refusal::BehindCompletion, "a not-imported completion label is not refused as behind a completion");
    Require(!Recorder::LookupLabel(base, 4, 3, &refusal).has_value() && refusal == (countAll ? Refusal::BehindCompletion : ordinary), "a GPU-stored label is refused as behind a completion before any write-back");
    Require(Recorder::LookupLabel(base, 4, 2, &refusal).has_value() && refusal == Refusal::None, "an early lookup of a completion label is refused");
    Recorder::NoteWrittenBack(base + 0x80, 0x40);
    Require(Recorder::PendingCompletionLabels() == registered + 1, "a disjoint write-back counted a label");
    Recorder::NoteWrittenBack(base, 4);
    Recorder::NoteWrittenBack(base + 2, 4);
    const std::uint64_t counted = countAll ? 3 : 2;
    Require(Recorder::PendingCompletionLabels() == counted, "a write-back over a GPU-stored label did not count it exactly once");
    Require(!Recorder::LookupLabel(base, 4, 3, &refusal).has_value() && refusal == Refusal::BehindCompletion, "a written-back GPU-stored label is not refused as behind a completion");
    Require(!Recorder::LookupLabel(base + 8, 4, 4, &refusal).has_value() && refusal == (countAll ? Refusal::BehindCompletion : ordinary), "an untouched GPU-stored label became behind a completion");
    recorder.Submit();
    device.WaitQueue();
    Require(Recorder::PendingCompletionLabels() == counted && Recorder::PendingWriteBackCompletions() == 1, "counts dropped before the batch finished");
    memory[0] = 7;
    memory[2] = 7;
    memory[16] = 7;
    recorder.Sync();
    Require(ran == 1 && Recorder::PendingCompletionLabels() == 0 && Recorder::PendingWriteBackCompletions() == 0 && recorder.Idle(), "counts did not return to 0 at finish");
    const bool storeAlways = std::getenv("APS5_LABEL_STORE_ALWAYS") != nullptr;
    Require(memory[0] == 1 && memory[2] == (storeAlways ? 1u : 7u) && memory[16] == 1, "completion stores ran for the wrong labels (overlapped and not-imported ones store, the untouched GPU-stored one skips)");
}

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        Require(file != nullptr, "cannot open a temporary file for stderr");
#ifdef _WIN32
        saved = _dup(_fileno(stderr));
        _dup2(_fileno(file), _fileno(stderr));
#else
        saved = dup(fileno(stderr));
        dup2(fileno(file), fileno(stderr));
#endif
    }

    std::string Finish() {
        std::fflush(stderr);
#ifdef _WIN32
        _dup2(saved, _fileno(stderr));
        _close(saved);
#else
        dup2(saved, fileno(stderr));
        close(saved);
#endif
        std::rewind(file);
        std::string text;
        char buffer[4096];
        for (std::size_t read = 0; (read = std::fread(buffer, 1, sizeof(buffer), file)) != 0;) text.append(buffer, read);
        std::fclose(file);
        std::fwrite(text.data(), 1, text.size(), stderr);
        return text;
    }

private:
    std::FILE* file = nullptr;
    int saved = -1;
};

std::size_t DeferredFailureLines(Recorder& recorder, std::size_t count, const std::function<void()>& failure, std::string* log = nullptr) {
    StderrCapture capture;
    for (std::size_t index = 0; index < count; ++index) recorder.OnComplete(failure);
    recorder.Sync();
    const auto text = capture.Finish();
    if (log != nullptr) *log = text;
    std::size_t lines = 0;
    for (auto at = text.find("deferred write-back failed"); at != std::string::npos; at = text.find("deferred write-back failed", at + 1)) ++lines;
    return lines;
}

void deferredFailureReportTests(Recorder& recorder) {
    std::uint32_t serial = 0;
    const auto plain = [&serial] { throw std::runtime_error("image table: poisoned entry " + std::to_string(serial++)); };
    const auto unaligned = [](std::uint32_t instruction) { return [instruction] { throw BdaUnalignedAccess("BDA access failed: unaligned", instruction); }; };
    Require(DeferredFailureLines(recorder, 12, plain) == 12, "a deferred failure other than an unaligned BDA access was not printed every time");
    Require(DeferredFailureLines(recorder, 12, unaligned(0x5c8)) == 8, "the first 8 deferred reports of an unaligned BDA access were not the only ones printed");
    std::string log;
    Require(DeferredFailureLines(recorder, 1012, unaligned(0x5c8), &log) == 1 && log.find("pc 0x5c8 1024 times so far") != std::string::npos, "the 1024th deferred report of an unaligned BDA access was not the only one printed");
    Require(DeferredFailureLines(recorder, 12, unaligned(0x7a0)) == 8, "unaligned BDA reports at one instruction limited the printing of the reports at another");
    Require(DeferredFailureLines(recorder, 12, plain) == 12, "unaligned BDA reports limited the printing of other deferred failures");
}

void directCompletionLabelTests(Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t size = 65536;
    struct Release {
        Recorder& recorder;
        void* mapping = nullptr;
        std::int64_t physical = -1;
        ~Release() {
            recorder.Sync();
            if (mapping != nullptr) sceKernelMunmap(mapping, size);
            if (physical >= 0) sceKernelReleaseDirectMemory(physical, size);
        }
    } release{recorder};
    Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, size, size, 0, &release.physical) == 0, "allocate completion label backing");
    Require(sceKernelMapDirectMemory(&release.mapping, size, 3, 0, release.physical, size) == 0, "map completion label backing");
    auto* memory = static_cast<std::byte*>(release.mapping);
    std::memset(memory, 7, size);
    const auto address = reinterpret_cast<std::uint64_t>(memory) + 4092;
    std::array<std::byte, 16> value;
    for (const auto bytes : {4u, 8u, 16u}) {
        value.fill(static_cast<std::byte>(bytes));
        const auto before = CollectWritesUncached(address, bytes);
        recorder.NotePendingWrite(address, bytes);
        recorder.AfterCompletions(address, std::span(value).first(bytes), 100 + bytes, 0, false);
        value.fill(std::byte{99});
        bool observed = false;
        Require(recorder.AfterRecordedWork([&] {
            observed = std::all_of(memory + 4092, memory + 4092 + bytes, [bytes](auto b) { return b == static_cast<std::byte>(bytes); });
        }), "completion observer was not recorded");
        recorder.Sync();
        Require(observed, "a following completion observed incorrect label bytes");
        Require(memory[4091] == std::byte{7} && memory[4108] == std::byte{7}, "completion label changed neighboring bytes");
        Require(recorder.Idle() && Recorder::PendingCompletionLabels() == 0, "completion label remained pending after sync");
        if (before != 0) {
            Require(StoredOver(address, bytes, before), "completion label was not stamped");
            Require(UnchangedSinceCollected(address, bytes, before), "completion label dirtied the guest page");
            const auto after = CollectWritesUncached(address, bytes);
            memory[4092] = std::byte{81};
            Require(!UnchangedSinceCollected(address, bytes, after), "completion store disabled subsequent guest tracking");
        }
    }
}

void afterRecordedWorkTests(const Device& device, Recorder& recorder) {
    alignas(64) static std::uint32_t memory[16];
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const std::array<std::byte, 4> value{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    recorder.Sync();
    std::vector<int> ran;
    std::uint32_t seen = 0;
    Require(recorder.Idle() && !recorder.AfterRecordedWork([&] { ran.push_back(0); }) && ran.empty() && Recorder::PendingCompletionLabels() == 0, "an idle recorder kept an action for recorded work");
    recorder.NotePendingWrite(0x52000, 0x100);
    Require(!Recorder::PendingLabelSince().has_value(), "a write started the label flush deadline");
    Require(recorder.AfterRecordedWork([&] { ran.push_back(1); }) && Recorder::PendingCompletionLabels() == 1 && Recorder::PendingLabelSince().has_value(), "an action behind the open batch is not pending under the label flush deadline");
    recorder.AfterCompletions(base, value, 6, 0, false);
    Require(recorder.AfterRecordedWork([&] { seen = memory[0]; ran.push_back(2); }) && Recorder::PendingCompletionLabels() == 3, "an action behind a completion label is not pending");
    recorder.Submit();
    Require(recorder.AfterRecordedWork([&] { ran.push_back(3); }) && Recorder::PendingCompletionLabels() == 4, "an action behind an in-flight batch is not pending");
    device.WaitQueue();
    Require(ran.empty(), "an action ran before its batch was reaped");
    recorder.Sync();
    Require(ran == std::vector<int>{1, 2, 3} && Recorder::PendingCompletionLabels() == 0 && recorder.Idle(), "actions behind recorded work did not run once each, in order");
    Require(seen == 1, "an action ran before the completion label recorded ahead of it");
}

int gpuTimingCompletionTests(const Device& device, bool incomplete) {
    const auto& context = device.GetContext();
    if (!device.Timestamps()) {
        std::cout << "skipped, queue timestamp queries unavailable\n";
        return 77;
    }
    Require(Recorder::GpuTimingEnabled(), "GPU timing completion test requires APS5_PROFILE_GPU");
    std::unique_lock gpu(GpuMutex());
    int completed = 0;
    Recorder recorder(context);
    recorder.Activate();
    for (int round = 0; round < 3; ++round) {
        const bool missingEnd = incomplete && round == 1;
        const auto first = recorder.BeginGpuTiming(0x12340);
        Require(first != Recorder::NoTiming, "first GPU timing range was not allocated");
        if (!missingEnd) recorder.EndGpuTiming(first);
        const auto second = recorder.BeginGpuTiming(0x12341);
        Require(second != Recorder::NoTiming && second != first, "second GPU timing range was not allocated");
        recorder.EndGpuTiming(second);
        recorder.OnComplete([&] { ++completed; });
        const auto serial = recorder.Submissions() + 1;
        recorder.Submit();
        recorder.Sync();
        Require(completed == round + 1 && recorder.Idle(), "GPU timing prevented batch completion");
        std::size_t missing = 0;
        const auto batches = recorder.CompletedBatches(serial - 1, serial, missing);
        Require(missing == 0 && batches.size() == 1 && batches.front().serial == serial, "GPU timing batch completion record is missing");
        const auto& batch = batches.front();
        if (missingEnd) {
            Require(batch.gpuStartNs == 0 && batch.gpuEndNs == 0, "incomplete query batch published GPU timings");
        } else {
            Require(batch.gpuStartNs > 0 && batch.gpuEndNs >= batch.gpuStartNs, "complete query batch lost GPU timings");
        }
    }
    std::cout << "GPU timing batch completion checked, incomplete=" << incomplete << '\n';
    return 0;
}

void batchStampTests(Recorder& recorder) {
    Require(recorder.Idle(), "batch stamps: the recorder is busy");
    double previousEnd = 0;
    for (int round = 0; round < 3; ++round) {
        recorder.NotePendingWrite(0x60000, 0x100);
        const auto serial = recorder.Submissions() + 1;
        recorder.Submit();
        recorder.Sync();
        std::size_t missing = 0;
        const auto batches = recorder.CompletedBatches(serial - 1, serial, missing);
        Require(missing == 0 && batches.size() == 1 && batches.front().serial == serial, "a finished batch has no completion record");
        const auto& batch = batches.front();
        if (Recorder::BatchStampsEnabled()) {
            Require(batch.gpuStartNs > 0 && batch.gpuEndNs >= batch.gpuStartNs && batch.gpuStartNs >= previousEnd, "batch stamps are missing or out of order");
            previousEnd = batch.gpuEndNs;
        } else {
            Require(batch.gpuStartNs == 0 && batch.gpuEndNs == 0, "batch stamps were written with profiling off");
        }
    }
    std::cout << "batch stamps " << (Recorder::BatchStampsEnabled() ? "checked" : "off") << '\n';
}

void labelTests(Recorder& recorder) {
    const std::array<std::byte, 4> value{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    Require(!recorder.PendingLabelIn(0x60000, 0x100), "an empty table reports a label");
    // A label this worker queued but has not recorded yet is inside the range for a CPU store decision.
    Recorder::NoteQueuedLabel(0x60010, value, 1, AgcDriver::GuestMemory::GpuLockThreadTag());
    Require(recorder.PendingLabelIn(0x60010, 4) && recorder.PendingLabelIn(0x60000, 0x100) && !recorder.PendingLabelIn(0x60014, 4) && !recorder.PendingLabelIn(0x60000, 0x10), "a queued label is not seen by PendingLabelIn");
    Recorder::ForgetQueuedLabels();
    Require(!recorder.PendingLabelIn(0x60000, 0x100), "a forgotten queued label is still pending");
    recorder.NoteLabel(0x60020, value, 2, 0);
    Require(recorder.PendingLabelIn(0x60020, 4) && recorder.PendingLabelIn(0x60000, 0x100) && !recorder.PendingLabelIn(0x60000, 0x20), "a recorded label is not seen by PendingLabelIn");
    recorder.Sync();
    Require(!recorder.PendingLabelIn(0x60000, 0x100), "a recorded label outlived its batch");
}

// The late rule (Recorder::NoteLabel): an entry with stamp <= afterStamp is served once its group
// closed, unless a non-label write covered it, it is queued, or APS5_LABEL_TRUST_LATE=0 (run the
// binary with that set for the switch-off case).
void lateLabelTests(Recorder& recorder) {
    using Refusal = Recorder::LabelRefusal;
    const std::array<std::byte, 4> one{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    const std::array<std::byte, 4> two{std::byte{2}, std::byte{0}, std::byte{0}, std::byte{0}};
    const std::array<std::byte, 8> pair{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}, std::byte{2}, std::byte{0}, std::byte{0}, std::byte{0}};
    const auto tag = AgcDriver::GuestMemory::GpuLockThreadTag();
    constexpr std::uint64_t a = 0x70000, b = 0x70100, c = 0x70200, d = 0x70300;
    Refusal refusal{};
    const auto before = Recorder::LateCounts();
    recorder.NoteLabel(a, one, 10, 0);
    const auto early = Recorder::LookupLabel(a, 4, 5, &refusal);
    Require(early.has_value() && !early->late && early->value == 1 && early->queue == 0 && early->stamp == 10 && early->generation == 0 && refusal == Refusal::None, "an early lookup is wrong");
    auto late = Recorder::LookupLabel(a, 4, 10, &refusal);
    if (!Recorder::LateTrust()) {
        Require(!late.has_value() && refusal == Refusal::TrustOff, "(1) a late lookup is not refused with the switch off");
        Require(Recorder::LateCounts().candidates == before.candidates + 1, "(1) the late candidate is not counted with the switch off");
        Recorder::CloseLabelGroup(AgcDriver::GuestMemory::TrackerGeneration());
        Require(!Recorder::LookupLabel(a, 4, 10, &refusal).has_value() && refusal == Refusal::TrustOff, "(1) a closed group is served with the switch off");
        Require(Recorder::LookupLabel(a, 4, 5).has_value(), "(1) the early lookup broke with the switch off");
        recorder.Sync();
        Require(!Recorder::LookupLabel(a, 4, 5).has_value(), "(7) the entry outlived its batch");
        std::cout << "late label tests ran with APS5_LABEL_TRUST_LATE=0\n";
        return;
    }
    Require(!late.has_value() && refusal == Refusal::Unclosed, "(2) an unclosed group is served");
    Require(Recorder::LateCounts().candidates == before.candidates + 1 && Recorder::LateCounts().unclosed == before.unclosed + 1, "(2) the unclosed refusal is not counted");
    const auto generation = AgcDriver::GuestMemory::TrackerGeneration();
    Recorder::CloseLabelGroup(generation);
    late = Recorder::LookupLabel(a, 4, 10, &refusal);
    Require(late.has_value() && late->late && late->value == 1 && late->generation == generation && late->stamp == 10 && refusal == Refusal::None, "(2) a closed group is not served");
    Require(Recorder::LookupLabel(a, 4, 9)->late == false, "(2) an early lookup became late");
    // (3) A disjoint write leaves the entry alone; an overlapping non-label write refuses the late
    // lookup only (the early one still composes what memory will hold in order).
    recorder.NotePendingWrite(a + 4, 16);
    Require(Recorder::LookupLabel(a, 4, 10).has_value(), "(3) a disjoint write flagged the entry");
    recorder.NotePendingWrite(a - 8, 12);
    late = Recorder::LookupLabel(a, 4, 10, &refusal);
    Require(!late.has_value() && refusal == Refusal::Overwritten, "(3) an overwritten entry is served late");
    Require(Recorder::LookupLabel(a, 4, 5).has_value(), "(3) an overwrite refused the early lookup");
    // The label's own note (inside NoteLabel) is no overwrite: an 8-byte label closed as a group.
    recorder.NoteLabel(b, pair, 20, 0);
    Recorder::CloseLabelGroup(AgcDriver::GuestMemory::TrackerGeneration());
    const auto wide = Recorder::LookupLabel(b, 8, 20, &refusal);
    Require(wide.has_value() && wide->late && wide->value == 0x200000001ull && refusal == Refusal::None, "(3) the label's own note refused it");
    // (4) A later label on the dword replaces the entry: unclosed until its own group closes.
    recorder.NoteLabel(a, two, 30, 0);
    Require(!Recorder::LookupLabel(a, 4, 40, &refusal).has_value() && refusal == Refusal::Unclosed, "(4) a replaced entry kept the old close or flag");
    const auto replaced = Recorder::LookupLabel(a, 4, 25);
    Require(replaced.has_value() && !replaced->late && replaced->value == 2 && replaced->stamp == 30, "(4) the replaced entry's early lookup is wrong");
    const auto generation2 = AgcDriver::GuestMemory::TrackerGeneration();
    Recorder::CloseLabelGroup(generation2);
    late = Recorder::LookupLabel(a, 4, 40, &refusal);
    Require(late.has_value() && late->late && late->value == 2 && late->generation == generation2, "(4) the replaced entry is not late-trustable after its close");
    // (5) Queued entries: another queue's is never seen, this queue's is early-only.
    const std::uint32_t other = tag == 7 ? 8 : 7;
    Recorder::NoteQueuedLabel(d, one, 50, other);
    Require(!Recorder::LookupLabel(d, 4, 60, &refusal).has_value() && refusal == Refusal::None && !Recorder::LookupLabel(d, 4, 40).has_value(), "(5) another queue's queued label is served");
    Recorder::NoteQueuedLabel(d, one, 51, tag);
    Recorder::ForgetQueuedLabels();
    Recorder::NoteQueuedLabel(c, one, 50, tag);
    Require(Recorder::LookupLabel(c, 4, 40).has_value() && !Recorder::LookupLabel(c, 4, 40)->late, "(5) this queue's queued label is not served early");
    Require(!Recorder::LookupLabel(c, 4, 60, &refusal).has_value() && refusal == Refusal::Queued, "(5) a queued entry is late-trusted");
    Recorder::CloseLabelGroup(AgcDriver::GuestMemory::TrackerGeneration());
    Require(!Recorder::LookupLabel(c, 4, 60, &refusal).has_value() && refusal == Refusal::Queued, "(5) a close closed a queued entry");
    Recorder::ForgetQueuedLabels();
    Require(!Recorder::LookupLabel(c, 4, 40).has_value(), "(5) a forgotten queued entry is still served");
    const auto after = Recorder::LateCounts();
    Require(after.queued == before.queued + 2 && after.overwritten == before.overwritten + 1 && after.unclosed == before.unclosed + 2 && after.candidates >= before.candidates + 6, "late counters are off");
    // (7) Entries leave the table with their batch.
    recorder.Sync();
    Require(!Recorder::LookupLabel(a, 4, 5).has_value() && !Recorder::LookupLabel(b, 8, 5).has_value() && !recorder.PendingLabelIn(0x70000, 0x400), "(7) entries outlived their batch");
}

void largeLabelTests(Recorder& recorder) {
    constexpr std::uint64_t slot = 0x71000, small = slot + 0x10, outside = slot - 4, wide = 0x72000;
    const std::array<std::byte, 4> one{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    std::array<std::byte, 256> clear{};
    std::array<std::byte, Recorder::LabelTableBytes> limit{};
    limit.fill(std::byte{3});
    recorder.Sync();
    Require(recorder.PendingLabels() == 0 && !Recorder::PendingLabelSince().has_value() && !Recorder::WideLabelIn(0, ~0ull), "the table is not empty before the large label tests");
    recorder.NoteLabel(small, one, 100, 0);
    recorder.NoteLabel(outside, one, 100, 0);
    Recorder::CloseLabelGroup(AgcDriver::GuestMemory::TrackerGeneration());
    Require(recorder.PendingLabels() == 2 && Recorder::LookupLabel(small, 4, 50).has_value(), "4-byte labels did not enter the table");
    recorder.NoteLabel(slot, clear, 101, 0);
    Recorder::CloseLabelGroup(AgcDriver::GuestMemory::TrackerGeneration());
    Require(recorder.PendingLabels() == 1 && Recorder::LookupLabel(outside, 4, 50).has_value(), "a label larger than the table bound entered dwords or removed one outside its range");
    Require(!Recorder::LookupLabel(small, 4, 50).has_value() && !Recorder::LookupLabel(slot, 8, 0).has_value(), "an older label under a larger one still composes the value");
    Require(Recorder::WideLabelIn(slot, 4) && Recorder::WideLabelIn(slot + clear.size() - 4, 4) && Recorder::WideLabelIn(slot - 0x100, 0x104) && !Recorder::WideLabelIn(slot - 0x100, 0x100) && !Recorder::WideLabelIn(slot + clear.size(), 4), "a large label's range is wrong");
    Require(recorder.PendingLabelIn(slot + 0x20, 0x20) && recorder.PendingWriteOverlaps(slot, clear.size()), "a large label is not pending");
    Require(Recorder::PendingLabelSince().has_value(), "a large label did not start the label flush deadline");
    recorder.NoteLabel(small, one, 102, 0);
    Recorder::CloseLabelGroup(AgcDriver::GuestMemory::TrackerGeneration());
    const auto later = Recorder::LookupLabel(small, 4, 101);
    Require(later.has_value() && later->value == 1 && later->stamp == 102, "a label after a large one is not served");
    recorder.NoteLabel(wide, limit, 103, 0);
    Recorder::CloseLabelGroup(AgcDriver::GuestMemory::TrackerGeneration());
    Require(recorder.PendingLabels() == 2 + Recorder::LabelTableBytes / 4 && !Recorder::WideLabelIn(wide, limit.size()), "a label at the table bound did not enter every dword");
    const auto bound = Recorder::LookupLabel(wide + Recorder::LabelTableBytes - 8, 8, 0);
    Require(bound.has_value() && bound->value == 0x0303030303030303ull && bound->stamp == 103, "a label at the table bound is not served");
    recorder.Sync();
    Require(recorder.PendingLabels() == 0 && !Recorder::WideLabelIn(0, ~0ull) && !recorder.PendingLabelIn(slot, clear.size()) && !recorder.PendingWriteOverlaps(slot, clear.size()), "large label tests left entries, ranges or writes behind");
}

// (6) GuestMemory::UnchangedSinceCollected: false outside the watched arena, and around a
// MarkWritten or a CPU write of the block inside it.
void unchangedSinceTests() {
    using namespace AgcDriver::GuestMemory;
    static std::uint32_t outside[16];
    Require(!UnchangedSinceCollected(reinterpret_cast<std::uint64_t>(outside), 4, TrackerGeneration()), "(6) a range outside the watched memory is unchanged");
    Require(!Watched(reinterpret_cast<std::uint64_t>(outside), 4) && CollectWrites(reinterpret_cast<std::uint64_t>(outside), 4) == 0, "(6) a range outside the watched memory is watched");
    constexpr std::size_t bytes = 65536;
    void* block = AllocateWatched(bytes, bytes);
    if (block == nullptr) {
        std::cout << "no write watching: UnchangedSinceCollected is always false\n";
        return;
    }
    auto* words = static_cast<volatile std::uint32_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    words[0] = 0;
    CollectWritesUncached(address, 4);
    const auto generation = TrackerGeneration();
    Require(UnchangedSinceCollected(address, 4, generation), "(6) an untouched range is not unchanged");
    Require(UnchangedSinceCollected(address, 4, generation), "(6) the check itself dirtied the range");
    Require(!UnchangedSinceCollected(address, 4, 0), "(6) generation 0 is unchanged");
    MarkWritten(address, 4);
    Require(UnchangedSinceCollected(address, 4, generation), "(6) a MarkWritten of the block (a GPU label record) refuses");
    Require(!UnchangedSince(address, 4, generation), "(6) a MarkWritten of the block is not seen by UnchangedSince");
    const auto generation2 = TrackerGeneration();
    Require(UnchangedSinceCollected(address, 4, generation2), "(6) unchanged after the stamp fails");
    words[1] = 1;
    Require(!UnchangedSinceCollected(address, 4, generation2), "(6) a CPU write to the page is not seen");
    Require(UnchangedSinceCollected(address, 4, TrackerGeneration()), "(6) the collect did not reset the page");
    const auto generation3 = TrackerGeneration();
    words[1] = 2;
    CollectWritesUncached(address, 4);
    Require(!UnchangedSinceCollected(address, 4, generation3), "(6) a CPU write collected by another caller is not seen");
    ReleaseWatched(block, bytes);
}

// (9) ProvedClearKeys: a surface's DCC keys are scanned once and answered from the proof after,
// until the key range is stamped (a GPU key store's MarkWritten) or the CPU writes it; a scan made
// while recorded work still writes the keys is not kept. With APS5_NO_KEY_FAST_PATH=1 every call
// scans and no proof is stored.
void keyProofTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t bytes = 65536;
    void* block = AllocateWatched(bytes, bytes);
    if (block == nullptr) {
        std::cout << "no write watching: key proofs not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    constexpr std::size_t keyCount = 1024;
    constexpr std::uint64_t surfaceBytes = keyCount * 256;
    auto* keys = static_cast<std::uint8_t*>(block);
    std::memset(keys, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    GuestTextureResource resource{};
    resource.baseAddress = address + 8192;
    resource.width = 256;
    resource.height = 256;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dccAddress = address;
    DccKeyProof proof;
    auto before = KeyProofCounts();
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "(9) 0x00 keys are not a 0000 clear");
    auto after = KeyProofCounts();
    Require(after.scanned == before.scanned + 1 && after.proved == before.proved, "(9) the first call did not scan");
    if (!KeyFastPath()) {
        Require(proof.generation == 0, "(9) APS5_NO_KEY_FAST_PATH stored a proof");
        Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000 && proof.generation == 0, "(9) APS5_NO_KEY_FAST_PATH proved keys");
        std::cout << "key fast path off: every call scans\n";
        return;
    }
    Require(proof.generation != 0 && proof.keys == DccKeys::Clear0000 && after.unstable == before.unstable, "(9) a stable scan left no proof");
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "(9) the proof answers other keys");
    before = KeyProofCounts();
    Require(before.proved == after.proved + 1 && before.scanned == after.scanned, "(9) the second call scanned");
    const auto generation = proof.generation;
    MarkWritten(address, keyCount);
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "(9) unchanged bytes read differently");
    after = KeyProofCounts();
    Require(after.scanned == before.scanned + 1 && after.proved == before.proved && proof.generation > generation, "(9) a stamped key range was proved");
    std::memset(keys, 0x40, keyCount);
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001, "(9) a CPU write of the keys was not seen");
    before = KeyProofCounts();
    Require(before.scanned == after.scanned + 1 && proof.keys == DccKeys::Clear0001, "(9) the CPU write did not make a scan");
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && KeyProofCounts().proved == before.proved + 1, "(9) the new keys were not proved");
    // A recorded write over the keys (noted and stamped as the driver's key stores are): the scans
    // are not kept until its batch signaled.
    recorder.NotePendingWrite(address, keyCount);
    MarkWritten(address, keyCount);
    before = KeyProofCounts();
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && proof.generation == 0, "(9) a scan under a pending write was kept");
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && proof.generation == 0, "(9) a second scan under a pending write was kept");
    after = KeyProofCounts();
    Require(after.scanned == before.scanned + 2 && after.unstable == before.unstable + 2 && after.proved == before.proved, "(9) unstable scans were miscounted");
    recorder.Submit();
    device.WaitQueue();
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && proof.generation != 0, "(9) a signaled write kept the scan unstable");
    recorder.Sync();
    before = KeyProofCounts();
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0001 && KeyProofCounts().proved == before.proved + 1, "(9) the proof did not hold after the batch finished");
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_KEY_PIN, GPU-OWNED-PLAN 9.3, K1): case 9 of keyPinProofTests, a
// live unit shadow over pinned keys (the shadowed-key-store fixture: the keys in the first 64 KiB
// unit of an imported, watched block). Today's key read publishes the shadow, so the pin refuses
// (Shadowed) and the read runs as today; at 2 nothing is verified, so nothing aborts. False when
// not tested.
bool keyPinShadowTests(const Device& device, Recorder& recorder, int mode) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    if (!UnitShadowEnabled() || context.hostImportAlignment == 0) {
        std::cout << "unit shadows or host imports unavailable: a pinned key proof under a unit shadow not tested\n";
        return false;
    }
    constexpr std::uint64_t unit = 65536;
    constexpr std::size_t bytes = 2 * unit;
    constexpr std::size_t surfaceBytes = 65536;
    constexpr std::size_t keyCount = surfaceBytes / 256;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, unit);
    Require(block != nullptr, "cannot reserve the pinned shadowed key block");
    const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
    Require(section != nullptr, "cannot create the pinned shadowed key block");
    GuestArena::GuestArenaMap_nid_postfix(block, bytes, section, 0, PAGE_READWRITE);
    CloseHandle(section);
#else
    void* block = AllocateWatched(bytes, unit);
    Require(block != nullptr, "cannot allocate the pinned shadowed key block");
#endif
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0xff, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr || !Watched(address, bytes)) {
        std::cout << "the key block is not imported and watched: a pinned key proof under a unit shadow not tested\n";
        return false;
    }
    recorder.Sync();
    CollectWritesUncached(address, bytes);
    GuestTextureResource resource{};
    resource.baseAddress = address + unit;
    resource.format = 56;
    resource.dccAddress = address;
    DccKeyProof proof;
    recorder.NotePendingFill(address, keyCount, 0xff);
    MarkWritten(address, keyCount);
    NoteKeysFillOnGpu(address, keyCount, DccKeys::Uncompressed);
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Uncompressed, "K1 (9): the keys of a pending driver store over imported keys did not read as uncompressed");
    if (mode != 0) Require(proof.pinNote != 0, "K1 (9): the pending driver store over imported keys was not pinned");
    auto destination = ShadowDestinationFor(context, *import, address, address + unit);
    Require(destination.has_value() && destination->seedUnits.empty(), "K1 (9): the key unit has no unit shadow destination");
    auto pattern = std::make_shared<Buffer>(context, static_cast<std::size_t>(unit), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memset(pattern->Bytes().data(), 0x40, pattern->Bytes().size());
    const auto commands = recorder.Commands();
    recorder.Keep(pattern);
    recorder.Keep(destination->slab);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    const VkBufferCopy piece{0, destination->offset, unit};
    context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, pattern->Handle(), destination->buffer, 1, &piece);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    recorder.MarkCovered(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    const ShadowedRange range{address, address + unit, destination->slab};
    MarkShadowed(*import, std::span(&range, 1), TrackerGeneration());
    Require(AnyShadowedOverlaps(address, keyCount), "K1 (9): the key unit is not shadowed");
    const auto before = KeyPinTotals();
    static_cast<void>(ProvedClearKeys(resource, surfaceBytes, proof));
    if (mode != 0) Require(KeyPinTotals()[KeyPin::Shadowed] == before[KeyPin::Shadowed] + 1, "K1 (9): a pin answered under a live unit shadow over the keys");
    Require(!AnyShadowedOverlaps(address, keyCount), "K1 (9): today's key read did not publish the unit shadow over the keys");
    recorder.Sync();
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_KEY_PIN, GPU-OWNED-PLAN 9.3, K1): pinned key proofs over a 64 KiB
// watched block of keys (0x00, a 0000 clear). The driver's key store is played as
// StoreUncompressedOnGpu makes it (a 0xff fill noted in the open batch, the keys stamped, the memo
// entry with the fill's note) without a command, so the bytes keep what the test wrote: today's key
// read answers Uncompressed from the memo until the batch is reaped and never keeps that scan. The
// mode follows the switch: unset or 0 takes no pin, 1 answers from pins, 2 takes and keeps them but
// runs today's scan wherever a pin held and checks it. Under the GPU mutex with `recorder` active.
// False when nothing could be tested.
bool keyPinProofTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    if (!KeyFastPath()) {
        std::cout << "key fast path off: pinned key proofs not tested\n";
        return false;
    }
    constexpr std::size_t bytes = 65536;
    void* block = AllocateWatched(bytes, bytes);
    if (block == nullptr) {
        std::cout << "no write watching: pinned key proofs not tested\n";
        return false;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    constexpr std::size_t keyCount = 1024;
    constexpr std::uint64_t surfaceBytes = keyCount * 256;
    auto* keys = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    GuestTextureResource resource{};
    resource.baseAddress = address + 8192;
    resource.width = 256;
    resource.height = 256;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dccAddress = address;
    const char* text = std::getenv("APS5_EXPERIMENT_KEY_PIN");
    const auto value = text != nullptr ? std::strtol(text, nullptr, 10) : 0;
    const int mode = value == 0 ? 0 : value == 2 ? 2 : 1;
    const bool pinning = mode != 0;
    const auto start = KeyPinTotals();
    const auto delta = [](const KeyPinCounts& before, KeyPin kind) { return KeyPinTotals()[kind] - before[kind]; };
    DccKeyProof proof;
    const auto read = [&] { return ProvedClearKeys(resource, surfaceBytes, proof); };
    const auto store = [&] {
        recorder.NotePendingFill(address, keyCount, 0xff);
        MarkWritten(address, keyCount);
        NoteKeysFillOnGpu(address, keyCount, DccKeys::Uncompressed);
    };
    recorder.Sync();
    std::memset(keys, 0x00, keyCount);

    // 1. Bytes nothing writes: a stable proof, no pin.
    Require(read() == DccKeys::Clear0000 && proof.generation != 0 && proof.pinNote == 0, "K1 (1): 0x00 keys did not give a stable 0000 proof");

    // 2. The driver's store: every mode answers Uncompressed from the memo; 1 and 2 pin it.
    store();
    auto before = KeyPinTotals();
    Require(read() == DccKeys::Uncompressed, "K1 (2): the keys of a pending driver store did not read as uncompressed");
    if (pinning) Require(proof.pinNote != 0 && proof.generation != 0 && proof.keys == DccKeys::Uncompressed && delta(before, KeyPin::Taken) == 1, "K1 (2): the pending driver store was not pinned");
    else Require(proof.generation == 0 && proof.pinNote == 0, "K1 (2): a key proof was kept under a pending driver store with the switch off");
    // The pin a take would hold, at every mode ([dcc]'s pinnable count).
    Require(DccKeyPinNote(address, keyCount) != 0 && (!pinning || DccKeyPinNote(address, keyCount) == proof.pinNote), "K1 (2): the pending driver store is not pinnable");

    // 3. The next read: 1 answers from the pin, 2 scans and verifies it, 0 scans.
    auto proofs = KeyProofCounts();
    before = KeyPinTotals();
    Require(read() == DccKeys::Uncompressed, "K1 (3): the pending store's keys did not read as uncompressed again");
    const auto scans = KeyProofCounts();
    if (mode == 1) Require(scans.proved == proofs.proved + 1 && scans.scanned == proofs.scanned && delta(before, KeyPin::KeyAnswer) == 1, "K1 (3) 1: the pin did not answer");
    else Require(scans.scanned == proofs.scanned + 1 && scans.proved == proofs.proved, "K1 (3): the second read did not scan");
    if (mode == 2) Require(delta(before, KeyPin::KeysVerified) == 1 && delta(before, KeyPin::KeyAnswer) == 0, "K1 (3) 2: the pin's prediction was not verified against today's scan");

    // 4. A CPU store into the keys under the pin: today still answers from the memo, and the pin is
    // taken again at the new collect.
    const auto generation = proof.generation;
    const auto note = proof.pinNote;
    std::memset(keys, 0x40, keyCount);
    before = KeyPinTotals();
    Require(read() == DccKeys::Uncompressed, "K1 (4): a CPU store into the keys changed the pending store's answer");
    if (pinning) Require(delta(before, KeyPin::Taken) == 1 && proof.pinNote == note && proof.generation > generation, "K1 (4): the pin was not taken again after a CPU store into the keys");

    // 5. A newer 0000 fill nothing stamped: the pinned store is not the newest writer any more.
    recorder.NotePendingFill(address, keyCount, 0x00);
    before = KeyPinTotals();
    Require(read() == DccKeys::Clear0000, "K1 (5): the keys under a newer 0000 fill did not read as 0000");
    if (pinning) Require(delta(before, KeyPin::NotNewest) == 1, "K1 (5): the pin was not refused under a newer writer");
    Require(proof.pinNote == 0 && proof.generation == 0 && DccKeyPinNote(address, keyCount) == 0, "K1 (5): a key proof was kept, or is pinnable, under a newer writer");

    // 6. The batch runs and is reaped: a stable proof from the bytes (0x40).
    recorder.Submit();
    device.WaitQueue();
    recorder.Sync();
    Require(read() == DccKeys::Clear0001 && proof.generation != 0 && proof.pinNote == 0, "K1 (6): the reaped keys gave no stable proof from their bytes");

    // 7. A pin whose batch ran but was not reaped still answers at 1; the reap ends it.
    store();
    Require(read() == DccKeys::Uncompressed && (!pinning || proof.pinNote != 0), "K1 (7): the pending driver store was not read or pinned");
    recorder.Submit();
    device.WaitQueue();
    before = KeyPinTotals();
    Require(read() == DccKeys::Uncompressed, "K1 (7): the keys of a store whose batch ran but was not reaped did not read as uncompressed");
    if (mode == 1) Require(delta(before, KeyPin::KeyAnswer) == 1, "K1 (7) 1: the pin did not answer before the reap");
    if (mode == 2) Require(delta(before, KeyPin::KeysVerified) == 1, "K1 (7) 2: the pin was not verified before the reap");
    recorder.Sync();
    before = KeyPinTotals();
    Require(read() == DccKeys::Clear0001 && proof.generation != 0 && proof.pinNote == 0, "K1 (7): the keys after the reap gave no stable proof from their bytes");
    if (pinning) Require(delta(before, KeyPin::NotPending) == 1, "K1 (7): the reap did not end the pin");

    // 8. A pin over a signaled batch after a key read (MemoizedKeys drops a signaled entry): when the
    // read dropped it, today's read answers from the pending fill's value instead, which no pin follows
    // (memo moved); when the entry is still there (the batch's fence not seen as signaled yet), today
    // still answers from the memo and the pin must agree with it (an answer at 1, verified at 2).
    store();
    Require(read() == DccKeys::Uncompressed && (!pinning || proof.pinNote != 0), "K1 (8): the pending driver store was not read or pinned");
    const auto pinned = proof.pinNote;
    recorder.Submit();
    device.WaitQueue();
    static_cast<void>(ReadDccKeys(address, surfaceBytes));
    const bool dropped = DccKeyPinNote(address, keyCount) == 0;
    before = KeyPinTotals();
    Require(read() == DccKeys::Uncompressed, "K1 (8): the keys of a signaled store after a key read did not read as uncompressed");
    if (pinning && dropped) Require(delta(before, KeyPin::MemoMoved) == 1 && proof.pinNote == 0, "K1 (8): a pin answered after the memo dropped its entry");
    if (mode == 1 && !dropped) Require(delta(before, KeyPin::KeyAnswer) == 1 && proof.pinNote == pinned, "K1 (8) 1: the pin did not answer while the memo still held its entry");
    if (mode == 2 && !dropped) Require(delta(before, KeyPin::KeysVerified) == 1, "K1 (8) 2: the pin was not verified while the memo still held its entry");
    std::cout << "K1 (8): the key read " << (dropped ? "dropped" : "kept") << " the signaled store's memo entry" << '\n';
    recorder.Sync();

    // 10. Another recorder active: the pin refuses without MemoRecorder's clear (the entries are
    // still there for the first recorder), and today's read under the other recorder clears them.
    store();
    Require(read() == DccKeys::Uncompressed && (!pinning || proof.pinNote != 0), "K1 (10): the pending driver store was not read or pinned");
    {
        const auto pinned = proof.pinNote;
        Recorder other(device.GetContext());
        other.Activate();
        if (pinning) Require(DccKeyPinRefusal(address, keyCount, pinned) == KeyPin::MemoMoved, "K1 (10): a pin held under another recorder");
        recorder.Activate();
        if (pinning) Require(DccKeyPinRefusal(address, keyCount, pinned) == KeyPin::Holds, "K1 (10): the pin check under another recorder dropped the memo's entries");
        other.Activate();
        before = KeyPinTotals();
        static_cast<void>(read());
        if (pinning) Require(delta(before, KeyPin::MemoMoved) == 1 && proof.pinNote == 0, "K1 (10): a pin answered under another recorder");
        recorder.Activate();
    }
    recorder.Sync();

    // 11. A pending 0000 fill with a note (NoteKeysFillOnGpu): today answers 0000 from the memo, which
    // no pin holds.
    recorder.NotePendingFill(address, keyCount, 0x00);
    MarkWritten(address, keyCount);
    NoteKeysFillOnGpu(address, keyCount, DccKeys::Clear0000);
    before = KeyPinTotals();
    Require(read() == DccKeys::Clear0000, "K1 (11): the keys of a pending 0000 fill did not read as 0000");
    if (pinning) Require(delta(before, KeyPin::FillNotPinned) == 1 && delta(before, KeyPin::Taken) == 0, "K1 (11): a pending fill of other keys was pinned or not counted");
    Require(proof.pinNote == 0 && proof.generation == 0 && DccKeyPinNote(address, keyCount) == 0, "K1 (11): a key proof was kept, or is pinnable, under a pending fill of other keys");
    recorder.Sync();

    // 9. A live unit shadow over pinned keys.
    const bool shadowed = keyPinShadowTests(device, recorder, mode);

    // 12. Nothing counted with the switch off; no answer at 2, no verify at 1, no race.
    const auto end = KeyPinTotals();
    if (mode == 0) Require(end.target == start.target && end.other == start.other, "K1: a pin was taken or counted with the switch off");
    if (mode == 1) Require(end[KeyPin::KeysVerified] == start[KeyPin::KeysVerified] && end[KeyPin::RefreshVerified] == start[KeyPin::RefreshVerified], "K1 1: a pin was verified");
    if (mode == 2) Require(end[KeyPin::KeyAnswer] == start[KeyPin::KeyAnswer] && end[KeyPin::RefreshHit] == start[KeyPin::RefreshHit], "K1 2: a pin answered");
    Require(end[KeyPin::Raced] == start[KeyPin::Raced], "K1: a pinned prediction was overtaken in a single-threaded test");
    std::cout << "Pinned key proofs (APS5_EXPERIMENT_KEY_PIN=" << mode << (shadowed ? "" : ", no unit shadow case") << "): passed\n";
    return true;
}

void targetKeyProofTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t bytes = 65536;
    void* block = AllocateWatched(bytes, bytes);
    if (block == nullptr) {
        std::cout << "no write watching: target key proofs not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    constexpr std::size_t keyCount = 1024;
    constexpr std::uint64_t surfaceBytes = keyCount * 256;
    auto* keys = static_cast<std::uint8_t*>(block);
    std::memset(keys, 0x20, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto scans = [] { return KeyProofCounts().rangeScanned; };
    const auto proofs = [] { return KeyProofCounts().rangeProved; };
    DccRangeProof proof;
    auto scanned = scans();
    auto proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::ClearRegister, "register clear keys read as another code");
    if (!KeyFastPath()) {
        Require(proof.generation == 0 && scans() == scanned && proofs() == proved, "target key proofs were kept while disabled");
        std::cout << "key fast path off: every target key read scans\n";
        return;
    }
    Require(scans() == scanned + 1 && proofs() == proved && proof.generation != 0 && proof.keys == DccKeys::ClearRegister, "the first read of a settled key range left no proof");
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::ClearRegister && scans() == scanned + 1 && proofs() == proved + 1, "an unchanged key range was scanned again");
    MarkWritten(address, keyCount);
    scanned = scans();
    proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::ClearRegister && scans() == scanned + 1 && proofs() == proved, "a key store over the range was answered from the proof");
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::ClearRegister && proofs() == proved + 1, "the rescan after a key store left no proof");
    std::memset(keys, 0xff, keyCount);
    scanned = scans();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Uncompressed && scans() == scanned + 1, "a CPU write of the keys was not seen");
    scanned = scans();
    proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes / 2, proof) == DccKeys::Uncompressed && scans() == scanned + 1 && proofs() == proved, "the proof of another key range answered");
    Require(ProvedCurrentDccKeys(address + 256, surfaceBytes / 2, proof) == DccKeys::Uncompressed && scans() == scanned + 2 && proofs() == proved, "the proof of a key range at another address answered");
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Uncompressed && scans() == scanned + 3, "a proof taken over a shorter range answered the whole range");
    recorder.NotePendingWrite(address, keyCount);
    MarkWritten(address, keyCount);
    NoteKeysFillOnGpu(address, keyCount, DccKeys::Clear0000);
    scanned = scans();
    proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Clear0000 && proof.generation == 0, "a pending key fill was not the answer, or its answer was kept");
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Clear0000 && scans() == scanned + 2 && proofs() == proved, "a pending key fill was answered from a proof");
    std::memset(keys, 0x00, keyCount);
    recorder.Submit();
    device.WaitQueue();
    recorder.Sync();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Clear0000 && proof.generation != 0, "the landed fill was not proved");
    proved = proofs();
    Require(ProvedCurrentDccKeys(address, surfaceBytes, proof) == DccKeys::Clear0000 && proofs() == proved + 1, "the proof after the fill landed did not hold");
}

// (8) The group close against a collect in progress: a CPU store made after the close must refuse
// the entry although another thread's resetting walk of the label's page (over a large range) may
// absorb the store and stamp the block with the generation it took before the close. And an entry
// whose batch was submitted before the close stays unclosed.
void closeRaceTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    using Refusal = Recorder::LabelRefusal;
    const std::array<std::byte, 4> one{std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
    Refusal refusal{};
    if (!Recorder::LateTrust()) return;
    recorder.NoteLabel(0x80000, one, 70, 0);
    recorder.Submit();
    Recorder::CloseLabelGroup(TrackerGeneration());
    Require(!Recorder::LookupLabel(0x80000, 4, 70, &refusal).has_value() && refusal == Refusal::Unclosed, "(8) a submitted batch's entry was closed");
    device.WaitQueue();
    recorder.Sync();
    constexpr std::size_t bytes = 1u << 20;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) return;
    const auto base = reinterpret_cast<std::uint64_t>(block);
    // Late in the range, so the walker is usually still ahead of the page when the store lands.
    const auto label = base + 4096 * 250;
    auto* word = reinterpret_cast<volatile std::uint32_t*>(label);
    *word = 0;
    CollectWritesUncached(base, bytes);
    // Joined on every exit, so a failed Require reports instead of terminating on the thread.
    struct Walker {
        std::atomic<bool> stop{false};
        std::thread thread;
        ~Walker() {
            stop.store(true, std::memory_order_relaxed);
            if (thread.joinable()) thread.join();
        }
    } walker;
    walker.thread = std::thread([&] {
        while (!walker.stop.load(std::memory_order_relaxed)) CollectWritesUncached(base, bytes);
    });
    for (std::uint32_t i = 0; i < 4000; ++i) {
        const std::uint64_t stamp = 100 + i;
        recorder.NoteLabel(label, one, stamp, 0);
        Recorder::CloseLabelGroup(TrackerGeneration());
        *word = i + 2;
        const auto hit = Recorder::LookupLabel(label, 4, stamp, &refusal);
        Require(hit.has_value() && hit->late && hit->generation != 0, "(8) the closed entry is not served late");
        Require(!UnchangedSinceCollected(label, 4, hit->generation), "(8) a store after the close is hidden by a racing collect");
    }
    walker.stop.store(true, std::memory_order_relaxed);
    walker.thread.join();
    recorder.Sync();
    Require(!Recorder::LookupLabel(label, 4, 5).has_value(), "(8) the entry outlived its batch");
    ReleaseWatched(block, bytes);
}

// A resource build over host-imported guest memory notes its in-place reads when its writes are
// marked, so a CPU copy into that memory is refused until the batch ran.
// Per-batch store runs (Recorder::RecordStore): the stores of a batch queue until Submit, a
// later writer or reader of a queued store's bytes has the run recorded in place first, and the
// stores land in the import's memory once the batch ran. With APS5_LABEL_RUNS_INLINE=1 or
// APS5_NO_LABEL_RUNS=1 (run the binary with either set) nothing queues and the stores land alike.
void storeRunTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: store runs not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the store test block");
    std::memset(block, 0, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the store test block refused: store runs not tested\n";
        return;
    }
    const bool perBatch = std::getenv("APS5_LABEL_RUNS_INLINE") == nullptr && std::getenv("APS5_NO_LABEL_RUNS") == nullptr && std::getenv("APS5_NO_LABEL_BATCHING") == nullptr;
    auto* words = static_cast<volatile std::uint32_t*>(block);
    const auto store = [&](std::size_t word, std::uint32_t value) {
        const std::array<std::byte, 4> value4{std::byte{static_cast<unsigned char>(value)}, std::byte{static_cast<unsigned char>(value >> 8u)}, std::byte{static_cast<unsigned char>(value >> 16u)}, std::byte{static_cast<unsigned char>(value >> 24u)}};
        recorder.RecordStore(import->buffer, address + word * 4 - import->base, value4, address + word * 4);
    };
    Require(recorder.Idle() && !recorder.HasQueuedStores(), "a fresh recorder has queued stores");
    store(0, 1);
    store(1, 2);
    store(4, 3);
    store(4, 4);
    Require(recorder.Recording(), "a store did not open a batch");
    Require(recorder.HasQueuedStores() == perBatch, "per-batch runs do not queue (or inline runs do)");
    Require(recorder.QueuedStoreOverlaps(address, 8) == perBatch && recorder.QueuedStoreOverlaps(address + 16, 4) == perBatch && !recorder.QueuedStoreOverlaps(address + 8, 8), "queued store ranges are wrong");
    const auto before = Recorder::StoreCounts();
    // A reader or writer of untouched bytes forces nothing; one over a queued store records the run.
    recorder.FlushStoresOverlapping(address + 8, 8);
    Require(recorder.HasQueuedStores() == perBatch, "a disjoint range flushed the run");
    recorder.FlushStoresOverlapping(address + 16, 4);
    Require(!recorder.HasQueuedStores() && Recorder::StoreCounts().runsForced == before.runsForced + (perBatch ? 1 : 0), "an overlapping writer did not record the run in place");
    store(2, 5);
    Require(recorder.HasQueuedStores() == perBatch, "a store after a forced run did not start a new run");
    recorder.Submit();
    Require(!recorder.HasQueuedStores() && Recorder::StoreCounts().runsAtSubmit == before.runsAtSubmit + (perBatch ? 1 : 0), "Submit did not record the run");
    recorder.Sync();
    Require(words[0] == 1 && words[1] == 2 && words[2] == 5 && words[4] == 4 && words[3] == 0, "the stores did not land (or the later store of a dword lost)");
    const auto after = Recorder::StoreCounts();
    Require(after.stores == before.stores + 1 && after.replaced == before.replaced + 0 && after.joined == before.joined + 0, "store counters are off");
    // The covered-access mask: reported once by the next Commands() call, then cleared.
    VkAccessFlags covered = 0xffffffffu;
    recorder.Commands(&covered);
    Require(covered == 0, "a new batch starts covered");
    recorder.MarkCovered(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    recorder.Commands(&covered);
    Require(covered == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT), "the covered mask is not reported");
    recorder.Commands(&covered);
    Require(covered == 0, "the covered mask survived a Commands() call");
    recorder.Sync();
    // The hazard tracker (counting mode only): accesses are noted without effect on the batch.
    const std::pair<std::uint64_t, std::uint64_t> range{address, address + 64};
    recorder.NoteAccess(Recorder::CommandClass::Fill, Recorder::Access{{}, std::span(&range, 1), {}, VK_PIPELINE_STAGE_TRANSFER_BIT});
    recorder.NoteAccess(Recorder::CommandClass::DispatchLeading, Recorder::Access{std::span(&range, 1), {}, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});
    Require(recorder.Recording() == Recorder::BarrierValidate(), "a noted access opened a batch with the tracker off (or none with it on)");
    recorder.Sync();
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    HostImportFor(context, address, bytes);
}

#ifdef _WIN32
bool deferredSettleTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    constexpr std::size_t unit = 65536;
    constexpr std::size_t settledBytes = std::size_t{1} << 20;
    if (context.hostImportAlignment == 0 || context.hostImportAlignment > unit || context.settleQueue == VK_NULL_HANDLE || !WriteWatched()) {
        std::cout << "host imports within 64 KiB, a settle queue or write watching unavailable: deferred settles not tested\n";
        return false;
    }
    const auto registerRange = [](void* block, std::size_t bytes) {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    };
    const auto unregisterRange = [](void* block) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    };
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    void* large = AllocateWatched(settledBytes, unit);
    void* spacer = AllocateWatched(unit, unit);
    void* small = AllocateWatched(unit, unit);
    struct Release {
        void* large;
        void* spacer;
        void* small;
        ~Release() {
            ReleaseWatched(large, settledBytes);
            ReleaseWatched(spacer, unit);
            ReleaseWatched(small, unit);
        }
    } release{large, spacer, small};
    const auto largeAddress = reinterpret_cast<std::uint64_t>(large);
    Require(reinterpret_cast<std::uint64_t>(small) != largeAddress + settledBytes && reinterpret_cast<std::uint64_t>(small) + unit != largeAddress, "(d) the 64 KiB block is next to the 1 MiB block, so their imports would join");
    registerRange(large, settledBytes);
    const auto* settled = HostImportFor(context, largeAddress, settledBytes);
    Require(settled != nullptr, "(d) the 1 MiB import was refused");
    if (settled->unwatched) {
        std::cout << "this device compares every Windows import: deferred settles not tested\n";
        unregisterRange(large);
        HostImportFor(context, largeAddress, settledBytes);
        return false;
    }
    const auto address = reinterpret_cast<std::uint64_t>(small);
    std::memset(small, 0x11, unit);
    registerRange(small, unit);
    const auto poll = [&](const std::function<bool()>& done) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!done() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            HostImportFor(context, address, unit);
        }
        return done();
    };
    const auto* import = HostImportFor(context, address, unit);
    Require(import != nullptr && import->unwatched && !Watched(address, unit), "(d) a 64 KiB import was not compared at first");
    Require(poll([&] { return Watched(address, unit); }), "(d) a 64 KiB import did not return to the write watch after its settle");
    const auto* back = HostImportFor(context, address, unit);
    Require(back == import && !back->unwatched, "(d) the import back in the write watch is still marked compared");
    BumpCollectEpoch();
    const auto collected = CollectWrites(address, unit);
    Require(collected != 0, "(d) the import back in the write watch is not collected");
    static_cast<volatile std::uint8_t*>(small)[4096] = 0x22;
    CollectWritesUncached(address, unit);
    Require(!UnchangedSince(address, unit, collected), "(d) a CPU store after the import returned to the write watch was missed");
    gpu.unlock();
    const auto beforeRounds = HostImportSettles();
    for (int round = 0; round < 64; ++round) {
        unregisterRange(small);
        registerRange(small, unit);
        const auto* again = HostImportFor(context, address, unit);
        Require(again != nullptr && again->unwatched && !Watched(address, unit), "(d) a 64 KiB buffer mapped again off the device lock was not imported compared");
    }
    const auto afterRounds = HostImportSettles();
    Require(afterRounds.submitted == beforeRounds.submitted && afterRounds.waited == beforeRounds.waited, "(d) importing 64 KiB buffers off the device lock submitted or waited for a settle");
    gpu.lock();
    const auto beforeLocked = HostImportSettles();
    for (int round = 0; round < 64; ++round) {
        unregisterRange(small);
        registerRange(small, unit);
        const auto* again = HostImportFor(context, address, unit);
        Require(again != nullptr && again->unwatched && !Watched(address, unit), "(d) a 64 KiB buffer mapped again under the device lock was not imported compared");
    }
    const auto afterLocked = HostImportSettles();
    Require(afterLocked.submitted == beforeLocked.submitted && afterLocked.waited == beforeLocked.waited, "(d) mapping a 64 KiB buffer again under the device lock, with no lookup between, submitted or waited for a settle");
    Require(poll([&] { return Watched(address, unit); }), "(d) the last buffer mapped again did not return to the write watch");
    unregisterRange(small);
    registerRange(small, unit);
    Require(HostImportFor(context, address, unit) != nullptr, "(d) a buffer mapped again after it returned to the write watch was not imported");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    HostImportFor(context, address, unit);
    unregisterRange(small);
    registerRange(small, unit);
    const auto* replaced = HostImportFor(context, address, unit);
    Require(replaced != nullptr && replaced->unwatched, "(d) a buffer mapped again while its settle may be in flight was not imported");
    Require(poll([&] { return Watched(address, unit); }), "(d) the buffer mapped again while its settle was in flight did not return to the write watch");
    unregisterRange(large);
    registerRange(large, unit);
    const auto* piece = HostImportFor(context, largeAddress, unit);
    Require(piece != nullptr && piece->unwatched, "(d) the first 64 KiB of the large block was not imported compared");
    unregisterRange(large);
    registerRange(large, settledBytes);
    const auto* whole = HostImportFor(context, largeAddress, settledBytes);
    Require(whole != nullptr && whole->bytes == settledBytes, "(d) the large block mapped whole again was not imported whole");
    Require(poll([&] { return Watched(largeAddress, settledBytes); }), "(d) a settled import over a piece that had left the write watch did not return to it");
    SetHostImportDeferredSettles(false);
    unregisterRange(small);
    registerRange(small, unit);
    const auto* kept = HostImportFor(context, address, unit);
    Require(kept != nullptr && kept->unwatched, "(d) with deferred settles off, a 64 KiB buffer mapped again was not imported compared");
    for (int round = 0; round < 20; ++round) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        HostImportFor(context, address, unit);
    }
    const bool stayed = !Watched(address, unit);
    SetHostImportDeferredSettles(true);
    Require(stayed, "(d) with deferred settles off, a 64 KiB import returned to the write watch");
    unregisterRange(small);
    unregisterRange(large);
    HostImportFor(context, address, unit);
    std::cout << "deferred settle tests passed\n";
    return true;
}

bool settledImportsWatchPrivate(const Context& context) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t settledBytes = std::size_t{1} << 20;
    void* large = AllocateWatched(settledBytes, 65536);
    const auto address = reinterpret_cast<std::uint64_t>(large);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(large, settledBytes, true, true, true);
    }
    const auto* settled = HostImportFor(context, address, settledBytes);
    const bool watched = settled != nullptr && !settled->unwatched;
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(large);
    }
    HostImportFor(context, address, settledBytes);
    ReleaseWatched(large, settledBytes);
    Require(settled != nullptr, "the 1 MiB import was refused");
    return watched;
}

bool settleFirstUseRaced(const Context& context, int copies) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t unit = 65536;
    void* small = AllocateWatched(unit, unit);
    struct Release {
        void* small;
        ~Release() { ReleaseWatched(small, unit); }
    } release{small};
    const auto address = reinterpret_cast<std::uint64_t>(small);
    std::memset(small, 0x11, unit);
    Recorder recorder(context);
    recorder.Activate();
    Buffer scratch(context, unit, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(small, unit, true, true, true);
    }
    const auto* import = HostImportFor(context, address, unit);
    Require(import != nullptr && import->unwatched && !Watched(address, unit), "(f) a 64 KiB import was not compared at first");
    const auto commands = recorder.Commands();
    for (int copy = 0; copy < copies; ++copy) CopyBuffer(context, commands, import->buffer, 0, scratch.Handle(), 0, unit);
    recorder.Submit();
    const auto submittedAt = std::chrono::steady_clock::now();
    const auto deadline = submittedAt + std::chrono::seconds(5);
    std::uint64_t collected = 0;
    bool returned = false;
    auto returnedAt = submittedAt;
    while (!returned && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        HostImportFor(context, address, unit);
        if (Watched(address, unit)) {
            returnedAt = std::chrono::steady_clock::now();
            BumpCollectEpoch();
            collected = CollectWrites(address, unit);
            returned = true;
        }
    }
    recorder.Reap();
    const bool racing = returned && recorder.InFlightBatches() != 0;
    while (recorder.InFlightBatches() != 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        recorder.Reap();
    }
    const auto reapedAt = std::chrono::steady_clock::now();
    recorder.Sync();
    CollectWritesUncached(address, unit);
    const bool unchanged = returned && UnchangedSince(address, unit, collected);
    const auto since = [&](std::chrono::steady_clock::time_point at) { return std::chrono::duration<double, std::milli>(at - submittedAt).count(); };
    std::cout << "(f) settled import back in the write watch " << since(returnedAt) << " ms after " << copies << " main-queue copies of it were submitted, copies reaped by " << since(reapedAt) << " ms; driver marks after its return=" << !unchanged << '\n';
    Require(returned, "(f) a settled import did not return to the write watch");
    Require(unchanged, "(f) the driver marked a settled import's pages after its settle read completed while earlier main-queue copies of it ran");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(small);
    }
    HostImportFor(context, address, unit);
    return racing;
}

bool settleFirstUseTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    constexpr std::size_t unit = 65536;
    if (context.hostImportAlignment == 0 || context.hostImportAlignment > unit || context.settleQueue == VK_NULL_HANDLE || !WriteWatched()) {
        std::cout << "host imports within 64 KiB, a settle queue or write watching unavailable: the first use of a settled import not tested\n";
        return false;
    }
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    if (!settledImportsWatchPrivate(context)) {
        std::cout << "this device compares every Windows import: the first use of a settled import not tested\n";
        return false;
    }
    for (int copies = 4096; copies <= 65536; copies *= 4) {
        if (settleFirstUseRaced(context, copies)) {
            std::cout << "settled import first use tests passed\n";
            return true;
        }
        std::cout << "(f) the main-queue copies had finished before the import returned to the write watch, so its read did not race them\n";
    }
    std::cout << "(f) the main-queue copies finished first in every round: the first use of a settled import not tested\n";
    return false;
}

bool settleEarlierImportTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    constexpr std::size_t unit = 65536;
    constexpr int copies = 4096;
    if (context.hostImportAlignment == 0 || context.hostImportAlignment > unit || context.settleQueue == VK_NULL_HANDLE || !WriteWatched()) {
        std::cout << "host imports within 64 KiB, a settle queue or write watching unavailable: an earlier import's first use after a settled import's return not tested\n";
        return false;
    }
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    if (!settledImportsWatchPrivate(context)) {
        std::cout << "this device compares every Windows import: an earlier import's first use after a settled import's return not tested\n";
        return false;
    }
    void* pair = AllocateWatched(2 * unit, unit);
    struct Release {
        void* pair;
        ~Release() {
            SetHostImportRangeWindow(std::uint64_t{64} << 20u);
            ReleaseWatched(pair, 2 * unit);
        }
    } release{pair};
    const auto address = reinterpret_cast<std::uint64_t>(pair);
    std::memset(pair, 0x11, 2 * unit);
    Recorder recorder(context);
    recorder.Activate();
    Buffer scratch(context, unit, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    SetHostImportRangeWindow(unit);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(pair, 2 * unit, true, true, true);
    }
    const auto* earlier = HostImportFor(context, address, 16);
    Require(earlier != nullptr && earlier->base == address && earlier->bytes == unit && earlier->unwatched, "(e) the first window of the range was not imported compared");
    const auto commands = recorder.Commands();
    for (int copy = 0; copy < copies; ++copy) CopyBuffer(context, commands, earlier->buffer, 0, scratch.Handle(), 0, unit);
    const auto* whole = HostImportFor(context, address + unit - 16, 32);
    Require(whole != nullptr && whole->base == address && whole->bytes == 2 * unit && whole->unwatched, "(e) an access across the windows did not replace the first window's import");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!Watched(address, 2 * unit) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        HostImportFor(context, address + unit - 16, 32);
    }
    Require(Watched(address, 2 * unit), "(e) the range's import did not return to the write watch");
    Require(recorder.Recording(), "(e) the earlier import's copies were submitted before the range's import returned to the write watch");
    BumpCollectEpoch();
    const auto collected = CollectWrites(address, 2 * unit);
    const auto submittedAt = std::chrono::steady_clock::now();
    recorder.Sync();
    const auto syncedAt = std::chrono::steady_clock::now();
    CollectWritesUncached(address, 2 * unit);
    const bool unchanged = UnchangedSince(address, 2 * unit, collected);
    std::cout << "(e) the range's import back in the write watch before the replaced window's first use ran (" << std::chrono::duration<double, std::milli>(syncedAt - submittedAt).count() << " ms of copies after it); driver marks after its return=" << !unchanged << '\n';
    Require(unchanged, "(e) the driver marked the pages of a settled import while a replaced import of them ran its first use");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(pair);
    }
    HostImportFor(context, address, 16);
    std::cout << "earlier import first use tests passed\n";
    return true;
}

bool settleRetireTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    constexpr std::size_t unit = 65536;
    constexpr int rounds = 64;
    if (context.hostImportAlignment == 0 || context.hostImportAlignment > unit || context.settleQueue == VK_NULL_HANDLE || !WriteWatched()) {
        std::cout << "host imports within 64 KiB, a settle queue or write watching unavailable: imports retired during their settle read not tested\n";
        return false;
    }
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    if (!settledImportsWatchPrivate(context)) {
        std::cout << "this device compares every Windows import: imports retired during their settle read not tested\n";
        return false;
    }
    void* pair = AllocateWatched(2 * unit, unit);
    struct Release {
        void* pair;
        ~Release() {
            SetHostImportRangeWindow(std::uint64_t{64} << 20u);
            ReleaseWatched(pair, 2 * unit);
        }
    } release{pair};
    const auto address = reinterpret_cast<std::uint64_t>(pair);
    std::memset(pair, 0x11, 2 * unit);
    const auto registerPair = [&] {
        GuestAllocations::Mutation mutation;
        mutation.Add(pair, 2 * unit, true, true, true);
    };
    const auto unregisterPair = [&] {
        GuestAllocations::Mutation mutation;
        mutation.Remove(pair);
    };
    Recorder recorder(context);
    recorder.Activate();
    SetHostImportRangeWindow(unit);
    registerPair();
    const auto before = HostImportSettles();
    for (int round = 0; round < rounds; ++round) {
        const auto* cell = HostImportFor(context, address, 16);
        Require(cell != nullptr && cell->base == address && cell->bytes == unit && cell->unwatched, "(r) the first window of the range was not imported compared");
        std::this_thread::sleep_for(std::chrono::microseconds(150));
        HostImportFor(context, address, 16);
        const auto* grown = HostImportFor(context, address + unit - 16, 32);
        Require(grown != nullptr && grown->base == address && grown->bytes == 2 * unit, "(r) an access across the windows did not replace the first window's import");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        HostImportFor(context, address + unit - 16, 32);
        recorder.Sync();
        recorder.FlushDeferredReleases();
        unregisterPair();
        registerPair();
    }
    const auto after = HostImportSettles();
    const auto* last = HostImportFor(context, address + unit - 16, 32);
    Require(last != nullptr && last->bytes == 2 * unit && last->unwatched, "(r) the range was not imported whole and compared after the rounds");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!Watched(address, 2 * unit) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        HostImportFor(context, address + unit - 16, 32);
    }
    const bool back = Watched(address, 2 * unit);
    std::cout << "(r) " << rounds << " window imports replaced after their settle read was submitted: " << after.parked - before.parked << " while the read ran, " << after.waited - before.waited << " waits, " << after.unmapWaits - before.unmapWaits << " unmaps waited for a read\n";
    Require(after.waited == before.waited, "(r) replacing an import during its settle read waited for the read");
    Require(after.parked != before.parked, "(r) no import was replaced while its settle read ran");
    Require(back, "(r) the import that replaced the windows did not return to the write watch");
    recorder.Sync();
    recorder.FlushDeferredReleases();
    unregisterPair();
    HostImportFor(context, address, 16);
    std::cout << "imports retired during their settle read tests passed\n";
    return true;
}

bool settleStorageImageTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    const auto& base = device.GetContext();
    constexpr std::size_t unit = 65536;
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    if (base.hostImportAlignment == 0 || base.hostImportAlignment > unit || base.settleQueue == VK_NULL_HANDLE || !WriteWatched()) {
        std::cout << "host imports within 64 KiB, a settle queue or write watching unavailable: a settled import under a storage image not tested\n";
        return false;
    }
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(base);
    if (!settledImportsWatchPrivate(base)) {
        std::cout << "this device compares every Windows import: a settled import under a storage image not tested\n";
        return false;
    }
    void* block = AllocateWatched(unit, unit);
    Require(block != nullptr, "(i) no watched block although write watching is on");
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, unit); }
    } release{block};
    const auto address = reinterpret_cast<std::uint64_t>(block);
    std::memset(block, 0x5a, unit);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, unit, true, true, true);
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = 128;
    resource.height = 128;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    Require(DescribeSurface(resource).guestBytes == unit, "(i) unexpected surface size");
    Require(Watched(address, unit) && ExistingHostImportFor(context, address, unit) == nullptr, "(i) the block was imported or out of the write watch before the image was made");
    Recorder recorder(context);
    recorder.Activate();
    const auto before = HostImportSettles();
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const auto* import = ExistingHostImportFor(context, address, unit);
        Require(import != nullptr && import->unwatched && !Watched(address, unit), "(i) the image's first upload did not import its block compared");
        const auto decided = [&] {
            const auto now = HostImportSettles();
            return now.rewatched != before.rewatched || now.keptCompared != before.keptCompared;
        };
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!decided() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            HostImportFor(context, address, unit);
        }
        Require(decided(), "(i) the settle read of the image's import did not finish");
        const auto after = HostImportSettles();
        const bool returned = Watched(address, unit);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
        const auto commands = recorder.Commands();
        recorder.Keep(image);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        image->MarkDirty();
        image->Refresh();
        image->WriteBack();
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        std::vector<std::byte> read(unit);
        AgcDriver::GuestMemory::Read(address, read);
        std::size_t lost = 0;
        for (std::size_t at = 0; at < unit; at += 4) {
            for (std::size_t c = 0; c < 4; ++c) {
                if (std::to_integer<std::uint8_t>(read[at + c]) == red[c]) continue;
                ++lost;
                break;
            }
        }
        std::cout << "(i) the image's import " << (returned ? "returned to the write watch" : "stayed compared") << " after its settle read (" << after.rewatched - before.rewatched << " back in the write watch, " << after.keptCompared - before.keptCompared << " kept compared); the refresh after the image's GPU results lost " << lost << " of " << unit / 4 << " texels\n";
        Require(lost == 0, "(i) the refresh after the image's GPU results dropped " + std::to_string(lost) + " of " + std::to_string(unit / 4) + " texels: the import's return to the write watch stamped its block as a CPU store after the image took its generation");
        Require(!returned && after.keptCompared == before.keptCompared + 1, "(i) the settled import under a live storage image returned to the write watch");
    }
    recorder.Sync();
    recorder.FlushDeferredReleases();
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    HostImportFor(context, address, unit);
    std::cout << "a settled import under a storage image tests passed\n";
    return true;
}
#endif

void remappedImportTests(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: remapped imports not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the remap test block");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the remap test block refused: remapped imports not tested\n";
        return;
    }
    const auto first = HostImportSerial(context, address, bytes, true);
    Require(first != 0 && HostImportSerial(context, address, bytes, true) == first, "an unchanged range lost its import");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        mutation.Add(block, bytes, true, true, true);
    }
    Require(HostImportFor(context, address, bytes) != nullptr, "the remapped range was not imported again");
    const auto second = HostImportSerial(context, address, bytes, true);
    Require(second != 0 && second != first, "the import of a range mapped again was kept");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    HostImportFor(context, address, bytes);
}

void hostImportRetirementTests(const Device& device) {
    const auto& context = device.GetContext();
    constexpr std::size_t UnitBytes = 65536;
    constexpr std::size_t AllocationBytes = 4 * UnitBytes;
    constexpr std::size_t CopyBytes = 256;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, AllocationBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(UnitBytes, AllocationBytes);
#endif
    Require(block != nullptr, "cannot allocate the host import retirement block");
    const auto release = [](void* pointer) {
#ifdef _WIN32
        VirtualFree(pointer, 0, MEM_RELEASE);
#else
        std::free(pointer);
#endif
    };
    std::unique_ptr<void, decltype(release)> allocation(block, release);
    auto* memory = static_cast<std::byte*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    Recorder recorder(context);
    recorder.Activate();
    Buffer readback(context, CopyBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::array<bool, 4> registered{};
    const auto cleanup = [&] {
        if (!gpu.owns_lock()) gpu.lock();
        recorder.Sync();
        recorder.FlushDeferredReleases();
        ClearHostImports(context.device);
        GuestAllocations::Mutation mutation;
        for (std::size_t index = 0; index < registered.size(); ++index) {
            if (!registered[index]) continue;
            mutation.Remove(memory + index * UnitBytes);
            registered[index] = false;
        }
        SetHostImportWindow(0);
    };
    const auto add = [&](std::size_t index) {
        GuestAllocations::Mutation mutation;
        mutation.Add(memory + index * UnitBytes, UnitBytes, true, true, true);
        registered[index] = true;
    };
    const auto mapped = [&](std::size_t index) {
        GuestAllocations::Mapped result;
        for (const auto& range : GuestAllocations::GuestAllocationsAcquire_nid_postfix()) {
            if (range->address == address + index * UnitBytes) result.push_back(range);
        }
        Require(result.size() == 1, "host import retirement lost its registered range");
        return result;
    };
    bool deferredEveryRetirement = true;
    bool preservedEveryRecorder = true;
    try {
        const std::array<const char*, 3> shapes{"same-base growth", "earlier-base overlap", "predecessor split"};
        for (std::size_t shape = 0; shape < shapes.size(); ++shape) {
            SetHostImportWindow(0);
            Require(ImportMappedRanges(context, {}, GuestAllocations::GuestAllocationsGeneration_nid_postfix(), true), "host import retirement could not adopt the device");
            const std::size_t first = shape == 1 ? 1 : 0;
            add(first);
            if (shape == 2) add(1);
            for (std::size_t index = 0; index < AllocationBytes; ++index) memory[index] = static_cast<std::byte>((index * 37u + shape * 19u + 11u) & 0xffu);
            const auto initial = mapped(first);
            const auto initialGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
            gpu.unlock();
            const bool fresh = ImportMappedRanges(context, initial, initialGeneration, false);
            const bool unchanged = ImportMappedRanges(context, initial, initialGeneration, false);
            gpu.lock();
            Require(fresh && unchanged, "host import retirement blocked a fresh or unchanged fast import");
            const auto sourceAddress = address + (shape == 2 ? UnitBytes : first * UnitBytes);
            const auto* imported = HostImportFor(context, sourceAddress, CopyBytes);
            Require(imported != nullptr, "host import retirement could not import its source");
            const auto source = imported->buffer;
            const auto sourceOffset = sourceAddress - imported->base;
            std::array<std::byte, CopyBytes> expected{};
            std::memcpy(expected.data(), reinterpret_cast<const void*>(sourceAddress), CopyBytes);
            recorder.Commands();
            recorder.Submit();
            device.WaitQueue();
            Require(!recorder.Recording() && recorder.InFlightBatches() == 1, "host import retirement requires a submitted batch without an open batch");
            std::size_t changed = 0;
            if (shape == 0) {
                changed = 1;
                add(changed);
            } else if (shape == 1) {
                add(changed);
            } else {
                SetHostImportWindow(UnitBytes);
                add(3);
                changed = 1;
            }
            const auto ranges = mapped(changed);
            const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
            gpu.unlock();
            const bool deferred = !ImportMappedRanges(context, ranges, generation, false);
            gpu.lock();
            const bool preserved = !recorder.Recording() && recorder.InFlightBatches() == 1;
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, source, sourceOffset, readback.Handle(), 0, CopyBytes);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            Require(ImportMappedRanges(context, ranges, generation, true), "host import retirement refused its retry under the GPU lock");
            Require(HostImportFor(context, sourceAddress, CopyBytes) != nullptr, "host import retirement lost the replacement import");
            recorder.Sync();
            readback.Invalidate();
            Require(std::equal(expected.begin(), expected.end(), readback.Bytes().begin()), "host import retirement changed bytes copied from the retired import");
            std::cout << "host import retirement: " << shapes[shape] << " GPU readback verified; deferred=" << deferred << "; recorder preserved=" << preserved << '\n';
            deferredEveryRetirement = deferredEveryRetirement && deferred;
            preservedEveryRecorder = preservedEveryRecorder && preserved;
            cleanup();
        }
    } catch (...) {
        cleanup();
        throw;
    }
    Require(deferredEveryRetirement, "unlocked mapping retired an import instead of requesting the GPU lock");
    Require(preservedEveryRecorder, "unlocked mapping changed recorder batches");
    std::cout << "Host import retirement locking and GPU readback tests passed\n";
}
#ifdef _WIN32
void secondMappingImportTests(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: imports after a second mapping not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
    constexpr std::size_t page = 16384;
    struct Release {
        void* mapping = nullptr;
        void* alias = nullptr;
        std::int64_t physical = -1;
        ~Release() {
            if (alias != nullptr) sceKernelMunmap(alias, page);
            if (mapping != nullptr) sceKernelMunmap(mapping, bytes);
            if (physical >= 0) sceKernelReleaseDirectMemory(physical, bytes);
        }
    } release;
    Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, bytes, bytes, 0, &release.physical) == 0, "allocate the second mapping test backing");
    Require(sceKernelMapDirectMemory(&release.mapping, bytes, 0x33, 0, release.physical, bytes) == 0, "map the second mapping test backing");
    const auto address = reinterpret_cast<std::uint64_t>(release.mapping);
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of singly mapped direct memory refused: imports after a second mapping not tested\n";
        return;
    }
    const auto first = HostImportSerial(context, address, bytes, true);
    Require(first != 0 && HostImportSerial(context, address, bytes, true) == first, "singly mapped direct memory lost its import");
    Require(sceKernelMapDirectMemory(&release.alias, page, 0x33, 0, release.physical, page) == 0, "map the first page a second time");
    Require(HostImportFor(context, address, bytes) != nullptr, "direct memory was not imported again after a second mapping of its first page");
    const auto second = HostImportSerial(context, address, bytes, true);
    Require(second != 0 && second != first, "the import of direct memory whose pages were replaced was kept");
    HostImportFor(context, address, bytes);
}

void requireImportServesGuest(const Context& context, Recorder& recorder, const HostImport& import, std::uint64_t begin, std::uint64_t end, std::uint32_t salt, const std::string& what) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t page = 4096;
    constexpr std::size_t probe = 256;
    const auto bytes = static_cast<std::size_t>(end - begin);
    Require(begin >= import.base && end <= import.base + import.bytes, what + " is not inside its import");
    const auto collected = CollectWritesUncached(begin, bytes);
    Require(collected != 0 && Watched(begin, bytes), what + " left the write watch");
    Require(UnchangedSinceCollected(begin, page, collected) && UnchangedSinceCollected(end - page, page, collected), what + " reports writes nobody made");
    auto* first = reinterpret_cast<std::byte*>(begin);
    auto* last = reinterpret_cast<std::byte*>(end - page);
    for (std::size_t index = 0; index < probe; ++index) {
        first[index] = static_cast<std::byte>((index * 7u + salt) & 0xffu);
        last[index] = static_cast<std::byte>((index * 13u + salt * 5u + 1u) & 0xffu);
    }
    Require(!UnchangedSinceCollected(begin, page, collected) && !UnchangedSinceCollected(end - page, page, collected), "a CPU store into " + what + " was not collected");
    Buffer readback(context, 2 * probe, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto commands = recorder.Commands();
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    CopyBuffer(context, commands, import.buffer, begin - import.base, readback.Handle(), 0, probe);
    CopyBuffer(context, commands, import.buffer, end - page - import.base, readback.Handle(), probe, probe);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    recorder.Sync();
    readback.Invalidate();
    const auto seen = readback.Bytes();
    Require(std::equal(first, first + probe, seen.begin()) && std::equal(last, last + probe, seen.begin() + probe), "the GPU reads other bytes than the guest's through " + what);
}

int reimportBench(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0 || context.settleQueue == VK_NULL_HANDLE || !AgcDriver::GuestMemory::WriteWatched()) {
        std::cout << "skipped, the re-import bench requires host imports, a settle queue and write watching\n";
        return 77;
    }
    constexpr std::size_t bytes = 1126ull << 20u;
    constexpr std::size_t step = 1440ull << 10u;
    constexpr std::size_t guard = 16ull << 10u;
    constexpr std::size_t inside = 256ull << 20u;
    constexpr std::size_t moves = 5;
    constexpr std::size_t page = 4096;
    constexpr std::size_t alignment = 2ull << 20u;
    constexpr std::size_t requiredImportBytes = bytes + inside;
    const auto availableImportBytes = HostImportBudgetRemaining(context);
    if (availableImportBytes < requiredImportBytes) {
        std::cout << "skipped, the re-import bench needs " << (requiredImportBytes >> 20u) << " MiB of simultaneous host imports; the current budget has " << (availableImportBytes >> 20u) << " MiB available\n";
        return 77;
    }
    struct Release {
        void* mapping = nullptr;
        std::int64_t physical = -1;
        ~Release() {
            if (mapping != nullptr) sceKernelMunmap(mapping, bytes);
            if (physical >= 0) sceKernelReleaseDirectMemory(physical, bytes);
        }
    };
    enum class Cut : std::uint8_t { Head, Tail, Inside };
    struct Pattern {
        Cut cut;
        const char* name;
        const char* moving;
    };
    constexpr std::array<Pattern, 3> patterns{{
        {Cut::Head, "head", "the head loses GPU access 1440 KiB at a time"},
        {Cut::Tail, "tail", "the tail loses GPU access 1440 KiB at a time"},
        {Cut::Inside, "inside", "a 16 KiB page inside the kept side loses all access, 256 MiB in and then 1440 KiB above the one before"},
    }};
    const auto milliseconds = [](std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) { return std::chrono::duration<double, std::milli>(to - from).count(); };
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    Recorder recorder(context);
    recorder.Activate();
    const auto cleanup = [&] {
        if (!gpu.owns_lock()) gpu.lock();
        recorder.Sync();
        recorder.FlushDeferredReleases();
        ClearHostImports(context.device);
    };
    std::uint32_t salt = 0;
    const auto measure = [&](const Pattern& pattern) {
        Release release;
        Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, bytes, alignment, 0, &release.physical) == 0, "allocate the re-import bench backing");
        Require(sceKernelMapDirectMemory(&release.mapping, bytes, 0x33, 0, release.physical, alignment) == 0, "map the re-import bench backing");
        const auto address = reinterpret_cast<std::uint64_t>(release.mapping);
        auto* memory = static_cast<std::byte*>(release.mapping);
        for (std::size_t offset = 0; offset < bytes; offset += 4 * page) std::memset(memory + offset, 0x5a, 64);
        std::cout << "re-import bench, " << pattern.name << " cuts: " << (bytes >> 20u) << " MiB of singly mapped direct memory at 0x" << std::hex << address << std::dec << ", watched=" << AgcDriver::GuestMemory::Watched(address, bytes) << ", 1 of 4 pages nonzero, " << pattern.moving << '\n';
        gpu.lock();
        auto begin = address;
        auto end = address + bytes;
        std::size_t again = 0;
        double slowest = 0;
        for (std::size_t move = 0; move <= moves; ++move) {
            double moveMs = 0;
            std::uint64_t smaller = 0;
            std::uint64_t smallerEnd = 0;
            if (move != 0) {
                std::uint64_t from = 0;
                std::uint64_t to = 0;
                if (pattern.cut == Cut::Head) {
                    from = begin;
                    to = begin = from + step;
                } else if (pattern.cut == Cut::Tail) {
                    to = end;
                    from = end = to - step;
                } else {
                    from = move == 1 ? address + inside : begin + step;
                    to = from + guard;
                    smaller = begin;
                    smallerEnd = from;
                    begin = to;
                }
                gpu.unlock();
                const auto moving = std::chrono::steady_clock::now();
                Require(sceKernelMprotect(reinterpret_cast<void*>(from), static_cast<std::size_t>(to - from), pattern.cut == Cut::Inside ? 0 : 0x03) == 0, "cut the re-import bench range");
                moveMs = milliseconds(moving, std::chrono::steady_clock::now());
                gpu.lock();
            }
            const auto started = std::chrono::steady_clock::now();
            const auto* import = HostImportFor(context, begin, static_cast<std::size_t>(end - begin));
            const auto imported = std::chrono::steady_clock::now();
            Require(import != nullptr && import->base == begin && import->base + import->bytes == end, "the re-import bench range was not imported whole");
            const char* kind = import->alias != nullptr ? "through a shared alias" : import->unwatched ? "compared" : "settled and watched";
            const bool kept = import->memoryBase == address && import->pinnedBytes == bytes;
            if (move != 0) {
                again += kept ? 0 : 1;
                slowest = std::max(slowest, milliseconds(started, imported));
            }
            double smallerMs = 0;
            if (smallerEnd != smaller) {
                const auto importing = std::chrono::steady_clock::now();
                Require(HostImportFor(context, smaller, static_cast<std::size_t>(smallerEnd - smaller)) != nullptr, "the smaller side of a cut inside the re-import bench range was not imported");
                smallerMs = milliseconds(importing, std::chrono::steady_clock::now());
            }
            const auto syncing = std::chrono::steady_clock::now();
            recorder.Sync();
            recorder.FlushDeferredReleases();
            const auto synced = std::chrono::steady_clock::now();
            std::printf("re-import %s %zu: %.1f MiB at +%.2f MiB, %s, %s, cut %.2f ms, import %.2f ms", pattern.name, move, static_cast<double>(end - begin) / 1048576.0, static_cast<double>(begin - address) / 1048576.0, kind, move == 0 ? "first import" : kept ? "kept" : "imported again", moveMs, milliseconds(started, imported));
            if (smallerEnd != smaller) std::printf(", smaller side %.2f MiB imported in %.2f ms", static_cast<double>(smallerEnd - smaller) / 1048576.0, smallerMs);
            std::printf(", recorder sync after it %.2f ms, %.1f MiB imported in all\n", milliseconds(syncing, synced), static_cast<double>(HostImportLiveBytes()) / 1048576.0);
            std::fflush(stdout);
        }
        const auto* tail = HostImportFor(context, begin, static_cast<std::size_t>(end - begin));
        Require(tail != nullptr && tail->base == begin && tail->base + tail->bytes == end, "the re-import bench lost the import of its kept side");
        requireImportServesGuest(context, recorder, *tail, begin, end, ++salt, std::string("the kept side after the ") + pattern.name + " cuts");
        std::printf("re-import bench, %s cuts: the kept side was imported again after %zu of %zu cuts, slowest import after a cut %.2f ms; CPU stores into it are collected and the GPU reads them through its buffer\n", pattern.name, again, moves, slowest);
        cleanup();
        gpu.unlock();
        return again;
    };
    std::string importedAgain;
    gpu.unlock();
    try {
        for (const auto& pattern : patterns) {
            if (measure(pattern) != 0) importedAgain += std::string(importedAgain.empty() ? "" : ", ") + pattern.name;
        }
    } catch (...) {
        cleanup();
        throw;
    }
    gpu.lock();
    Require(importedAgain == "head, tail, inside", "a cut kept memory imported across the changed host range instead of importing the remaining side again: " + importedAgain);
    std::printf("re-import bench: every cut at the head, at the tail and inside the range retired the original import before importing the remaining side\n");
    return 0;
}

int hostImportCutTests(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0 || context.settleQueue == VK_NULL_HANDLE || !AgcDriver::GuestMemory::WriteWatched()) {
        std::cout << "skipped, the host import cut tests require host imports, a settle queue and write watching\n";
        return 77;
    }
    constexpr std::size_t bytes = 32ull << 20u;
    constexpr std::size_t run = 2ull << 20u;
    constexpr std::size_t guard = 16ull << 10u;
    constexpr std::size_t page = 4096;
    constexpr std::size_t inside = 6ull << 20u;
    struct Case {
        const char* name;
        std::size_t offset;
        std::size_t length;
        bool unmap;
        std::size_t kept;
        std::size_t keptEnd;
        std::size_t smaller;
        std::size_t smallerEnd;
    };
    constexpr std::array<Case, 5> cases{{
        {"a page made inaccessible inside the import, nearer its start", inside + guard, guard, false, inside + 2 * guard, bytes, 0, inside + guard},
        {"a page made inaccessible inside the import, nearer its end", bytes - inside - 2 * guard, guard, false, 0, bytes - inside - 2 * guard, bytes - inside - guard, bytes},
        {"an unmap of a 2 MiB run inside the import", inside, run, true, inside + run, bytes, 0, inside},
        {"an unmap of the import's first 2 MiB run", 0, run, true, run, bytes, 0, 0},
        {"an unmap of 1440 KiB at the start of the import's first 2 MiB run", 0, 1440ull << 10u, true, 1440ull << 10u, bytes, 0, 0},
    }};
    struct Backing {
        void* mapping = nullptr;
        std::int64_t physical = -1;
        std::size_t hole = 0;
        std::size_t holeBytes = 0;
        ~Backing() {
            auto* memory = static_cast<std::byte*>(mapping);
            if (memory != nullptr && hole != 0) sceKernelMunmap(memory, hole);
            if (memory != nullptr && hole + holeBytes < bytes) sceKernelMunmap(memory + hole + holeBytes, bytes - hole - holeBytes);
            if (physical >= 0) sceKernelReleaseDirectMemory(physical, bytes);
        }
    };
    std::unique_lock gpu(GpuMutex());
    PrepareImportWatch(context);
    Recorder recorder(context);
    recorder.Activate();
    const auto cleanup = [&] {
        if (!gpu.owns_lock()) gpu.lock();
        recorder.Sync();
        recorder.FlushDeferredReleases();
        ClearHostImports(context.device);
    };
    std::uint32_t salt = 0;
    const auto check = [&](const Case& test) {
        const std::string name = test.name;
        Backing backing;
        Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, bytes, run, 0, &backing.physical) == 0, "allocate the host import cut test backing");
        Require(sceKernelMapDirectMemory(&backing.mapping, bytes, 0x33, 0, backing.physical, run) == 0, "map the host import cut test backing");
        const auto address = reinterpret_cast<std::uint64_t>(backing.mapping);
        auto* memory = static_cast<std::byte*>(backing.mapping);
        for (std::size_t offset = 0; offset < bytes; offset += page) std::memset(memory + offset, static_cast<int>(0x40u + (offset / page) % 64u), 64);
        gpu.lock();
        const auto* whole = HostImportFor(context, address, bytes);
        Require(whole != nullptr && whole->base == address && whole->bytes == bytes && whole->alias == nullptr && !whole->unwatched, name + ": the range was not imported whole, settled and watched");
        auto* cut = memory + test.offset;
        gpu.unlock();
        if (test.unmap) {
            Require(sceKernelMunmap(cut, test.length) == 0, name + ": the unmap failed");
            backing.hole = test.offset;
            backing.holeBytes = test.length;
        } else {
            Require(sceKernelMprotect(cut, test.length, 0) == 0, name + ": the protection failed");
        }
        gpu.lock();
        const auto begin = address + test.kept;
        const auto end = address + test.keptEnd;
        const auto* kept = HostImportFor(context, begin, static_cast<std::size_t>(end - begin));
        Require(kept != nullptr && kept->base == begin && kept->base + kept->bytes == end, name + ": the larger side is not imported whole");
        Require(kept->memoryBase == begin && kept->pinnedBytes == end - begin, "the side of " + name + " is served from the import that pinned the changed host range");
        requireImportServesGuest(context, recorder, *kept, begin, end, ++salt, "the larger side of " + name);
        if (test.smallerEnd != test.smaller) {
            const auto* smaller = HostImportFor(context, address + test.smaller, test.smallerEnd - test.smaller);
            Require(smaller != nullptr && smaller->base == address + test.smaller && smaller->bytes == test.smallerEnd - test.smaller && smaller->memoryBase == address + test.smaller && smaller->pinnedBytes == test.smallerEnd - test.smaller, "the smaller side of " + name + " is not imported apart from the larger one");
            requireImportServesGuest(context, recorder, *smaller, address + test.smaller, address + test.smallerEnd, ++salt, "the smaller side of " + name);
            Require(HostImportFor(context, begin, static_cast<std::size_t>(end - begin)) == kept && kept->memoryBase == begin && kept->pinnedBytes == end - begin, "importing the smaller side of " + name + " replaced the safe import of the larger one");
        }
        if (test.unmap) {
            gpu.unlock();
            void* again = cut;
            Require(sceKernelMapDirectMemory(&again, test.length, 0x33, 0x10, backing.physical + static_cast<std::int64_t>(test.offset), guard) == 0 && again == cut, name + ": the pages could not be mapped again");
            backing.hole = 0;
            backing.holeBytes = 0;
            gpu.lock();
            const auto from = reinterpret_cast<std::uint64_t>(cut);
            const auto* remapped = HostImportFor(context, address, bytes);
            Require(remapped != nullptr && remapped->base == address && remapped->bytes == bytes, name + ": the range mapped again is not imported whole");
            Require(remapped->memoryBase == address && remapped->pinnedBytes == bytes, "after " + name + " the pages mapped again are served from the import that pinned the old ones");
            requireImportServesGuest(context, recorder, *remapped, from, from + test.length, ++salt, "the pages mapped again after " + name);
            requireImportServesGuest(context, recorder, *remapped, begin, end, ++salt, "the larger side after " + name + " and a new mapping");
        }
        cleanup();
        gpu.unlock();
        std::cout << "host import after cut: " << name << ": the changed import was retired before the remaining range was imported, and the GPU and write watch see the guest's bytes" << (test.unmap ? ", also after the pages were mapped again" : "") << '\n';
    };
    const auto remapWhileTrimmed = [&] {
        const std::string name = "an unmap and a new mapping of the retained side while the old import is retired";
        Backing backing;
        Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, bytes, run, 0, &backing.physical) == 0, "allocate the host import cut test backing");
        Require(sceKernelMapDirectMemory(&backing.mapping, bytes, 0x33, 0, backing.physical, run) == 0, "map the host import cut test backing");
        const auto address = reinterpret_cast<std::uint64_t>(backing.mapping);
        auto* memory = static_cast<std::byte*>(backing.mapping);
        for (std::size_t offset = 0; offset < bytes; offset += page) std::memset(memory + offset, static_cast<int>(0x40u + (offset / page) % 64u), 64);
        gpu.lock();
        const auto* whole = HostImportFor(context, address, bytes);
        Require(whole != nullptr && whole->base == address && whole->bytes == bytes && whole->alias == nullptr && !whole->unwatched, name + ": the range was not imported whole, settled and watched");
        gpu.unlock();
        Require(sceKernelMprotect(memory, run, 0x03) == 0, name + ": the protection of the head failed");
        Require(sceKernelMunmap(memory + run, bytes - run) == 0, name + ": the unmap of the kept side failed");
        backing.hole = run;
        backing.holeBytes = bytes - run;
        void* again = memory + run;
        Require(sceKernelMapDirectMemory(&again, bytes - run, 0x33, 0x10, backing.physical + static_cast<std::int64_t>(run), guard) == 0 && again == memory + run, name + ": the kept side could not be mapped again");
        backing.hole = 0;
        backing.holeBytes = 0;
        gpu.lock();
        const auto begin = address + run;
        const auto end = address + bytes;
        const auto* remapped = HostImportFor(context, begin, static_cast<std::size_t>(end - begin));
        Require(remapped != nullptr && remapped->base == begin && remapped->base + remapped->bytes == end, name + ": the pages mapped again are not imported whole");
        requireImportServesGuest(context, recorder, *remapped, begin, end, ++salt, "the pages mapped again after " + name);
        Require(remapped->memoryBase == begin && remapped->pinnedBytes == end - begin, "after " + name + " the pages mapped again are served from the import that pinned the old ones");
        cleanup();
        gpu.unlock();
        std::cout << "host import cut: " << name << ": the pages mapped again were imported again and the GPU and the write watch see the guest's bytes\n";
    };
    gpu.unlock();
    try {
        for (const auto& test : cases) check(test);
        remapWhileTrimmed();
    } catch (...) {
        cleanup();
        throw;
    }
    gpu.lock();
    std::cout << "Host import cut lifetime tests passed\n";
    return 0;
}
#endif

void movedMetadataTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: moved DCC metadata not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the moved metadata block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* firstKeys = texels + surfaceBytes;
    auto* secondKeys = firstKeys + 4096;
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the moved metadata block refused: moved DCC metadata not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    Require(DescribeSurface(resource).guestBytes == surfaceBytes, "the moved metadata surface has an unexpected size");
    const auto first = address + surfaceBytes;
    const auto second = first + 4096;
    const auto withKeys = [&](std::uint64_t keys) {
        auto described = resource;
        described.dccAddress = keys;
        return described;
    };
    const auto holds = [&](const StorageTexture& image, std::array<std::uint8_t, 4> texel) {
        Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {side, side, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        const auto pixels = readback.Bytes();
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            if (std::to_integer<std::uint8_t>(pixels[i]) != texel[i % 4]) return false;
        }
        return true;
    };
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(firstKeys, 0xff, keyCount);
    std::memset(secondKeys, 0x00, keyCount);
    const auto original = CachedStorageSurface(context, withKeys(first));
    Require(original->Descriptor().dccAddress == first && holds(*original, {0x55, 0x55, 0x55, 0x55}), "the first image does not hold the stored texels");
    const auto moved = CachedStorageSurface(context, withKeys(second));
    Require(moved->Descriptor().dccAddress == second, "the storage image kept the keys the surface no longer names");
    Require(!StorageImageCached(context, original.get()), "the image of the old keys is still the surface's");
    Require(holds(*moved, {0, 0, 0, 0}), "a fast clear of the moved keys did not reach the image");
    Require(CachedStorageSurface(context, resource) == moved && moved->Descriptor().dccAddress == second, "a descriptor without metadata replaced the image");
    const auto back = CachedStorageSurface(context, withKeys(first));
    Require(back != moved && back->Descriptor().dccAddress == first, "the keys moving back did not remake the image");
    Require(holds(*back, {0x55, 0x55, 0x55, 0x55}), "the image remade under the first keys does not hold the stored texels");
}

void viewPastLastMipTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: views past the last mip not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t bytes = 0x60000;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the past-last-mip block");
    auto* texels = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the past-last-mip block refused: views past the last mip not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    const auto view = [&](std::uint32_t level) {
        const std::array<std::uint32_t, 8> words{
            static_cast<std::uint32_t>(address >> 8u),
            static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((side - 1u) & 3u) << 30u),
            ((side - 1u) >> 2u) | ((side - 1u) << 14u),
            0xfacu | (level << 12u) | (level << 16u) | (0x1bu << 20u) | (9u << 28u),
            0u,
            2u << 4u,
            0u,
            0u,
        };
        return DecodeTextureResource(words);
    };
    const auto texel = [&](const StorageTexture& image, std::uint32_t level) {
        Buffer readback(context, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
        copy.imageExtent = {1, 1, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        std::uint32_t value = 0;
        std::memcpy(&value, readback.Bytes().data(), sizeof(value));
        return value;
    };
    std::memset(texels, 0x55, bytes);
    std::memset(texels + 0x8400, 0x66, 4);
    std::memset(texels + 0x4800, 0x77, 4);
    const auto allocatedView = view(2);
    const auto pastView = view(3);
    Require(allocatedView.mipCount == 3 && pastView.mipCount == 4 && DescribeSurface(pastView).guestBytes == bytes, "a 256x256 32 bpp SW_64KB_R_X surface viewed past MAX_MIP 2 decoded wrongly");
    const auto allocated = CachedStorageSurface(context, allocatedView);
    Require(allocated->Descriptor().mipCount == 3, "the allocated levels' storage image has an unexpected level count");
    const auto past = CachedStorageSurface(context, pastView);
    Require(past->Descriptor().mipCount == 4, "the storage image of a view past the last mip lacks the level it names");
    Require(!StorageImageCached(context, allocated.get()), "the allocated levels' image is still the surface's after a view past its last mip");
    Require(CachedStorageSurface(context, allocatedView) == past, "a view of the allocated levels took another image than the view past the last mip");
    Require(texel(*past, 2) == 0x66666666u, "level 2 was not read from its addrlib tail slot");
    Require(texel(*past, 3) == 0x77777777u, "the level past the last mip was not read from its addrlib tail slot");
}

void resourceReadTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: resource read notes not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the test block");
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the test block refused: resource read notes not tested\n";
        return;
    }
    const auto element = address + 4096;
    constexpr std::size_t elementBytes = 1024;
    {
        GuestBufferMemory memory(context);
        memory.AddReadable(element, elementBytes);
        memory.Upload(false);
        const auto reads = memory.InPlaceReads();
        Require(reads.size() == 1 && reads[0].first == element && reads[0].second == element + elementBytes, "an imported region is not an in-place read");
        Require(memory.Writes().empty(), "a readable element counts as written");
    }
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {static_cast<std::uint32_t>(element), static_cast<std::uint32_t>(element >> 32u) & 0xffffu, elementBytes, 0x31000000u};
    binding.bufferWritten = {false};
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    {
        ShaderResources resources(context, compute);
        Require(recorder.Idle() && !recorder.PendingReadOverlaps(element, 16), "the build itself noted a read");
        resources.MarkGpuWrites(recorder);
        Require(recorder.PendingReadOverlaps(element, 16) && recorder.PendingReadOverlaps(element + elementBytes - 4, 4) && !recorder.PendingReadOverlaps(element + elementBytes, 16), "the build's in-place read was not noted");
        Require(!recorder.PendingWriteOverlaps(element, elementBytes), "a read-only element was noted as written");
        const auto info = recorder.DescribePendingRead(element, 16);
        Require(info.has_value() && info->kind == Kind::DispatchElement && info->open, "the build's read has the wrong kind");
        recorder.Submit();
        device.WaitQueue();
        Require(!recorder.PendingReadOverlaps(element, 16), "the build's read refuses after its batch ran");
        recorder.Sync();
    }
    std::weak_ptr<ShaderResources::DrawBindings> snapshotLifetime;
    {
        auto snapshotContext = context;
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, compute);
        auto first = resources.PrepareDrawBindings(snapshotRecorder);
        Require(first != nullptr && first->snapshots.size() == 1, "read-only draw input was not snapshotted");
        std::memset(reinterpret_cast<void*>(element), 0x22, elementBytes);
        AgcDriver::GuestMemory::MarkWritten(element, elementBytes);
        auto second = resources.PrepareDrawBindings(snapshotRecorder);
        Require(second != nullptr && second->snapshots.size() == 1, "cached draw input was not snapshotted again");
        Require(first->allocation.set != second->allocation.set, "in-flight draws share mutable descriptor bindings");
        std::memset(reinterpret_cast<void*>(element), 0x33, elementBytes);
        AgcDriver::GuestMemory::MarkWritten(element, elementBytes);
        auto downloaded = std::make_shared<Buffer>(snapshotContext, elementBytes * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = snapshotRecorder.Commands();
        RecordMemoryBarrier(snapshotContext, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const auto copy = snapshotContext.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
        Require(first->snapshots[0].Bytes().size() == elementBytes && second->snapshots[0].Bytes().size() == elementBytes, "a draw snapshot does not cover its input");
        VkBufferCopy region{first->snapshots[0].offset, 0, elementBytes};
        copy(commands, first->snapshots[0].buffer->Handle(), downloaded->Handle(), 1, &region);
        region.srcOffset = second->snapshots[0].offset;
        region.dstOffset = elementBytes;
        copy(commands, second->snapshots[0].buffer->Handle(), downloaded->Handle(), 1, &region);
        RecordMemoryBarrier(snapshotContext, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        snapshotLifetime = first;
        first.reset();
        second.reset();
        Require(!snapshotLifetime.expired(), "draw snapshot was released before its GPU batch");
        snapshotRecorder.Sync();
        const auto contents = downloaded->Bytes();
        Require(std::all_of(contents.begin(), contents.begin() + elementBytes, [](std::byte value) { return value == std::byte{0x11}; }), "first draw observed overwritten input");
        Require(std::all_of(contents.begin() + elementBytes, contents.end(), [](std::byte value) { return value == std::byte{0x22}; }), "second draw observed overwritten input");
        snapshotRecorder.NotePendingWrite(element, elementBytes);
        Require(resources.PrepareDrawBindings(snapshotRecorder) == nullptr, "GPU-produced input was replaced with stale CPU memory");
        snapshotRecorder.Sync();
    }
    Require(snapshotLifetime.expired(), "draw snapshots outlived their recorder");
    recorder.Activate();
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    // The import is retired by the next reconcile; the block itself is left to the process.
    HostImportFor(context, address, bytes);
}

bool readWrittenStagingTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: read and written staging not tested\n";
        return false;
    }
    constexpr std::size_t bytes = 8u << 20u;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the test block");
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the test block refused: read and written staging not tested\n";
        return false;
    }
    const auto boundInPlace = [&](std::uint64_t element, std::size_t elementBytes, bool read) {
        GuestBufferMemory memory(context);
        memory.AllowDeviceStaging();
        memory.AddWritable(element, elementBytes, false, read);
        memory.Upload(false);
        const auto reads = memory.InPlaceReads();
        const bool direct = reads.size() == 1 && reads[0].first == element && reads[0].second == element + elementBytes;
        Require(direct || reads.empty(), "an element was bound neither in place nor staged");
        memory.RecordCopyBacks(recorder);
        recorder.Sync();
        return direct;
    };
    Require(!boundInPlace(address + 4096, 1u << 20u, false), "a written 1 MiB element was not staged");
    Require(boundInPlace(address + 4096, 4u << 20u, false), "a written-only 4 MiB element was staged past the written window");
    Require(!boundInPlace(address + 4096, 4u << 20u, true), "a read and written 4 MiB element was bound in place");
    const auto buildInPlace = [&](std::uint64_t dispatchThreads, bool read, std::uint32_t stride, std::uint32_t extent) {
        const auto element = address + 4096;
        constexpr std::uint32_t elementBytes = 4u << 20u;
        const std::uint32_t records = stride != 0 ? elementBytes / stride : elementBytes;
        ShaderRecompiler::RecompileResult program;
        ShaderRecompiler::DescriptorBinding binding;
        binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
        binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
        binding.descriptorSet = 0;
        binding.binding = 0;
        binding.count = 1;
        binding.guestDescriptor = {static_cast<std::uint32_t>(element), (static_cast<std::uint32_t>(element >> 32u) & 0xffffu) | (stride << 16u), records, 0x31000000u};
        binding.bufferWritten = {true};
        binding.bufferRead = {read};
        binding.bufferExtent = {extent};
        program.bindings.push_back(binding);
        const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
        ShaderResources resources(context, compute, {}, false, dispatchThreads);
        const auto reads = resources.InPlaceReads();
        const bool direct = std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return range.first <= element && range.second >= element + elementBytes; });
        resources.MarkGpuWrites(recorder);
        recorder.Sync();
        return direct;
    };
    Require(!buildInPlace((4u << 20u) / 16u, true, 16, 0), "a dispatch sweeping a read and written 4 MiB element bound it in place");
    Require(buildInPlace(1024, true, 16, 0), "a dispatch touching 16 KiB of a read and written 4 MiB element staged all of it");
    Require(buildInPlace(0, true, 16, 0), "a dispatch of unknown size staged a read and written 4 MiB element");
    Require(buildInPlace((4u << 20u) / 16u, false, 16, 0), "a dispatch sweeping a written-only 4 MiB element staged it");
    Require(!buildInPlace((4u << 20u) / 32u, true, 0, 32), "a dispatch sweeping a read and written 4 MiB raw element bound it in place");
    Require(buildInPlace(1024, true, 0, 32), "a dispatch touching 32 KiB of a read and written 4 MiB raw element staged all of it");
    Require(buildInPlace((4u << 20u) / 32u, true, 0, 0), "a raw element without an access extent was staged");
    Require(std::all_of(static_cast<const std::byte*>(block), static_cast<const std::byte*>(block) + bytes, [](std::byte value) { return value == std::byte{0x11}; }), "a staged copy-back changed the guest bytes");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    }
    HostImportFor(context, address, bytes);
    return true;
}

bool sparseRunImportTests(const Device& device, bool copied) {
    const auto& context = device.GetContext();
    constexpr std::size_t unit = 65536;
    constexpr std::size_t heap = 8 * unit;
    struct Heap {
        const Context& context;
        void* area = nullptr;
        std::uint8_t* guest = nullptr;
        std::size_t mapped = 0;
        ~Heap() {
            ClearHostImports(context.device);
            {
                GuestAllocations::Mutation mutation;
                for (std::size_t index = 0; index < mapped; ++index) mutation.Remove(guest + index * unit);
            }
            SetHostImportWindow(0);
#ifdef _WIN32
            if (area != nullptr) VirtualFree(area, 0, MEM_RELEASE);
#else
            if (area != nullptr) munmap(area, heap + unit);
#endif
        }
    } state{context};
#ifdef _WIN32
    state.area = VirtualAlloc(nullptr, heap, MEM_RESERVE, PAGE_NOACCESS);
    Require(state.area != nullptr, "cannot reserve the sparse run heap");
    state.guest = static_cast<std::uint8_t*>(state.area);
#else
    void* area = mmap(nullptr, heap + unit, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(area != MAP_FAILED, "cannot reserve the sparse run heap");
    state.area = area;
    state.guest = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(area) + unit - 1) & ~std::uintptr_t{unit - 1});
#endif
    auto* guest = state.guest;
    const auto address = reinterpret_cast<std::uint64_t>(guest);
    const auto map = [&] {
        auto* piece = guest + state.mapped * unit;
#ifdef _WIN32
        Require(VirtualAlloc(piece, unit, MEM_COMMIT, PAGE_READWRITE) != nullptr, "cannot map a piece of the sparse run heap");
#else
        Require(mprotect(piece, unit, PROT_READ | PROT_WRITE) == 0, "cannot map a piece of the sparse run heap");
#endif
        for (std::size_t at = 0; at < unit; ++at) piece[at] = static_cast<std::uint8_t>((state.mapped * unit + at) * 7u + 3u);
        GuestAllocations::Mutation mutation;
        mutation.Add(piece, unit, true, true, true);
        ++state.mapped;
    };
    for (int piece = 0; piece < 3; ++piece) map();
    PrepareImportWatch(context);
    SetHostImportWindow(unit);
    if (HostImportFor(context, address, unit) == nullptr) {
        std::cout << "host import of the sparse run heap refused: sparse run imports not tested\n";
        return false;
    }
    Recorder recorder(context);
    recorder.Activate();
    const auto run = [&] { return HostImportFor(context, address, state.mapped * unit); };
    const auto bound = [&](GuestBufferMemory& memory, std::uint64_t begin, std::uint64_t bytes) {
        std::uint32_t adjustment = 0;
        const auto view = memory.Descriptor(begin, bytes, adjustment);
        const auto* import = run();
        return import != nullptr && view.buffer == import->buffer;
    };
    const auto upload = [&](bool written, std::uint64_t pastRun, const std::function<void(GuestBufferMemory&)>& check) {
        GuestBufferMemory memory(context);
        if (written) memory.AddWritable(address, heap);
        else memory.AddReadable(address, heap);
        if (pastRun != 0) memory.AddReadable(address + pastRun, 16);
        memory.Upload(true);
        check(memory);
        memory.WriteBack();
    };
    const auto committed = 3 * unit;
    upload(false, 0, [&](GuestBufferMemory& memory) {
        Require(!bound(memory, address, heap) && memory.InPlaceReads().empty(), "a sparse run heap was joined the first time its run was seen");
    });
    if (copied) {
        upload(false, 5 * unit, [&](GuestBufferMemory& memory) {
            std::uint32_t adjustment = 0;
            const auto view = memory.Descriptor(address, heap, adjustment);
            Require(!bound(memory, address, heap) && memory.InPlaceReads().empty(), "a sparse run heap with a view past its stable run is bound in place");
            Buffer readback(context, committed, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            {
                std::unique_lock gpu(GpuMutex());
                const auto commands = recorder.Commands();
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                CopyBuffer(context, commands, view.buffer, view.offset, readback.Handle(), 0, committed);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
                recorder.Sync();
            }
            readback.Invalidate();
            Require(std::memcmp(readback.Bytes().data(), guest, committed) == 0, "the GPU does not read a copied sparse run heap's mapped bytes");
        });
        recorder.Sync();
        std::cout << "sparse run import tests passed (copied)\n";
        return true;
    }
    upload(false, 0, [&](GuestBufferMemory& memory) {
        std::uint32_t adjustment = 0;
        const auto view = memory.Descriptor(address, heap, adjustment);
        const auto ranges = memory.AddressRanges();
        Require(adjustment == 0 && view.range == committed && ranges.size() == 1 && ranges[0].begin == address && ranges[0].end == address + committed, "a sparse run heap's view or address range is not cut at its last committed byte");
        const auto* import = run();
        Require(import != nullptr && view.buffer == import->buffer && view.offset == address - import->base && ranges[0].deviceAddress == import->address + (address - import->base), "a sparse run heap is not bound in place from its run's import");
        const auto direct = memory.DirectRegions();
        const auto reads = memory.InPlaceReads();
        const auto cuts = memory.CutRegions();
        Require(direct.has_value() && direct->size() == 1 && direct->front() == std::pair<std::uint64_t, std::uint64_t>{address, address + committed} && reads.size() == 1 && reads.front() == direct->front(), "a sparse run heap bound in place is not named up to its import's end for reuse");
        Require(cuts.size() == 1 && cuts.front().begin == address && cuts.front().end == address + heap && cuts.front().committedEnd == address + committed, "a sparse run heap bound in place does not report its cut");
        Buffer readback(context, committed, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        {
            std::unique_lock gpu(GpuMutex());
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, view.buffer, view.offset, readback.Handle(), 0, committed);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Sync();
        }
        readback.Invalidate();
        Require(std::memcmp(readback.Bytes().data(), guest, committed) == 0, "the GPU does not read a sparse run heap's bytes through its in-place view");
    });
    upload(true, 0, [&](GuestBufferMemory& memory) {
        Require(!bound(memory, address, heap), "a written sparse run heap is bound in place");
    });
    upload(false, 5 * unit, [&](GuestBufferMemory& memory) {
        Require(!bound(memory, address, heap), "a sparse run heap with a view past its run is bound in place");
    });
    {
        auto regionContext = context;
        DescriptorCache cache(regionContext);
        regionContext.descriptorCache = &cache;
        Recorder regionRecorder(regionContext);
        regionRecorder.Activate();
        ShaderRecompiler::RecompileResult program;
        ShaderRecompiler::DescriptorBinding binding;
        binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
        binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
        binding.descriptorSet = 0;
        binding.binding = 0;
        binding.count = 1;
        binding.guestDescriptor = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, static_cast<std::uint32_t>(heap), 0x31000000u};
        binding.bufferWritten = {false};
        program.bindings.push_back(binding);
        program.pushConstants.resize(16);
        program.memoryOffsetDword = 0;
        const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
        {
            ShaderResources resources(regionContext, compute);
            const auto reads = resources.InPlaceReads();
            Require(resources.Reusable() && reads.size() == 1 && reads.front() == std::pair<std::uint64_t, std::uint64_t>{address, address + committed}, "a set over a sparse run heap bound in place is not reusable");
            Require(resources.Revalidate(compute), "a set over an unchanged sparse run heap failed its proof");
            map();
            Require(!resources.Revalidate(compute), "a set over a sparse run heap mapped past its cut was reused");
            regionRecorder.Submit();
            device.WaitQueue();
            regionRecorder.Sync();
        }
        recorder.Activate();
        upload(false, 0, [&](GuestBufferMemory& memory) {
            Require(!bound(memory, address, heap), "a grown sparse run heap was joined again the first time its grown run was seen");
        });
        upload(false, 0, [&](GuestBufferMemory& memory) {
            std::uint32_t adjustment = 0;
            const auto view = memory.Descriptor(address, heap, adjustment);
            Require(bound(memory, address, heap) && view.range == 4 * unit, "a grown sparse run heap is not joined again over its grown run");
        });
    }
    recorder.Sync();
    std::cout << "sparse run import tests passed (in place)\n";
    return true;
}

void misalignedRegionTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    const auto alignment = context.limits.minStorageBufferOffsetAlignment;
    constexpr std::size_t bytes = 65536;
    if (alignment < 8 || alignment > 256) {
        std::cout << "storage buffer offset alignment " << alignment << ": misaligned draw regions not tested\n";
        return;
    }
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "host imports or write watching unavailable: misaligned draw regions not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    auto* guest = static_cast<std::uint8_t*>(block);
    for (std::size_t at = 0; at < bytes; ++at) guest[at] = static_cast<std::uint8_t>(at * 5u + 1u);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the watched block refused: misaligned draw regions not tested\n";
        return;
    }
    constexpr std::uint32_t offset = 4096 + 4;
    constexpr std::size_t elementBytes = 64;
    const auto view = address + offset;
    Require((view - import->base) % alignment != 0 && (view - import->base) % 4 == 0, "the test view is not a DWORD-aligned view off the storage buffer offset alignment");
    ShaderRecompiler::RecompileResult vertex;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {static_cast<std::uint32_t>(view), static_cast<std::uint32_t>(view >> 32u) & 0xffffu, static_cast<std::uint32_t>(elementBytes), 0x31000000u};
    binding.bufferWritten = {false};
    vertex.bindings.push_back(binding);
    vertex.pushConstants.resize(16);
    vertex.memoryOffsetDword = 0;
    const ShaderRecompiler::RecompileResult fragment;
    {
        auto regionContext = context;
        DescriptorCache cache(regionContext);
        regionContext.descriptorCache = &cache;
        Recorder regionRecorder(regionContext);
        regionRecorder.Activate();
        const ColorTarget target{};
        ShaderResources resources(regionContext, vertex, fragment, target, 0, 0);
        Require(resources.Reusable(), "a misaligned read-only draw region bound in place is not reusable");
        const auto reads = resources.InPlaceReads();
        Require(reads.size() == 1 && reads.front().first == view && reads.front().second == view + elementBytes, "the misaligned draw region is not read in place");
        std::array<std::byte, PipelinePushConstantBytes> push{};
        resources.PatchPushConstants(push);
        const auto adjustment = static_cast<std::uint32_t>(push[0]);
        Require(adjustment == (view - import->base) % alignment, "the misaligned region's push constant offset is not its distance from the aligned binding offset");
        const auto check = [&] {
            const auto bindings = resources.PrepareDrawBindings(regionRecorder);
            Require(bindings != nullptr && bindings->snapshots.size() == 1, "the read-only misaligned draw input was not snapshotted");
            const auto contents = bindings->snapshots.front().Bytes();
            Require(bindings->snapshots.front().address == view - adjustment && contents.size() == adjustment + elementBytes, "the draw snapshot does not start at the aligned offset below the view");
            Require(std::memcmp(contents.data() + adjustment, guest + offset, elementBytes) == 0, "the shader's patched offset into the draw snapshot misses the view's bytes");
        };
        check();
        guest[offset + 8] ^= 0xffu;
        AgcDriver::GuestMemory::MarkWritten(view, elementBytes);
        check();
        regionRecorder.Sync();
    }
    recorder.Activate();
}

void misalignedSnapshotTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    const auto alignment = context.limits.minStorageBufferOffsetAlignment;
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: misaligned draw snapshots not tested\n";
        return;
    }
    if (alignment < 8 || alignment > 256) {
        std::cout << "storage buffer offset alignment " << alignment << ": misaligned draw snapshots not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the misaligned snapshot block");
    auto* guest = static_cast<std::uint8_t*>(block);
    for (std::size_t at = 0; at < bytes; ++at) guest[at] = static_cast<std::uint8_t>(at * 7u + 3u);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the test block refused: misaligned draw snapshots not tested\n";
        return;
    }
    constexpr std::uint32_t outer = 4096;
    constexpr std::uint32_t offset = outer + 4;
    constexpr std::size_t outerBytes = 128;
    constexpr std::size_t elementBytes = 64;
    const auto words = [](std::uint64_t at, std::size_t size) { return std::array<std::uint32_t, 4>{static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(at >> 32u) & 0xffffu, static_cast<std::uint32_t>(size), 0x31000000u}; };
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 2;
    for (const auto word : words(address + outer, outerBytes)) binding.guestDescriptor.push_back(word);
    for (const auto word : words(address + offset, elementBytes)) binding.guestDescriptor.push_back(word);
    binding.bufferWritten = {false, false};
    program.bindings.push_back(binding);
    program.pushConstants.resize(16);
    program.memoryOffsetDword = 0;
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    {
        auto snapshotContext = context;
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, compute);
        std::array<std::byte, PipelinePushConstantBytes> push{};
        resources.PatchPushConstants(push);
        const auto adjustment = std::to_integer<std::uint32_t>(push[1]);
        Require(push[0] == std::byte{0} && adjustment == offset % alignment, "the inner view's shader offset is not its distance from the binding");
        const auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
        Require(bindings != nullptr && bindings->snapshots.size() == 2, "read-only draw inputs were not snapshotted");
        const auto outerContents = bindings->snapshots[0].Bytes();
        Require(outerContents.size() >= outerBytes && std::memcmp(outerContents.data(), guest + outer, outerBytes) == 0, "an aligned draw snapshot misses its view's bytes");
        const auto contents = bindings->snapshots[1].Bytes();
        Require(contents.size() >= adjustment + elementBytes, "a draw snapshot ends before the bytes the shader reads");
        Require(std::memcmp(contents.data() + adjustment, guest + offset, elementBytes) == 0, "the shader's offset into a draw snapshot misses the view's bytes");
        snapshotRecorder.Sync();
    }
    recorder.Activate();
}

class DescriptorUpdates {
public:
    struct Contents {
        std::map<std::pair<std::uint32_t, std::uint32_t>, VkDescriptorBufferInfo> buffers;
        std::map<std::pair<std::uint32_t, std::uint32_t>, VkDescriptorImageInfo> images;
        std::map<std::pair<VkDescriptorSet, std::uint32_t>, std::uint32_t> written;
    };
    explicit DescriptorUpdates(Context& context) : context(context), previous(context.functions),
        update(context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")) {
        Require(active == nullptr, "nested descriptor update capture");
        if (previous != nullptr) functions = *previous;
        functions.updateDescriptorSets = captureUpdate;
        context.functions = &functions;
        active = this;
    }
    ~DescriptorUpdates() { context.functions = previous; active = nullptr; }
    std::vector<Contents> calls;
    std::uint32_t copies = 0;

private:
    static VKAPI_ATTR void VKAPI_CALL captureUpdate(VkDevice device, std::uint32_t count, const VkWriteDescriptorSet* writes, std::uint32_t copyCount, const VkCopyDescriptorSet* copies) {
        Contents contents;
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& write = writes[i];
            contents.written[{write.dstSet, write.dstBinding}] += write.descriptorCount;
            for (std::uint32_t element = 0; element < write.descriptorCount; ++element) {
                const auto key = std::pair{write.dstBinding, write.dstArrayElement + element};
                if (write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) contents.buffers[key] = write.pBufferInfo[element];
                else contents.images[key] = write.pImageInfo[element];
            }
        }
        active->calls.push_back(std::move(contents));
        active->copies += copyCount;
        active->update(device, count, writes, copyCount, copies);
    }
    Context& context;
    const DeviceFunctions* previous;
    DeviceFunctions functions{};
    PFN_vkUpdateDescriptorSets update;
    static inline DescriptorUpdates* active = nullptr;
};

void drawDescriptorWriteTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    DescriptorUpdates updates(context);
    {
        DescriptorCache cache(context);
        context.descriptorCache = &cache;
        Recorder snapshots(context);
        snapshots.Activate();
        ShaderRecompiler::RecompileResult program;
        ShaderRecompiler::DescriptorBinding data{};
        data.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
        data.role = ShaderRecompiler::DescriptorRole::FlattenedSrt;
        data.binding = 2;
        data.count = 1;
        data.guestDescriptor = {1, 2, 3, 4};
        program.bindings.push_back(data);
        data.binding = 7;
        program.bindings.push_back(data);
        ShaderRecompiler::DescriptorBinding sampler{};
        sampler.kind = ShaderRecompiler::DescriptorKind::Sampler;
        sampler.role = ShaderRecompiler::DescriptorRole::GuestSamplers;
        sampler.binding = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers);
        sampler.count = 2;
        sampler.guestDescriptor.resize(8);
        sampler.samplerDepthCompare = {false, false};
        program.bindings.push_back(sampler);
        const CompiledShader shader{ShaderRecompiler::ShaderStage::Compute, &program, 0};
        ShaderResources resources(context, shader);
        Require(updates.calls.size() == 1, "building resources did not write one descriptor set");
        const auto original = updates.calls.front();
        Require(original.buffers.size() == 2 && original.images.size() == 2, "descriptor test lacks mixed bindings and an image array");
        for (std::size_t allocation = 0; allocation < 2; ++allocation) {
            const std::array<std::uint32_t, 4> words{5, 6, 7, 8};
            const ShaderResources::MovedBuffer moved{allocation, 0, 16, words};
            const auto draw = resources.PrepareDrawBindings(snapshots, std::span(&moved, 1));
            Require(draw != nullptr && draw->snapshots.size() == 1, "moved shader data did not create a draw snapshot");
            Require(updates.calls.size() == allocation + 2 && updates.copies == 0, "draw descriptors copied GPU memory or needed multiple updates");
            const auto& actual = updates.calls.back();
            Require(actual.buffers.size() == original.buffers.size() && actual.images.size() == original.images.size(), "snapshot update omitted unchanged descriptors");
            for (const auto& [key, before] : original.buffers) {
                const auto after = actual.buffers.at(key);
                const bool snapshotted = key.first == (allocation == 0 ? 2u : 7u);
                const auto& snapshot = draw->snapshots.front();
                const auto expected = snapshotted ? snapshot.buffer->Handle() : before.buffer;
                const auto expectedOffset = snapshotted ? snapshot.offset : before.offset;
                Require(after.buffer == expected && after.offset == expectedOffset && after.range == before.range, "snapshot update replaced the wrong buffer or changed its range");
            }
            for (const auto& [key, before] : original.images) {
                const auto after = actual.images.at(key);
                Require(after.sampler == before.sampler && after.imageView == before.imageView && after.imageLayout == before.imageLayout, "snapshot update changed an image descriptor");
            }
        }
        snapshots.Sync();
    }
    recorder.Activate();
}

void drawSnapshotTableSetTests(const Device& device, Recorder& recorder) {
    namespace Abi = ShaderRecompiler::ImageTableAbi;
    using ShaderRecompiler::DescriptorKind;
    using ShaderRecompiler::DescriptorRole;
    const auto& base = device.GetContext();
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    if (context.imageTableCapacity < 2u) {
        std::cout << "image tables unavailable: draw snapshot table sets not tested\n";
        return;
    }
    constexpr std::uint32_t width = 4;
    constexpr std::uint32_t height = 4;
    alignas(256) static std::array<float, width * height> texels{};
    texels.fill(1.0f);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels.data()));
    const std::array<std::uint32_t, 8> texture{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (22u << 20u) | (((width - 1u) & 3u) << 30u), ((width - 1u) >> 2u) | ((height - 1u) << 14u), 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
    constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    const auto addTable = [&](ShaderRecompiler::RecompileResult& program, ShaderRecompiler::RuntimeAbi::Stage stage, std::uint32_t set, std::uint32_t elements) {
        ShaderRecompiler::DescriptorBinding map{};
        map.kind = DescriptorKind::StorageBuffer;
        map.role = DescriptorRole::ImageTableMap;
        map.descriptorSet = set;
        map.binding = static_cast<std::uint32_t>(stage) * bindingCount + static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::ImageTableMap);
        map.count = 1;
        map.guestDescriptor.assign(Abi::HeaderWords + elements, 0u);
        map.guestDescriptor[Abi::HeaderVersion] = Abi::Version;
        map.guestDescriptor[Abi::HeaderElementCount] = elements;
        map.guestDescriptor[Abi::HeaderElementFaults] = Abi::HeaderWords;
        map.guestDescriptor[Abi::HeaderSamplerFaults] = Abi::HeaderWords + elements;
        program.bindings.push_back(map);
        ShaderRecompiler::DescriptorBinding images{};
        images.kind = DescriptorKind::SampledImage;
        images.role = DescriptorRole::ImageTable;
        images.descriptorSet = set;
        images.binding = ShaderRecompiler::RuntimeAbi::ImageTableBinding;
        images.count = elements;
        for (std::uint32_t element = 0; element < elements; ++element) images.guestDescriptor.insert(images.guestDescriptor.end(), texture.begin(), texture.end());
        images.imageShapes.assign(elements, ShaderRecompiler::DescriptorImageShape::Image2D);
        images.imageDepthCompare.assign(elements, false);
        images.imageSamplers.assign(elements, 0u);
        program.bindings.push_back(std::move(images));
    };
    DescriptorUpdates updates(context);
    {
        DescriptorCache cache(context);
        context.descriptorCache = &cache;
        Recorder snapshots(context);
        snapshots.Activate();
        ShaderRecompiler::RecompileResult vertex, fragment;
        ShaderRecompiler::DescriptorBinding data{};
        data.kind = DescriptorKind::StorageBuffer;
        data.role = DescriptorRole::FlattenedSrt;
        data.binding = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::FlattenedSrt);
        data.count = 1;
        data.guestDescriptor = {1, 2, 3, 4};
        vertex.bindings.push_back(data);
        addTable(vertex, ShaderRecompiler::RuntimeAbi::Stage::Main, 0u, 1u);
        addTable(fragment, ShaderRecompiler::RuntimeAbi::Stage::Fragment, ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet, 2u);
        const ColorTarget target{};
        ShaderResources resources(context, vertex, fragment, target, 0, 0);
        Require(resources.SplitSets(), "a fragment image table did not get its own descriptor set");
        const std::array<std::uint32_t, 4> movedWords{5, 6, 7, 8};
        const ShaderResources::MovedBuffer moved{0, 0, 16, movedWords};
        const auto draw = resources.PrepareDrawBindings(snapshots, std::span(&moved, 1));
        Require(draw != nullptr && draw->snapshots.size() == 1, "moved shader data did not create a draw snapshot");
        const auto& written = updates.calls.back().written;
        const auto tableWrites = [&](VkDescriptorSet set) {
            const auto it = written.find({set, ShaderRecompiler::RuntimeAbi::ImageTableBinding});
            return it == written.end() ? 0u : it->second;
        };
        Require(tableWrites(draw->allocation.set) == 1u, "the draw snapshot wrote the wrong number of image table descriptors into the vertex set");
        Require(tableWrites(draw->fragmentAllocation.set) == 2u, "the draw snapshot wrote the wrong number of image table descriptors into the fragment set");
        snapshots.Sync();
    }
    ClearCachedTextures(context.device);
    recorder.Activate();
}

bool drawSnapshotArenaTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    constexpr std::size_t bytes = 2 * 65536;
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "skipped, host imports unavailable: draw snapshot arena not tested\n";
        return false;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "skipped, host import of the test block refused: draw snapshot arena not tested\n";
        return false;
    }
    constexpr std::array<std::size_t, 2> elementBytes{100, SnapshotArena::InputLimit};
    const std::array<std::uint64_t, 2> elements{address + 4096, address + 65536};
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 2;
    for (std::size_t index = 0; index < elements.size(); ++index) {
        const auto element = elements[index];
        for (const auto word : {static_cast<std::uint32_t>(element), static_cast<std::uint32_t>(element >> 32u) & 0xffffu, static_cast<std::uint32_t>(elementBytes[index]), 0x31000000u}) binding.guestDescriptor.push_back(word);
    }
    binding.bufferWritten = {false, false};
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    {
        auto snapshotContext = context;
        DescriptorUpdates updates(snapshotContext);
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, compute);
        const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 1);
        std::vector<std::shared_ptr<ShaderResources::DrawBindings>> held;
                const auto holds = [&](const ShaderResources::DrawBindings& bindings, std::byte expected) {
            return std::all_of(bindings.snapshots.begin(), bindings.snapshots.end(), [&](const auto& snapshot) {
                const auto contents = snapshot.Bytes();
                return std::all_of(contents.begin(), contents.end(), [&](std::byte value) { return value == expected; });
            });
        };
        {
            const std::array<std::byte, 2> marks{std::byte{0x01}, std::byte{0x02}};
            for (std::size_t index = 0; index < elements.size(); ++index) std::memset(reinterpret_cast<void*>(elements[index]), std::to_integer<int>(marks[index]), elementBytes[index]);
            AgcDriver::GuestMemory::BumpCollectEpoch();
            const auto calls = updates.calls.size();
            const auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
            Require(bindings != nullptr && bindings->snapshots.size() == 2, "draw inputs for the descriptor check were not snapshotted");
            Require(updates.calls.size() == calls + 1, "a draw did not write exactly one descriptor update");
            Require(bindings->snapshots[0].buffer == bindings->snapshots[1].buffer && bindings->snapshots[1].offset >= bindings->snapshots[0].offset + elementBytes[0] && bindings->snapshots[1].offset != 0, "the descriptor check needs two slices of one chunk, the second at a nonzero offset");
            for (std::size_t index = 0; index < elements.size(); ++index) {
                const auto& snapshot = bindings->snapshots[index];
                const auto& descriptor = updates.calls.back().buffers.at({0, static_cast<std::uint32_t>(index)});
                Require(descriptor.buffer == snapshot.buffer->Handle() && descriptor.offset == snapshot.offset && descriptor.range == elementBytes[index], "a draw descriptor does not name the buffer, offset and range of its slice");
                const auto visible = snapshot.buffer->Bytes().subspan(static_cast<std::size_t>(descriptor.offset), static_cast<std::size_t>(descriptor.range));
                Require(std::all_of(visible.begin(), visible.end(), [&](std::byte value) { return value == marks[index]; }), "reading through a draw descriptor did not return the bytes of its own slice");
            }
        }
        const auto keptBefore = snapshotRecorder.OpenKeptBytes();
        const std::size_t draws = SnapshotArena::ChunkBytes / elementBytes[1] + 8;
        for (std::size_t draw = 0; draw < draws; ++draw) {
            const auto value = static_cast<std::byte>(0x20 + draw % 64);
            for (std::size_t index = 0; index < elements.size(); ++index) std::memset(reinterpret_cast<void*>(elements[index]), std::to_integer<int>(value), elementBytes[index]);
            AgcDriver::GuestMemory::BumpCollectEpoch();
            auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
            Require(bindings != nullptr && bindings->snapshots.size() == 2, "small read-only draw inputs were not snapshotted");
            for (std::size_t index = 0; index < elements.size(); ++index) {
                const auto& snapshot = bindings->snapshots[index];
                Require(snapshot.address == elements[index] && snapshot.bytes == elementBytes[index] && snapshot.offset % alignment == 0, "a draw snapshot slice is misplaced or misaligned");
                Require(snapshot.offset + snapshot.bytes <= snapshot.buffer->Bytes().size(), "a draw snapshot slice ends past its chunk");
            }
            Require(holds(*bindings, value), "a draw snapshot slice does not hold the guest bytes of its draw");
            held.push_back(std::move(bindings));
        }
        Require(snapshotRecorder.OpenKeptBytes() - keptBefore >= draws * (elementBytes[0] + elementBytes[1]), "draw snapshot slices were not counted as kept bytes");
        std::vector<std::tuple<const Buffer*, VkDeviceSize, std::size_t>> ranges;
        for (const auto& bindings : held) {
            for (const auto& snapshot : bindings->snapshots) ranges.emplace_back(snapshot.buffer.get(), snapshot.offset, snapshot.bytes);
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t index = 1; index < ranges.size(); ++index) {
            const auto& [previousChunk, previousOffset, previousBytes] = ranges[index - 1];
            const auto& [chunk, offset, sliceBytes] = ranges[index];
            Require(previousChunk != chunk || previousOffset + previousBytes <= offset, "two draw snapshot slices share bytes of one chunk");
        }
        Require(std::get<0>(ranges.front()) != std::get<0>(ranges.back()), "the draw snapshot arena never moved to a new chunk");
        for (std::size_t draw = 0; draw < held.size(); ++draw) Require(holds(*held[draw], static_cast<std::byte>(0x20 + draw % 64)), "a later draw snapshot overwrote an earlier draw's slice");
        const auto keptBeforeSmall = snapshotRecorder.OpenKeptBytes();
        const auto firstSmall = snapshotRecorder.AllocateSnapshot(4);
        auto lastSmall = firstSmall;
        std::size_t smallSlices = 1;
        while (lastSmall.chunk == firstSmall.chunk) {
            lastSmall = snapshotRecorder.AllocateSnapshot(4);
            ++smallSlices;
        }
        const auto chargedSmall = snapshotRecorder.OpenKeptBytes() - keptBeforeSmall;
        Require(chargedSmall >= SnapshotArena::ChunkBytes - static_cast<std::size_t>(firstSmall.offset), "a chunk filled by " + std::to_string(smallSlices) + " small draw snapshot slices was charged " + std::to_string(chargedSmall) + " kept bytes, less than the chunk it retains");
        Require(chargedSmall <= 2 * SnapshotArena::ChunkBytes, "small draw snapshot slices were charged more than the chunk space they consumed");
        const auto firstBatchRanges = ranges;
        snapshotRecorder.Submit();
        std::vector<std::shared_ptr<ShaderResources::DrawBindings>> second;
        constexpr std::size_t secondDraws = 16;
        for (std::size_t draw = 0; draw < secondDraws; ++draw) {
            const auto value = static_cast<std::byte>(0xa0 + draw);
            for (std::size_t index = 0; index < elements.size(); ++index) std::memset(reinterpret_cast<void*>(elements[index]), std::to_integer<int>(value), elementBytes[index]);
            AgcDriver::GuestMemory::BumpCollectEpoch();
            auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
            Require(bindings != nullptr && bindings->snapshots.size() == 2, "draw inputs after a submit were not snapshotted");
            Require(holds(*bindings, value), "a draw snapshot slice after a submit does not hold the guest bytes of its draw");
            second.push_back(std::move(bindings));
        }
        for (std::size_t draw = 0; draw < held.size(); ++draw) Require(holds(*held[draw], static_cast<std::byte>(0x20 + draw % 64)), "a draw snapshot slice of the submitted batch changed while the next batch allocated");
        std::vector<std::tuple<const Buffer*, VkDeviceSize, std::size_t>> secondRanges;
        for (const auto& bindings : second) {
            for (const auto& snapshot : bindings->snapshots) secondRanges.emplace_back(snapshot.buffer.get(), snapshot.offset, snapshot.bytes);
        }
        for (const auto& [chunk, offset, sliceBytes] : secondRanges) {
            for (const auto& [firstChunkPointer, firstOffset, firstBytes] : firstBatchRanges) {
                Require(chunk != firstChunkPointer || offset + sliceBytes <= firstOffset || firstOffset + firstBytes <= offset, "a draw snapshot slice allocated after a submit overlaps a slice of the submitted batch");
            }
        }
        const auto sameSlice = [](const ShaderResources::DrawBindings::Snapshot& a, const ShaderResources::DrawBindings::Snapshot& b) { return a.buffer == b.buffer && a.offset == b.offset; };
        const auto keptBeforeRepeat = snapshotRecorder.OpenKeptBytes();
        const auto repeated = resources.PrepareDrawBindings(snapshotRecorder);
        const auto again = resources.PrepareDrawBindings(snapshotRecorder);
        Require(repeated != nullptr && again != nullptr && again->snapshots.size() == 2, "unchanged small draw inputs were not snapshotted");
        if (AgcDriver::GuestMemory::CollectEpoch() != 0) {
            Require(sameSlice(again->snapshots[0], repeated->snapshots[0]) && sameSlice(again->snapshots[1], repeated->snapshots[1]), "a draw in the same collect epoch copied its unchanged small inputs again");
            Require(snapshotRecorder.OpenKeptBytes() == keptBeforeRepeat, "draw snapshot slices reused in the batch that allocated them were charged again");
        } else if (!AgcDriver::GuestMemory::Watched(elements[0], elementBytes[0]) || !AgcDriver::GuestMemory::Watched(elements[1], elementBytes[1])) {
            Require(!sameSlice(again->snapshots[0], repeated->snapshots[0]) && !sameSlice(again->snapshots[1], repeated->snapshots[1]), "a draw reused a small input's slice without a collect epoch or a write watch to bound it");
        } else {
            Require(sameSlice(again->snapshots[0], repeated->snapshots[0]) && sameSlice(again->snapshots[1], repeated->snapshots[1]), "a draw without a collect epoch copied its unchanged, write-watched small inputs again");
        }
        AgcDriver::GuestMemory::BumpCollectEpoch();
        const auto nextEpoch = resources.PrepareDrawBindings(snapshotRecorder);
        Require(nextEpoch != nullptr && nextEpoch->snapshots.size() == 2, "unchanged small inputs were not snapshotted in a new collect epoch");
        if (AgcDriver::GuestMemory::Watched(elements[0], elementBytes[0]) && AgcDriver::GuestMemory::Watched(elements[1], elementBytes[1])) {
            Require(sameSlice(nextEpoch->snapshots[0], repeated->snapshots[0]) && sameSlice(nextEpoch->snapshots[1], repeated->snapshots[1]), "a draw in a new collect epoch copied its unchanged, write-watched small inputs again");
        }
        std::memset(reinterpret_cast<void*>(elements[0]), 0x5a, elementBytes[0]);
        AgcDriver::GuestMemory::BumpCollectEpoch();
        const auto changed = resources.PrepareDrawBindings(snapshotRecorder);
        Require(changed != nullptr && !sameSlice(changed->snapshots[0], repeated->snapshots[0]) && std::all_of(changed->snapshots[0].Bytes().begin(), changed->snapshots[0].Bytes().end(), [](std::byte value) { return value == std::byte{0x5a}; }), "a draw after a store and a new collect epoch reused the slice taken before the store");
        std::weak_ptr<Buffer> firstChunk = held.front()->snapshots.front().buffer;
        held.clear();
        second.clear();
        Require(!firstChunk.expired(), "a draw snapshot chunk was released before its batch completed");
        snapshotRecorder.Sync();
        snapshotRecorder.FlushDeferredReleases();
        const auto reclaimDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!firstChunk.expired() && std::chrono::steady_clock::now() < reclaimDeadline) std::this_thread::yield();
        Require(firstChunk.expired(), "a draw snapshot chunk was not reclaimed once its batch completed");
    }
    recorder.Activate();
    return true;
}

void imageTableFirstLayerRevalidateTests(const Device& device, Recorder& recorder) {
    namespace Abi = ShaderRecompiler::ImageTableAbi;
    using ShaderRecompiler::DescriptorKind;
    using ShaderRecompiler::DescriptorRole;
    const auto& base = device.GetContext();
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    if (context.imageTableCapacity < 1u) {
        std::cout << "image tables unavailable: image table first-layer revalidation not tested\n";
        return;
    }
    constexpr std::uint32_t width = 4;
    constexpr std::uint32_t height = 4;
    constexpr std::uint32_t lastSlice = 3;
    alignas(256) static std::array<float, width * height * (lastSlice + 1u) * 16u> texels{};
    texels.fill(1.0f);
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels.data()));
    const std::array<std::uint32_t, 8> texture{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (22u << 20u) | (((width - 1u) & 3u) << 30u), ((width - 1u) >> 2u) | ((height - 1u) << 14u), 0xfacu | (13u << 28u), lastSlice, 0u, 0u, 0u};
    const auto whole = DecodeTextureResource(texture);
    Require(whole.dimension == TextureDimension::k2DArray && whole.depthOrLastArray == lastSlice && DescribeSurface(whole).guestBytes <= sizeof(texels), "the image table test T# is not a 2D array over its texels");
    constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding map{};
    map.kind = DescriptorKind::StorageBuffer;
    map.role = DescriptorRole::ImageTableMap;
    map.binding = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Stage::Main) * bindingCount + static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::ImageTableMap);
    map.count = 1;
    map.guestDescriptor.assign(Abi::HeaderWords + 1u, 0u);
    map.guestDescriptor[Abi::HeaderVersion] = Abi::Version;
    map.guestDescriptor[Abi::HeaderElementCount] = 1u;
    map.guestDescriptor[Abi::HeaderElementFaults] = Abi::HeaderWords;
    map.guestDescriptor[Abi::HeaderSamplerFaults] = Abi::HeaderWords + 1u;
    program.bindings.push_back(map);
    ShaderRecompiler::DescriptorBinding images{};
    images.kind = DescriptorKind::SampledImage;
    images.role = DescriptorRole::ImageTable;
    images.binding = ShaderRecompiler::RuntimeAbi::ImageTableBinding;
    images.count = 1;
    images.guestDescriptor.assign(texture.begin(), texture.end());
    images.imageShapes.assign(1, ShaderRecompiler::DescriptorImageShape::Image2D);
    images.imageDepthCompare.assign(1, false);
    images.imageSamplers.assign(1, 0u);
    program.bindings.push_back(std::move(images));
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    {
        ShaderResources resources(context, compute);
        Require(resources.SampledTextures().size() == 1, "a 2D read of a 2D array in an image table bound no texture");
        const auto bound = resources.SampledTextures().front();
        ShaderResources::ProofReport report;
        Require(resources.Revalidate(std::span(&compute, 1), &report), "an image table holding a 2D read of a 2D array did not revalidate");
        Require(report.path == ShaderResources::ProofPath::Full, "an image table holding a 2D read of a 2D array was not proven on the full path");
        Require(resources.SampledTextures().front() == bound, "revalidating an image table holding a 2D read of a 2D array changed its texture");
        Require(resources.Revalidate(std::span(&compute, 1)), "an image table holding a 2D read of a 2D array did not revalidate twice");
        recorder.Sync();
    }
    ClearCachedTextures(context.device);
}

void drawSnapshotReuseTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    constexpr std::size_t bytes = 65536;
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "host imports or write watching unavailable: draw snapshot reuse not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the watched block refused: draw snapshot reuse not tested\n";
        return;
    }
    if (!AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "host imports are compared, not watched: draw snapshot reuse not tested\n";
        return;
    }
    const auto element = address + 4096;
    constexpr std::size_t elementBytes = 8192;
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {static_cast<std::uint32_t>(element), static_cast<std::uint32_t>(element >> 32u) & 0xffffu, elementBytes, 0x31000000u};
    binding.bufferWritten = {false};
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    {
        auto snapshotContext = context;
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, compute);
        const auto snapshot = [&](std::byte expected) {
            const auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
            Require(bindings != nullptr && bindings->snapshots.size() == 1, "read-only draw input was not snapshotted");
            const auto buffer = bindings->snapshots[0].buffer;
            const auto contents = buffer->Bytes();
            Require(contents.size() == elementBytes && std::all_of(contents.begin(), contents.end(), [&](std::byte value) { return value == expected; }), "a draw snapshot does not hold the guest bytes of its draw");
            return buffer;
        };
        const auto first = snapshot(std::byte{0x11});
        Require(snapshot(std::byte{0x11}) == first, "an unchanged draw input was copied again");
        std::memset(reinterpret_cast<void*>(element), 0x22, elementBytes);
        const auto afterCpu = snapshot(std::byte{0x22});
        Require(afterCpu != first, "a draw snapshot outlived a CPU store to its range");
        Require(snapshot(std::byte{0x22}) == afterCpu, "the recopied draw input was not kept");
        std::memset(reinterpret_cast<void*>(element), 0x33, elementBytes);
        AgcDriver::GuestMemory::MarkWritten(element, 4);
        const auto afterStore = snapshot(std::byte{0x33});
        Require(afterStore != afterCpu, "a draw snapshot outlived a driver store to its range");
        std::memset(reinterpret_cast<void*>(element), 0x33, elementBytes);
        Require(snapshot(std::byte{0x33}) == afterStore, "a CPU store of the same bytes made the draw input copy again");
        AgcDriver::GuestMemory::MarkWritten(element, 4);
        Require(snapshot(std::byte{0x33}) == afterStore, "a driver store of the same bytes made the draw input copy again");
        reinterpret_cast<std::uint8_t*>(element)[elementBytes - 1] = 0x34;
        const auto tail = resources.PrepareDrawBindings(snapshotRecorder);
        Require(tail != nullptr && tail->snapshots.size() == 1 && tail->snapshots[0].buffer != afterStore && tail->snapshots[0].buffer->Bytes()[elementBytes - 1] == std::byte{0x34}, "a draw snapshot outlived a store that changed its last byte");
        std::memset(reinterpret_cast<void*>(element), 0x33, elementBytes);
        {
            GuestAllocations::Mutation mutation;
        }
        Require(snapshot(std::byte{0x33}) != afterStore, "a draw snapshot outlived a registry mutation");
        snapshotRecorder.Sync();
    }
    recorder.Activate();
}

void drawSnapshotPatchTests(const Device& device) {
    const auto& context = device.GetContext();
    constexpr std::size_t bytes = 4 * 65536;
    std::unique_lock gpu(GpuMutex());
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "host imports or write watching unavailable: snapshot patching not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    auto* guest = static_cast<std::uint8_t*>(block);
    for (std::size_t at = 0; at < bytes; ++at) guest[at] = static_cast<std::uint8_t>(at * 13u + 5u);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the watched block refused: snapshot patching not tested\n";
        return;
    }
    if (!AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "host imports are compared, not watched: snapshot patching not tested\n";
        return;
    }
    constexpr std::size_t elementOffset = 4096;
    constexpr std::size_t elementBytes = 3 * 65536;
    const auto element = address + elementOffset;
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {static_cast<std::uint32_t>(element), static_cast<std::uint32_t>(element >> 32u) & 0xffffu, static_cast<std::uint32_t>(elementBytes), 0x31000000u};
    binding.bufferWritten = {false};
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    auto snapshotContext = context;
    DescriptorCache cache(snapshotContext);
    snapshotContext.descriptorCache = &cache;
    Recorder snapshotRecorder(snapshotContext);
    snapshotRecorder.Activate();
    {
        ShaderResources resources(snapshotContext, compute);
        const auto snapshot = [&] {
            auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
            Require(bindings != nullptr && bindings->snapshots.size() == 1, "read-only draw input was not snapshotted");
            auto buffer = bindings->snapshots[0].buffer;
            const auto contents = buffer->Bytes();
            Require(contents.size() == elementBytes && std::memcmp(contents.data(), reinterpret_cast<const void*>(element), elementBytes) == 0, "a draw snapshot does not hold the guest bytes of its draw");
            return buffer;
        };
        const auto settle = [&](const std::weak_ptr<Buffer>& buffer) {
            snapshotRecorder.Sync();
            gpu.unlock();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (buffer.use_count() > 1 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            gpu.lock();
            Require(buffer.use_count() == 1, "a finished draw still holds its snapshot");
        };
        std::weak_ptr<Buffer> first = snapshot();
        settle(first);
        guest[elementOffset + 10] ^= 0xffu;
        guest[elementOffset + 65536 + 300] ^= 0xffu;
        guest[elementOffset + 2 * 65536 + 7] = guest[elementOffset + 2 * 65536 + 7];
        guest[elementOffset + elementBytes - 1] ^= 0x5au;
        AgcDriver::GuestMemory::BumpCollectEpoch();
        {
            const auto patched = snapshot();
            Require(patched == first.lock(), "a stale snapshot nothing else holds was copied whole instead of patched");
        }
        settle(first);
        std::memset(guest + elementOffset + 65536, 0x77, 65536);
        AgcDriver::GuestMemory::BumpCollectEpoch();
        {
            const auto patched = snapshot();
            Require(patched == first.lock(), "a snapshot whose whole middle block changed was not patched");
        }
        settle(first);
        const auto held = snapshot();
        Require(held == first.lock(), "an unchanged snapshot was not reused");
        const std::vector<std::byte> before(held->Bytes().begin(), held->Bytes().end());
        guest[elementOffset + 2 * 65536 + 100] ^= 0xffu;
        AgcDriver::GuestMemory::BumpCollectEpoch();
        const auto fresh = snapshot();
        Require(fresh != held, "a snapshot a recorded draw still holds was patched");
        Require(std::memcmp(held->Bytes().data(), before.data(), before.size()) == 0, "a held snapshot's bytes changed");
        snapshotRecorder.Sync();
    }
}

void drawSnapshotEpochTests(const Device& device) {
    const auto& context = device.GetContext();
    constexpr std::size_t bytes = 4 * 65536;
    std::unique_lock gpu(GpuMutex());
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "host imports unavailable: draw snapshot reuse within a collect epoch not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    auto* guest = static_cast<std::uint8_t*>(block);
    for (std::size_t at = 0; at < bytes; ++at) guest[at] = static_cast<std::uint8_t>(at * 7u + 3u);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the block refused: draw snapshot reuse within a collect epoch not tested\n";
        return;
    }
    constexpr std::size_t elementOffset = 4096;
    constexpr std::size_t elementBytes = 2 * 65536;
    const auto element = address + elementOffset;
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {static_cast<std::uint32_t>(element), static_cast<std::uint32_t>(element >> 32u) & 0xffffu, static_cast<std::uint32_t>(elementBytes), 0x31000000u};
    binding.bufferWritten = {false};
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    auto snapshotContext = context;
    DescriptorCache cache(snapshotContext);
    snapshotContext.descriptorCache = &cache;
    Recorder snapshotRecorder(snapshotContext);
    snapshotRecorder.Activate();
    {
        ShaderResources resources(snapshotContext, compute);
        const auto snapshot = [&] {
            auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
            Require(bindings != nullptr && bindings->snapshots.size() == 1, "read-only draw input was not snapshotted");
            auto buffer = bindings->snapshots[0].buffer;
            const auto contents = buffer->Bytes();
            Require(contents.size() == elementBytes && std::memcmp(contents.data(), reinterpret_cast<const void*>(element), elementBytes) == 0, "a draw snapshot does not hold the guest bytes of its draw");
            return buffer;
        };
        AgcDriver::GuestMemory::BumpCollectEpoch();
        const auto first = snapshot();
        Require(snapshot() == first, "a second draw in the same collect epoch copied its unchanged input again");
        guest[elementOffset + 65536 + 17] ^= 0xffu;
        AgcDriver::GuestMemory::BumpCollectEpoch();
        const auto changed = snapshot();
        Require(changed != first, "a draw after a store and a new collect epoch reused the snapshot taken before the store");
        Require(snapshot() == changed, "a second draw after the store copied its input again");
        snapshotRecorder.Sync();
    }
    std::cout << "draw snapshot reuse within a collect epoch: ok\n";
}

void drawSnapshotCapTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    const char* previous = std::getenv("APS5_DRAW_SNAPSHOT_MAX_MIB");
    const std::string saved = previous != nullptr ? previous : "";
#ifdef _WIN32
    Require(_putenv_s("APS5_DRAW_SNAPSHOT_MAX_MIB", "1") == 0, "cannot set snapshot cap");
#else
    Require(setenv("APS5_DRAW_SNAPSHOT_MAX_MIB", "1", 1) == 0, "cannot set snapshot cap");
#endif
    struct RestoreCap {
        const char* previous;
        std::string saved;
        ~RestoreCap() {
#ifdef _WIN32
            _putenv_s("APS5_DRAW_SNAPSHOT_MAX_MIB", saved.c_str());
#else
            if (previous != nullptr) {
                setenv("APS5_DRAW_SNAPSHOT_MAX_MIB", saved.c_str(), 1);
            } else {
                unsetenv("APS5_DRAW_SNAPSHOT_MAX_MIB");
            }
#endif
        }
    } restore{previous, saved};
    constexpr std::size_t bytes = 3 * 1024 * 1024;
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "host imports or write watching unavailable: draw snapshot cap not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the watched block refused: draw snapshot cap not tested\n";
        return;
    }
    const auto smallAddress = address + 4096;
    constexpr std::size_t smallBytes = 1024 * 1024 - 4096;
    const auto largeAddress = address + 1024 * 1024 + 8192;
    constexpr std::size_t largeBytes = 1024 * 1024 + 4096;
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding smallBinding;
    smallBinding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    smallBinding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    smallBinding.descriptorSet = 0;
    smallBinding.binding = 0;
    smallBinding.count = 1;
    smallBinding.guestDescriptor = {static_cast<std::uint32_t>(smallAddress), static_cast<std::uint32_t>(smallAddress >> 32u) & 0xffffu, static_cast<std::uint32_t>(smallBytes), 0x31000000u};
    smallBinding.bufferWritten = {false};
    program.bindings.push_back(smallBinding);
    ShaderRecompiler::DescriptorBinding largeBinding;
    largeBinding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    largeBinding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    largeBinding.descriptorSet = 0;
    largeBinding.binding = 1;
    largeBinding.count = 1;
    largeBinding.guestDescriptor = {static_cast<std::uint32_t>(largeAddress), static_cast<std::uint32_t>(largeAddress >> 32u) & 0xffffu, static_cast<std::uint32_t>(largeBytes), 0x31000000u};
    largeBinding.bufferWritten = {false};
    program.bindings.push_back(largeBinding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    {
        auto snapshotContext = context;
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, compute);
        const auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
        Require(bindings != nullptr, "read-only draw bindings were not prepared");
        Require(bindings->snapshots.size() == 1, "an input exceeding the draw snapshot cap was snapshotted");
        Require(bindings->snapshots[0].address == smallAddress, "the snapshotted input does not match the smaller buffer below the cap");
        Require(snapshotRecorder.ReusableDrawSnapshot(largeAddress, largeBytes) == nullptr, "an input exceeding the draw snapshot cap was kept in the snapshot cache");
        if (Watched(address, bytes)) {
            Require(snapshotRecorder.ReusableDrawSnapshot(smallAddress, smallBytes) != nullptr, "an input below the draw snapshot cap was not kept in the snapshot cache");
        }
        const auto contents = bindings->snapshots[0].buffer->Bytes();
        Require(contents.size() == smallBytes && std::all_of(contents.begin(), contents.end(), [](std::byte value) { return value == std::byte{0x11}; }), "a draw snapshot does not hold the guest bytes of its draw");
        std::memset(reinterpret_cast<void*>(smallAddress), 0x22, smallBytes);
        std::memset(reinterpret_cast<void*>(largeAddress), 0x33, largeBytes);
        Require(std::all_of(contents.begin(), contents.end(), [](std::byte value) { return value == std::byte{0x11}; }), "snapshotted input observed CPU overwrite");
        const auto* inPlace = reinterpret_cast<const std::byte*>(largeAddress);
        Require(inPlace[0] == std::byte{0x33} && std::all_of(inPlace, inPlace + largeBytes, [](std::byte value) { return value == std::byte{0x33}; }), "in-place bound input did not observe CPU write");
        std::memset(block, 0x11, bytes);
        snapshotRecorder.Sync();
#ifdef _WIN32
        Require(_putenv_s("APS5_DRAW_SNAPSHOT_MAX_MIB", "64") == 0, "cannot raise snapshot cap");
#else
        Require(setenv("APS5_DRAW_SNAPSHOT_MAX_MIB", "64", 1) == 0, "cannot raise snapshot cap");
#endif
        const auto again = resources.PrepareDrawBindings(snapshotRecorder);
        Require(again != nullptr && again->snapshots.size() == 1 && again->snapshots[0].address == smallAddress, "built resources read the draw snapshot cap again instead of keeping the one they were built with");
        snapshotRecorder.Sync();
#ifdef _WIN32
        Require(_putenv_s("APS5_DRAW_SNAPSHOT_MAX_MIB", "1") == 0, "cannot restore snapshot cap");
#else
        Require(setenv("APS5_DRAW_SNAPSHOT_MAX_MIB", "1", 1) == 0, "cannot restore snapshot cap");
#endif
    }
    const auto firstAddress = address + 4096;
    constexpr std::size_t firstBytes = 600 * 1024;
    const auto secondAddress = address + 600 * 1024 + 8192;
    constexpr std::size_t secondBytes = 600 * 1024;
    ShaderRecompiler::RecompileResult cumulativeProgram;
    ShaderRecompiler::DescriptorBinding firstBinding;
    firstBinding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    firstBinding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    firstBinding.descriptorSet = 0;
    firstBinding.binding = 0;
    firstBinding.count = 1;
    firstBinding.guestDescriptor = {static_cast<std::uint32_t>(firstAddress), static_cast<std::uint32_t>(firstAddress >> 32u) & 0xffffu, static_cast<std::uint32_t>(firstBytes), 0x31000000u};
    firstBinding.bufferWritten = {false};
    cumulativeProgram.bindings.push_back(firstBinding);
    ShaderRecompiler::DescriptorBinding secondBinding;
    secondBinding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    secondBinding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    secondBinding.descriptorSet = 0;
    secondBinding.binding = 1;
    secondBinding.count = 1;
    secondBinding.guestDescriptor = {static_cast<std::uint32_t>(secondAddress), static_cast<std::uint32_t>(secondAddress >> 32u) & 0xffffu, static_cast<std::uint32_t>(secondBytes), 0x31000000u};
    secondBinding.bufferWritten = {false};
    cumulativeProgram.bindings.push_back(secondBinding);
    const CompiledShader cumulativeCompute{ShaderRecompiler::ShaderStage::Compute, &cumulativeProgram, 0};
    {
        auto snapshotContext = context;
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, cumulativeCompute);
        const auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
        Require(bindings != nullptr, "read-only draw bindings were not prepared for cumulative test");
        Require(bindings->snapshots.size() == 1, "cumulative cap permitted both inputs to snapshot");
        Require(bindings->snapshots[0].address == firstAddress, "the earlier binding in the layout was not prioritised for the snapshot budget");
        snapshotRecorder.Sync();
    }
    recorder.Activate();
}

void drawSnapshotEvictionTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t bytes = 65536;
    void* block = WriteWatched() ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "write watching unavailable: draw snapshot eviction not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    const auto address = reinterpret_cast<std::uint64_t>(block);
    Recorder cache(device.GetContext());
    const auto buffer = std::make_shared<Buffer>(device.GetContext(), 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto registry = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    const auto generation = CollectWrites(address, 4096);
    Require(generation != 0, "the watched block has no generation");
    constexpr auto cap = Recorder::DrawSnapshotEntries;
    cache.KeepDrawSnapshot(address, 16, generation, registry, buffer, Recorder::SnapshotUse::Vertex);
    for (std::size_t size = 1; size <= cap; ++size) cache.KeepDrawSnapshot(address, size, generation, registry, buffer);
    Require(cache.ReusableDrawSnapshot(address, 1) == buffer, "a kept snapshot is not reusable");
    cache.KeepDrawSnapshot(address, cap + 1, generation, registry, buffer);
    Require(cache.ReusableDrawSnapshot(address, 2) == nullptr, "the least recently used snapshot survived the cap");
    Require(cache.ReusableDrawSnapshot(address, 1) == buffer && cache.ReusableDrawSnapshot(address, 3) == buffer && cache.ReusableDrawSnapshot(address, cap + 1) == buffer, "eviction dropped a more recently used snapshot");
    cache.KeepDrawSnapshot(address, cap + 2, generation, registry, buffer);
    Require(cache.ReusableDrawSnapshot(address, 4) == nullptr && cache.ReusableDrawSnapshot(address, 1) == buffer, "the second eviction did not take the next oldest");
    Require(cache.ReusableDrawSnapshot(address, 16, Recorder::SnapshotUse::Vertex) == buffer, "storage snapshots evicted a vertex snapshot");
    cache.KeepDrawSnapshot(address, 1, generation, registry, buffer);
    Require(cache.ReusableDrawSnapshot(address, 1) == buffer && cache.ReusableDrawSnapshot(address, 5) == buffer, "replacing a kept snapshot evicted another");
    cache.KeepDrawSnapshot(address + 8192, Recorder::DrawSnapshotBudget, generation, registry, buffer);
    Require(cache.ReusableDrawSnapshot(address, 1) == nullptr && cache.ReusableDrawSnapshot(address, 5) == nullptr, "the byte budget did not evict the older snapshots");
    cache.KeepDrawSnapshot(address, 16, generation, registry, buffer);
    std::memset(block, 0x5a, 16);
    CollectWrites(address, 16);
    Require(cache.ReusableDrawSnapshot(address, 16) == nullptr && cache.ReusableDrawSnapshot(address, 16) == nullptr, "a snapshot outlived a CPU store");
}

// LOCAL DEBUG (APS5_PROFILE_DRAW, REVALIDATE-PLAN 7 and 9, I0): the draw-input lease counters over
// real CopyDrawInput calls, as the environment sets them up (counting with a collect memo key,
// counting without one, off). Every row checks the same observable results (reuse, buffers, derived
// values) and that NothingPendingOver gives no proof at PendingSerial 0 (NoSerial, counted apart from
// query nothing-over) but one after a bump; the counting rows also check each row's prediction: (a) a
// repeat holds P (memo), C and M together; (c) an epoch bump makes C miss (epoch), the call walks and refills; (b) a CPU store seen at
// the next epoch makes UnchangedSince fail, a copy is made and kept, and M misses (set serial moved)
// next time; (d) an unwatch makes C miss (unwatch), an allocation change makes C miss (mapping) and
// erases the entry through sameRegistry; (j) a padding refusal; (h) a vertex lower_bound served by a
// wider entry, then an in-between size kept: M misses and finds the new entry; a PendingSerial bump
// makes P's memo miss and its query answer; a probe without the GpuMutex gets no lease (counted); an
// addition to the pending registry counts U only off the GpuMutex. W and M wrong stay 0. Off, nothing
// is counted and no lease is handed out.
bool drawInputLeaseCounterTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    using Use = Recorder::SnapshotUse;
    using C = InputLeaseCounts;
    constexpr std::size_t block = 65536;
    constexpr std::size_t bytes = 4 * block;
    void* memory = AllocateWatched(bytes, block);
    if (memory == nullptr) {
        std::cout << "no write watching: the draw input lease counters not tested\n";
        return false;
    }
    struct Release {
        void* memory;
        ~Release() { ReleaseWatched(memory, bytes); }
    } release{memory};
    const bool counting = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const bool keyed = counting && std::getenv("APS5_DEBUG_COVERAGE") != nullptr && std::getenv("APS5_NO_COLLECT_MEMO") == nullptr && std::getenv("APS5_SHARED_COLLECT_MEMO") == nullptr;
    const std::uint64_t one = keyed ? 1 : 0;
    const auto& context = device.GetContext();
    auto* words = static_cast<std::uint32_t*>(memory);
    for (std::uint32_t i = 0; i < bytes / sizeof(std::uint32_t); ++i) words[i] = i * 5 + 1;
    const auto address = reinterpret_cast<std::uint64_t>(memory);
    constexpr std::size_t size = 256;
    auto before = InputLeaseCountsNow();
    // Each step's counts since the last step (counting rows only); W and M wrong stay 0 in every row.
    const auto expect = [&](std::initializer_list<std::pair<std::uint64_t C::*, std::uint64_t>> counts, const char* what) {
        const auto now = InputLeaseCountsNow();
        if (counting) {
            for (const auto& [field, count] : counts) Require(now.*field - before.*field == count, what);
        }
        Require(now.cWrong == before.cWrong && now.mWrong == before.mWrong, what);
        before = now;
    };
    // PendingSerial 0, which a fresh test process still has (no storage image went pending yet), is no
    // proof for row P: the query reports NoSerial and the probe counts it apart from nothing-over.
    if (StorageTexture::PendingSerial() == 0) {
        auto over = StorageTexture::PendingOver::Nothing;
        Require(StorageTexture::NothingPendingOver(address, size, &over) == 0 && over == StorageTexture::PendingOver::NoSerial, "NothingPendingOver gave a proof at PendingSerial 0");
        {
            InputLeaseShadow shadow(&recorder, address + 2 * block, size, Use::Index32);
        }
        expect({{&C::indexCalls, 1}, {&C::inserted, 1}, {&C::pNoSerial, 1}, {&C::pQuery, 0}, {&C::pMemo, 0}}, "row P counted PendingSerial 0 as a proof");
    } else {
        std::cout << "PendingSerial is not 0 at the start: row P's serial-0 case not tested\n";
    }
    // Row P's memo needs a PendingSerial other than 0; with nothing over the range, the query proves it.
    StorageTexture::BumpPendingSerial();
    BumpCollectEpoch();
    {
        auto over = StorageTexture::PendingOver::NoSerial;
        const auto serial = StorageTexture::PendingSerial();
        Require(StorageTexture::NothingPendingOver(address, size, &over) == serial && over == StorageTexture::PendingOver::Nothing, "NothingPendingOver gave no proof with nothing pending");
    }
    const auto index = [&](std::size_t bufferBytes = 0) { return CopyDrawInput(context, &recorder, address, size, 4, Use::Index32, bufferBytes); };
    const auto equalsGuest = [&](const DrawInputCopy& copy) { return std::memcmp(copy.buffer->Bytes().data(), memory, size) == 0; };
    // (a)
    const auto first = index();
    Require(!first.reused && first.generation != 0 && equalsGuest(first), "the first draw input was not copied");
    expect({{&C::indexCalls, 1}, {&C::inserted, 1}, {&C::present, 0}, {&C::pQuery, 1}, {&C::cNotCovered, one}, {&C::cNoKey, 1 - one}, {&C::mNoLease, 1}, {&C::copied, 1}}, "the first input's lease counts are wrong");
    KeepDrawInput(&recorder, address, first, Use::Index32, 189);
    const auto second = index();
    Require(second.reused && second.buffer == first.buffer && second.derived == 189, "an unchanged draw input was copied again");
    expect({{&C::present, 1}, {&C::pMemo, 1}, {&C::cWould, one}, {&C::mNoLease, 1}, {&C::mWould, 0}, {&C::reused, 1}, {&C::held, 0}}, "the repeat's lease counts are wrong");
    const auto third = index();
    Require(third.reused && third.buffer == first.buffer, "an unchanged draw input was copied a third time");
    expect({{&C::present, 1}, {&C::pMemo, 1}, {&C::cWould, one}, {&C::mWould, 1}, {&C::reused, 1}, {&C::held, one}}, "a third use did not hold every row");
    // (c)
    BumpCollectEpoch();
    Require(index().reused, "an epoch bump made an unchanged draw input copy");
    expect({{&C::cEpoch, one}, {&C::cWould, 0}, {&C::pMemo, 1}, {&C::mWould, 1}, {&C::held, 0}}, "an epoch bump did not make row C miss");
    Require(index().reused, "an unchanged draw input was copied after an epoch bump");
    expect({{&C::cWould, one}, {&C::held, one}}, "row C did not refill after an epoch bump");
    // (b)
    words[5] = 0xdead;
    BumpCollectEpoch();
    const auto stored = index();
    Require(!stored.reused && stored.buffer != first.buffer && equalsGuest(stored), "a draw input outlived a CPU store");
    expect({{&C::cEpoch, one}, {&C::mWould, 1}, {&C::copied, 1}}, "a CPU store's lease counts are wrong");
    KeepDrawInput(&recorder, address, stored, Use::Index32, 7);
    Require(index().derived == 7, "the copy after the CPU store was not kept");
    expect({{&C::mMoved, 1}, {&C::mWould, 0}, {&C::reused, 1}}, "row M held across its entry's replacement");
    // (d)
    Unwatch(address + 3 * block, block);
    Require(index().derived == 7, "an unwatch elsewhere dropped the kept copy");
    expect({{&C::cUnwatch, one}, {&C::mWould, 1}}, "an unwatch did not make row C miss");
    alignas(64) static std::byte other[64];
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(other, sizeof(other), true, true, true);
    }
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(other);
    }
    const auto remapped = index();
    Require(!remapped.reused && equalsGuest(remapped), "a draw input outlived a registry mutation");
    expect({{&C::cMapping, one}, {&C::mWould, 1}, {&C::copied, 1}}, "an allocation change did not make row C miss, or row M did not find the entry it erased");
    KeepDrawInput(&recorder, address, remapped, Use::Index32, 9);
    // (j)
    const auto padded = index(2 * size);
    Require(!padded.reused && padded.generation != 0 && padded.buffer->Bytes().size() == 2 * size && equalsGuest(padded), "a kept copy without the padding was reused");
    expect({{&C::padding, 1}, {&C::copied, 0}, {&C::reused, 0}}, "a padding refusal was not counted");
    Require(index().derived == 9, "the padding refusal dropped the kept copy");
    expect({{&C::reused, 1}}, "the kept copy after a padding refusal was not reused");
    // (h)
    const auto pool = address + block;
    const auto vertex = [&](std::size_t length) { return CopyDrawInput(context, &recorder, pool, length, 1, Use::Vertex); };
    const auto whole = vertex(12288);
    Require(!whole.reused, "the vertex pool was reused before it was kept");
    KeepDrawInput(&recorder, pool, whole, Use::Vertex, 0);
    Require(vertex(4096).buffer == whole.buffer && vertex(4096).buffer == whole.buffer, "a shorter vertex read did not reuse the longer snapshot");
    expect({{&C::vertexCalls, 3}, {&C::inserted, 2}, {&C::mNoLease, 2}, {&C::mWould, 1}}, "a vertex read served by a wider entry was not leased");
    const auto middle = std::make_shared<Buffer>(context, 8192, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    recorder.KeepDrawSnapshot(pool, 8192, CollectWrites(pool, 8192), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), middle, Use::Vertex);
    Require(vertex(4096).buffer == middle, "a vertex read did not take the in-between snapshot kept after its lease");
    expect({{&C::mMoved, 1}, {&C::mWould, 0}}, "row M held across an in-between vertex snapshot");
    Require(vertex(4096).buffer == middle, "a vertex read left the in-between snapshot");
    expect({{&C::mWould, 1}}, "row M did not refill with the in-between snapshot");
    // Row P's memo after a PendingSerial bump.
    StorageTexture::BumpPendingSerial();
    Require(index().derived == 9, "a PendingSerial bump dropped the kept copy");
    expect({{&C::pMemo, 0}, {&C::pQuery, 1}}, "a PendingSerial bump did not make row P's memo miss");
    Require(index().derived == 9, "the kept copy was not reused");
    expect({{&C::pMemo, 1}}, "row P's query did not refill its memo");
    // A probe without the GpuMutex, an input bound in place, and the additions.
    bool unleased = false;
    std::thread([&] { unleased = recorder.InputLeaseFor(address, size, Use::Index32) == nullptr; }).join();
    Require(unleased, "a probe without the GpuMutex got a lease");
    {
        InputLeaseShadow shadow(&recorder, pool, 4096, Use::Vertex);
        shadow.InPlace();
    }
    expect({{&C::withoutGpuMutex, 1}, {&C::vertexCalls, 1}, {&C::inPlace, 1}}, "a probe without the GpuMutex or an input bound in place was not counted");
    if (counting) {
        const auto additions = PendingAdditions();
        NotePendingAddition(PendingAdditionSite::Reconcile);
        std::thread([] { NotePendingAddition(PendingAdditionSite::Shadow); }).join();
        const auto now = InputLeaseCountsNow();
        Require(PendingAdditions() == additions + 2, "the pending additions were not counted");
        Require(now.unlocked[static_cast<std::size_t>(PendingAdditionSite::Reconcile)] == before.unlocked[static_cast<std::size_t>(PendingAdditionSite::Reconcile)] && now.unlocked[static_cast<std::size_t>(PendingAdditionSite::Shadow)] == before.unlocked[static_cast<std::size_t>(PendingAdditionSite::Shadow)] + 1, "U counted an addition under the GpuMutex, or missed one off it");
    } else {
        const auto now = InputLeaseCountsNow();
        Require(recorder.InputLeaseFor(address, size, Use::Index32) == nullptr, "a lease was handed out without the counters");
        Require(now.indexCalls == 0 && now.vertexCalls == 0 && now.inserted == 0 && now.present == 0 && now.pQuery == 0 && now.withoutGpuMutex == 0 && now.mNoLease == 0, "the lease counters counted without APS5_PROFILE_DRAW");
    }
    recorder.Sync();
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_INPUT_LEASES, REVALIDATE-PLAN 9): =2's death rows of L1 (each ends
// the process through abort(), turned into exit code 3 as for the snapshot hash's =3 case).
enum class InputLeaseDeath : std::uint8_t { None, C, M, P1, P2 };

// (q2): a removal's PendingSerial bump (no addition), between row P's prediction and today's call.
void bumpPendingSerialForTest() {
    StorageTexture::BumpPendingSerial();
}

[[maybe_unused]] void inputLeaseExitOnAbort(int) {
    std::_Exit(3);
}

void prepareInputLeaseDeath(const char* what) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#else
    std::signal(SIGABRT, inputLeaseExitOnAbort);
#endif
    std::cout << "APS5_EXPERIMENT_INPUT_LEASES=2: " << what << "; the next draw input must abort" << std::endl;
}

// A speculative read gate that admits every read (row E's mirror test).
struct InputLeaseTestGate final : AgcDriver::GuestMemory::ISpeculativeReadGate {
    void Enter() override {}
    void Leave() override {}
};

// LOCAL EXPERIMENT (APS5_EXPERIMENT_INPUT_LEASES, REVALIDATE-PLAN 5 and 9, L1 rows C, M and E): real
// CopyDrawInput calls in the switch's mode (every row has the epoch coverage that gives the thread a
// collect memo key). Every case gives the same observable results in every mode (reuse, buffers,
// derived values, generations); per step the switch's counts show which rows held: =1 applies them,
// =2 predicts and verifies them, unset counts nothing. (a) a repeat holds C and M and returns the same
// buffer and derived value, without walking (profiled rows); (c) an epoch bump makes C miss, the call
// refills it; (b) a CPU store seen at the next epoch makes UnchangedSince fail, a copy is made, and M
// misses next time (set serial moved); (d) an unwatch makes C miss, an allocation change makes C miss
// and erases the entry through sameRegistry; (j) a padding refusal gives the same generation as a
// collect; (i) a budget eviction of the leased entry makes M miss with no stale iterator used; (h) a
// vertex lower_bound served by a wider entry holds M, then an in-between size kept makes M miss and
// return the new entry; (k) a host write queued after the lease is drained by UnchangedSince and a copy
// is made; (n) a second thread that adopted the epoch and holds the GpuMutex does not take row C
// (owner mismatch) and walks (profiled rows), and C then misses back on this thread; (m) nested
// speculative read scopes on a helper thread: the gate's mirror equals the thread_local at every
// point and a flush is refused inside. Row P (stage S4) holds at the flush hook of every one of these
// inputs (nothing is pending over them; one after-input check each at =2); its own cases follow over a
// second block of storage images (registered, without host imports, as the slice tests make them):
// both skip points of a vertex range nothing is pending over, by query then by memo; (e) a pending
// image over a vertex range: P does not hold and today's path flushes it; (f) an image dirty only in
// another slice: P does not hold over the slice with no pending unit, nor after units are added there
// without a PendingSerial bump; (o) with no flush hook installed the hook arguments are cleared on
// return, and a later hook does not skip, at the S they would carry nor at another; (p) a storage image
// marked dirty by a thread without the GpuMutex (markLayersPending's real addition site) turns row P
// off for good (counted once; the next vertex input over it holds no point and today's path flushes
// it; =2 prints U and does not abort), and so does MarkShadowed's call made directly from such a
// thread (last, as the switch stays off). Not tested here (open gaps): (g) MarkShadowed over the range
// (a unit-shadow slab needs a watched host import), and (l) that FlushDepthPlanes still runs where row
// P holds, with and without HookFlushCpuBlocks (no depth plane fixture in this test; the
// _hook_flush rows check only the holds).
// `death` (=2): the coverage dropped under the same key (C); an in-between vertex entry kept without its
// set serial's bump (M); a pending image hidden from row P's query (P1), and the same with a removal's
// PendingSerial bump between the prediction and today's call (P2), which must not excuse it: the next
// input must abort.
bool drawInputLeaseTests(const Device& device, Recorder& recorder, std::unique_lock<AgcDriver::GuestMemory::GpuMutexType>& gpu, InputLeaseDeath death) {
    using namespace AgcDriver::GuestMemory;
    using Use = Recorder::SnapshotUse;
    constexpr std::size_t block = 65536;
    constexpr std::size_t bytes = 4 * block;
    void* memory = AllocateWatched(bytes, block);
    if (memory == nullptr) {
        std::cout << "no write watching: the draw input leases not tested\n";
        return false;
    }
    struct Release {
        void* memory;
        ~Release() { ReleaseWatched(memory, bytes); }
    } release{memory};
    const int mode = InputLeases();
    const bool keyed = std::getenv("APS5_DEBUG_COVERAGE") != nullptr && std::getenv("APS5_NO_COLLECT_MEMO") == nullptr && std::getenv("APS5_SHARED_COLLECT_MEMO") == nullptr;
    const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const bool rowC = InputLeaseRowOn(InputLeaseRow::C);
    const bool rowM = InputLeaseRowOn(InputLeaseRow::M);
    const std::uint64_t one = keyed ? 1 : 0;
    const auto& context = device.GetContext();
    auto* words = static_cast<std::uint32_t*>(memory);
    for (std::uint32_t i = 0; i < bytes / sizeof(std::uint32_t); ++i) words[i] = i * 7 + 3;
    const auto address = reinterpret_cast<std::uint64_t>(memory);
    constexpr std::size_t size = 256;
    const bool rowP = InputLeaseRowOn(InputLeaseRow::P);
    BumpCollectEpoch();
    // Row P needs a PendingSerial other than 0 (a fresh process may still have 0: no proof).
    StorageTexture::BumpPendingSerial();
    auto before = InputLeaseExperimentNow();
    // The step's holds since the last step: rows C and M (Any: not checked) and the skip points row P
    // held, each applied at =1, predicted and verified at =2 (none void or raced, and one after-input
    // check for each of the step's `inputs` that held a point); none counted with the switch unset or the
    // row off.
    constexpr std::uint64_t Any = ~std::uint64_t{0};
    using X = InputLeaseExperimentCounts;
    const auto expectRows = [&](std::uint64_t c, std::uint64_t m, std::uint64_t p, std::uint64_t inputs, const char* what) {
        const auto now = InputLeaseExperimentNow();
        const auto delta = [&](std::uint64_t X::*field) { return now.*field - before.*field; };
        const auto cRows = rowC ? c : 0;
        const auto mRows = rowM ? m : 0;
        const auto pRows = rowP ? p : 0;
        if (mode == 1) {
            if (c != Any) Require(delta(&X::cHeld) == cRows, what);
            if (m != Any) Require(delta(&X::mHeld) == mRows, what);
            Require(delta(&X::pHeld) == pRows && delta(&X::cPredicted) == 0 && delta(&X::mPredicted) == 0 && delta(&X::pPredicted) == 0, what);
        } else if (mode == 2) {
            if (c != Any) Require(delta(&X::cPredicted) == cRows && delta(&X::cVerified) == cRows, what);
            if (m != Any) Require(delta(&X::mPredicted) == mRows && delta(&X::mVerified) == mRows, what);
            Require(delta(&X::cVoided) == 0 && delta(&X::pPredicted) == pRows && delta(&X::pVerified) == pRows && delta(&X::pRaced) == 0 && delta(&X::pAfter) == (pRows != 0 ? inputs : 0) && delta(&X::pAfterMoved) == 0, what);
            Require(delta(&X::cHeld) == 0 && delta(&X::mHeld) == 0 && delta(&X::pHeld) == 0, what);
        } else {
            Require(now.cHeld == 0 && now.mHeld == 0 && now.pHeld == 0 && now.cPredicted == 0 && now.mPredicted == 0 && now.pPredicted == 0, what);
        }
        before = now;
    };
    // Every step of rows C and M below is one CopyDrawInput, whose flush hook row P holds.
    const auto expect = [&](std::uint64_t c, std::uint64_t m, const char* what) { expectRows(c, m, 1, 1, what); };
    const auto index = [&](std::size_t bufferBytes = 0) { return CopyDrawInput(context, &recorder, address, size, 4, Use::Index32, bufferBytes); };
    const auto equalsGuest = [&](const DrawInputCopy& copy) { return std::memcmp(copy.buffer->Bytes().data(), memory, size) == 0; };
    // (a)
    const auto first = index();
    Require(!first.reused && first.generation != 0 && equalsGuest(first), "the first draw input was not copied");
    expect(0, 0, "the first draw input held a row");
    KeepDrawInput(&recorder, address, first, Use::Index32, 189);
    const auto second = index();
    Require(second.reused && second.buffer == first.buffer && second.derived == 189, "an unchanged draw input was copied again");
    expect(one, 0, "the repeat did not hold row C alone");
    const auto walkedBefore = ThreadCollectedBytes();
    const auto third = index();
    Require(third.reused && third.buffer == first.buffer && third.derived == 189, "an unchanged draw input was copied a third time");
    if (profiled) Require(ThreadCollectedBytes() == walkedBefore, "a draw input whose rows held walked");
    expect(one, 1, "a third use did not hold rows C and M");
    if (death == InputLeaseDeath::C) {
        prepareInputLeaseDeath("the coverage is dropped under the same key");
        DropCollectCoverageForTest();
        static_cast<void>(index());
        Require(false, "=2 did not abort on a row C lease its thread's coverage does not hold");
    }
    // (c)
    BumpCollectEpoch();
    Require(index().reused, "an epoch bump made an unchanged draw input copy");
    expect(0, 1, "an epoch bump did not make row C miss");
    Require(index().reused, "an unchanged draw input was copied after an epoch bump");
    expect(one, 1, "row C did not refill after an epoch bump");
    // (b)
    words[5] = 0xdead;
    BumpCollectEpoch();
    const auto stored = index();
    Require(!stored.reused && stored.buffer != first.buffer && equalsGuest(stored), "a draw input outlived a CPU store");
    expect(0, 1, "a CPU store's draw input did not hold row M, or held row C");
    KeepDrawInput(&recorder, address, stored, Use::Index32, 7);
    Require(index().derived == 7, "the copy after the CPU store was not kept");
    expect(one, 0, "row M held across its entry's replacement");
    // (d)
    Unwatch(address + 3 * block, block);
    Require(index().derived == 7, "an unwatch elsewhere dropped the kept copy");
    expect(0, 1, "an unwatch did not make row C miss");
    alignas(64) static std::byte other[64];
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(other, sizeof(other), true, true, true);
    }
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(other);
    }
    const auto remapped = index();
    Require(!remapped.reused && equalsGuest(remapped) && remapped.registryGeneration == GuestAllocations::GuestAllocationsGeneration_nid_postfix(), "a draw input outlived a registry mutation, or its registry generation is not the current one");
    expect(0, 1, "an allocation change did not make row C miss, or row M did not find the entry it erased");
    KeepDrawInput(&recorder, address, remapped, Use::Index32, 9);
    // (j)
    const auto padded = index(2 * size);
    Require(!padded.reused && padded.generation != 0 && padded.buffer->Bytes().size() == 2 * size && equalsGuest(padded), "a kept copy without the padding was reused");
    if (keyed) Require(padded.generation == CollectWrites(address, size), "a padding refusal's generation is not the collect's");
    expect(one, 0, "a padding refusal's rows are wrong");
    Require(index().derived == 9, "the padding refusal dropped the kept copy");
    expect(one, 1, "the kept copy after a padding refusal did not hold rows C and M");
    // (i) A budget-sized entry evicts every draw-input entry, the leased one first.
    recorder.KeepDrawSnapshot(address + 2 * block, Recorder::DrawInputBudget, CollectWrites(address + 2 * block, 4096), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), first.buffer, Use::Vertex);
    const auto evicted = index();
    Require(!evicted.reused && equalsGuest(evicted), "a draw input whose kept copy was evicted was reused");
    expect(one, 0, "row M held across its entry's eviction");
    KeepDrawInput(&recorder, address, evicted, Use::Index32, 11);
    Require(index().derived == 11, "the copy after the eviction was not kept");
    expect(one, 0, "row M held across the copy kept after the eviction");
    Require(index().derived == 11, "the copy after the eviction was not reused");
    expect(one, 1, "rows C and M did not hold after the eviction");
    // (h)
    const auto pool = address + block;
    const auto vertex = [&](std::size_t length) { return CopyDrawInput(context, &recorder, pool, length, 1, Use::Vertex); };
    const auto whole = vertex(12288);
    Require(!whole.reused, "the vertex pool was reused before it was kept");
    expect(0, 0, "the first vertex read held a row");
    KeepDrawInput(&recorder, pool, whole, Use::Vertex, 0);
    Require(vertex(4096).buffer == whole.buffer, "a shorter vertex read did not reuse the longer snapshot");
    expect(0, 0, "a new vertex lease held a row");
    Require(vertex(4096).buffer == whole.buffer, "a shorter vertex read did not reuse the longer snapshot again");
    expect(one, 1, "a vertex read served by a wider entry did not hold rows C and M");
    const auto middle = std::make_shared<Buffer>(context, 8192, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if (death == InputLeaseDeath::M) {
        prepareInputLeaseDeath("an in-between vertex entry is kept without its set serial's bump");
        InputLeaseSuppressBumpsForTest(true);
        recorder.KeepDrawSnapshot(pool, 8192, CollectWrites(pool, 8192), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), middle, Use::Vertex);
        static_cast<void>(vertex(4096));
        Require(false, "=2 did not abort on a row M lease whose lookup finds another entry");
    }
    recorder.KeepDrawSnapshot(pool, 8192, CollectWrites(pool, 8192), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), middle, Use::Vertex);
    Require(vertex(4096).buffer == middle, "a vertex read did not take the in-between snapshot kept after its lease");
    expect(one, 0, "row M held across an in-between vertex snapshot");
    Require(vertex(4096).buffer == middle, "a vertex read left the in-between snapshot");
    expect(one, 1, "row M did not refill with the in-between snapshot");
    // (k) A host write over the index range, queued for the next tracker lock or memo hit (the hook
    // installed by hand, as APS5_EXPERIMENT_HOST_WRITE_STAMPS=1 installs it).
    {
        struct HostWriteHook {
            HostWriteHook() { GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(&note); }
            ~HostWriteHook() { GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(nullptr); }
            HostWriteHook(const HostWriteHook&) = delete;
            HostWriteHook& operator=(const HostWriteHook&) = delete;
            static void note(std::uintptr_t written, std::size_t length) { NoteHostWrite(written, length); }
        } hook;
        {
            auto* target = reinterpret_cast<std::uint8_t*>(address);
            const GuestArena::HostWrite write(target, 16);
            Require(write.Open(), "the host write over the draw input did not open");
            std::memset(target, 0x66, 16);
        }
        const auto written = index();
        Require(!written.reused && equalsGuest(written), "a draw input outlived a host write queued after its lease");
        expect(one, 1, "a host write's draw input did not hold rows C and M");
        KeepDrawInput(&recorder, address, written, Use::Index32, 13);
        Require(index().derived == 13, "the copy after the host write was not kept");
        expect(one, 0, "row M held across the copy kept after the host write");
    }
    // (n) A second thread that adopted this thread's epoch: the same collect memo key, another owner.
    gpu.unlock();
    std::string failure;
    DrawInputCopy adopted;
    bool adoptedWalked = false;
    std::thread([&] {
        try {
            std::lock_guard lock(GpuMutex());
            AdoptLatestCollectEpoch();
            const auto walked = ThreadCollectedBytes();
            adopted = index();
            adoptedWalked = ThreadCollectedBytes() != walked;
        } catch (const std::exception& error) {
            failure = error.what();
        }
    }).join();
    gpu.lock();
    Require(failure.empty(), failure);
    Require(adopted.reused && adopted.derived == 13, "a thread that adopted the epoch did not reuse the kept copy");
    if (profiled && keyed) Require(adoptedWalked, "a thread that adopted the epoch did not walk the draw input");
    expect(0, 1, "a thread that adopted the epoch took row C (another thread's coverage)");
    Require(index().derived == 13, "the kept copy was not reused after another thread's input");
    expect(0, 1, "row C held for another thread's coverage owner");
    Require(index().derived == 13, "the kept copy was not reused on this thread again");
    expect(one, 1, "rows C and M did not hold again on this thread");
    // (m) Row E's mirror: nested scopes on a helper thread.
    std::thread([&] {
        try {
            InputLeaseTestGate outer;
            InputLeaseTestGate inner;
            const auto check = [](bool reader, const char* what) { Require(SpeculativeReader() == reader && SpeculativeGateMirrored(), what); };
            check(false, "a new thread reads a speculative gate, or its mirror differs");
            {
                const SpeculativeReadScope outerScope(outer);
                check(true, "the outer speculative scope's gate is not read, or its mirror differs");
                {
                    const SpeculativeReadScope innerScope(inner);
                    check(true, "the inner speculative scope's gate is not read, or its mirror differs");
                    bool refused = false;
                    try {
                        FlushGpuWrites(address, 16);
                    } catch (const SpeculativeReadRefused&) {
                        refused = true;
                    }
                    Require(refused, "a flush inside a speculative scope was not refused");
                }
                check(true, "the outer gate was not restored, or its mirror differs");
            }
            check(false, "the gate outlived its scopes, or its mirror differs");
        } catch (const std::exception& error) {
            failure = error.what();
        }
    }).join();
    Require(failure.empty(), failure);
    Require(!SpeculativeReader() && SpeculativeGateMirrored(), "this thread's speculative gate or its mirror moved");
    // Row P's own cases (stage S4). A vertex input as prepareDrawInputs makes it: its lease view, then
    // InPlaceDrawInput (skip point (a), its FlushPending, where the device binds draw inputs in place),
    // then the copy (skip point (b), CopyDrawInput's flush hook) unless it was bound in place.
    TextureDetiler detiler(context);
    auto imageContext = context;
    imageContext.hostImportAlignment = 0;
    imageContext.detiler = &detiler;
    imageContext.recorder = &recorder;
    const auto vertexInput = [&](std::uint64_t at, std::size_t length) {
        InputLeaseShadow shadow(&recorder, at, length, Use::Vertex);
        bool behind = false;
        if (InPlaceDrawInput(imageContext, at, length, 1, &behind, &shadow) != nullptr) {
            shadow.InPlace();
            return true;
        }
        const auto copy = CopyDrawInput(imageContext, &recorder, at, length, 1, Use::Vertex, 0, &shadow);
        Require(std::memcmp(copy.buffer->Bytes().data(), reinterpret_cast<const void*>(at), length) == 0, "a vertex input's copy differs from guest memory");
        KeepDrawInput(&recorder, at, copy, Use::Vertex, 0);
        return false;
    };
    const auto points = [&](bool inPlace) -> std::uint64_t { return (imageContext.drawInputsInPlace ? 1u : 0u) + (inPlace ? 0u : 1u); };
    const auto clear = address + 2 * block + 16384;
    expectRows(Any, Any, points(vertexInput(clear, 2048)), 1, "row P did not hold at the skip points of a vertex range nothing is pending over (query)");
    expectRows(Any, Any, points(vertexInput(clear, 2048)), 1, "row P did not hold at the skip points of a vertex range nothing is pending over (memo)");
    void* imageBlock = AllocateWatched(bytes, block);
    Require(imageBlock != nullptr, "no watched block for the storage images although write watching is on");
    std::memset(imageBlock, 0x33, bytes);
    const auto images = reinterpret_cast<std::uint64_t>(imageBlock);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(imageBlock, bytes, true, true, true);
    }
    struct UnregisterImages {
        const Context& context;
        void* block;
        ~UnregisterImages() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
            ReleaseWatched(block, bytes);
        }
    } unregisterImages{context, imageBlock};
    CollectWritesUncached(images, bytes);
    const auto resource = [&](std::uint64_t at, bool layered) {
        GuestTextureResource described{};
        described.baseAddress = at;
        described.width = 128;
        described.height = 128;
        described.depthOrLastArray = layered ? 1 : 0;
        described.mipCount = 1;
        described.tileMode = TextureTileMode::kR64KBX;
        described.dimension = layered ? TextureDimension::k2DArray : TextureDimension::k2D;
        described.format = 56;
        described.dstSelX = 4;
        described.dstSelY = 5;
        described.dstSelZ = 6;
        described.dstSelW = 7;
        return described;
    };
    const auto flat = resource(images, false);
    Require(DescribeSurface(flat).guestBytes == block, "row P's image fixture is not one 64 KiB block");
    const auto pendingImage = [&](const GuestTextureResource& described) {
        auto image = StorageTexture::Create(imageContext, detiler, described, 0);
        static_cast<void>(recorder.Commands());
        recorder.Keep(image);
        return image;
    };
    if (death == InputLeaseDeath::P1 || death == InputLeaseDeath::P2) {
        auto image = pendingImage(flat);
        image->MarkDirty();
        StorageTexture::HidePendingFromQueryForTest(true);
        if (death == InputLeaseDeath::P2) InputLeaseSetPendingProbeHookForTest(&bumpPendingSerialForTest);
        prepareInputLeaseDeath(death == InputLeaseDeath::P1 ? "a pending image is hidden from row P's query" : "a pending image is hidden from row P's query and a removal bumps PendingSerial after the prediction");
        static_cast<void>(CopyDrawInput(imageContext, &recorder, images, size, 4, Use::Index32));
        Require(false, "=2 did not abort on a row P hold over a pending image");
    }
    // (e)
    {
        auto image = pendingImage(flat);
        image->MarkDirty();
        Require(image->Dirty(), "(e) the storage image did not go pending");
        static_cast<void>(vertexInput(images, 4096));
        expectRows(Any, Any, 0, 1, "(e) row P held over a pending storage image");
        Require(!image->Dirty(), "(e) the pending storage image over the vertex range was not flushed");
        recorder.Sync();
    }
    // (f)
    {
        const auto layered = resource(images + 2 * block, true);
        const auto geometry = DescribeSurface(layered);
        Require(geometry.imageLayers == 2 && geometry.layerBytes == block, "(f) the layered image does not hold two 64 KiB slices");
        auto image = pendingImage(layered);
        image->MarkSliceDirty(1);
        const auto slice0 = layered.baseAddress + geometry.GuestLayerOffset(0);
        static_cast<void>(vertexInput(slice0, 4096));
        expectRows(Any, Any, 0, 1, "(f) row P held over an image dirty in another slice");
        Require(image->Dirty(), "(f) an image with no pending unit over the range was flushed");
        // Units of a dirty image go pending without a PendingSerial bump.
        const auto serial = StorageTexture::PendingSerial();
        image->MarkSliceDirty(0);
        Require(StorageTexture::PendingSerial() == serial, "(f) a dirty image's new units bumped PendingSerial");
        static_cast<void>(vertexInput(slice0, 4096));
        expectRows(Any, Any, 0, 1, "(f) row P held over units added without a bump");
        image->WriteBack();
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        Require(!image->Dirty(), "(f) the layered image stayed pending after its write-back");
    }
    // (o) No flush hook: the arguments live only while FlushGpuWrites runs; then a hook (reinstalled)
    // over the same range, at another S, finds none.
    SetFlushHook(nullptr);
    const auto unhooked = index();
    recorder.Activate();
    Require(!InputLeaseHookArgsSet(), "(o) the flush hook's arguments outlived CopyDrawInput's FlushGpuWrites");
    Require(equalsGuest(unhooked), "(o) a draw input without a flush hook differs from guest memory");
    expectRows(Any, Any, 0, 1, "(o) row P held without a flush hook");
    // First at the S arguments left behind would still carry (a hold or prediction there could only be
    // theirs), then at another S.
    FlushGpuWrites(address, size);
    expectRows(0, 0, 0, 0, "(o) a later flush hook at the same S skipped with arguments left behind");
    StorageTexture::BumpPendingSerial();
    FlushGpuWrites(address, size);
    expectRows(0, 0, 0, 0, "(o) a later flush hook at another S skipped with arguments left behind");
    // (p) The kill switch (last: it stays on), first through a real addition site: markLayersPending's
    // push, counted before it lands whenever InputLeaseViewed() (so without APS5_PROFILE_DRAW too).
    const bool killed = mode != 0 && rowP;
    {
        auto image = pendingImage(flat);
        const auto killedBefore = InputLeaseExperimentNow().pKilled;
        std::thread([&] {
            try {
                Require(!GpuMutex().HeldByThisThread(), "(p) the marking thread holds the GpuMutex");
                image->MarkDirty();
            } catch (const std::exception& error) {
                failure = error.what();
            }
        }).join();
        Require(failure.empty(), failure);
        Require(image->Dirty(), "(p) the storage image marked dirty without the GpuMutex did not go pending");
        Require(PendingRowOff() == killed && InputLeaseExperimentNow().pKilled - killedBefore == (killed ? 1u : 0u), "(p) a storage image marked dirty without the GpuMutex did not turn row P off once, or turned it off with the switch unset or row P left out");
        static_cast<void>(vertexInput(images, 4096));
        expectRows(Any, Any, 0, 1, "(p) row P held over a storage image marked dirty without the GpuMutex");
        Require(!image->Dirty(), "(p) the storage image marked dirty without the GpuMutex was not flushed");
        recorder.Sync();
    }
    // MarkShadowed's site cannot be driven here: its call, made directly from a thread without the
    // GpuMutex, counts one more kill while the switch applies row P.
    const auto killedBefore = InputLeaseExperimentNow().pKilled;
    std::thread([] { NotePendingAddition(PendingAdditionSite::Shadow); }).join();
    Require(PendingRowOff() == killed && InputLeaseExperimentNow().pKilled - killedBefore == (killed ? 1u : 0u), "(p) MarkShadowed's addition without the GpuMutex was not counted, or turned row P off with the switch unset or row P left out");
    Require(equalsGuest(index()), "(p) a draw input after the kill switch differs from guest memory");
    if (killed) {
        expectRows(Any, Any, 0, 1, "(p) row P held after the kill switch");
    } else {
        expectRows(Any, Any, 1, 1, "(p) row P stopped holding without the switch");
    }
    recorder.Sync();
    return true;
}

// The =3 death case below must end the process with a plain exit code (see Locks.cpp's
// PrepareScopeDeath): ctest fails a run that ends in an exception without reading its output.
[[maybe_unused]] void snapshotHashExitOnAbort(int) {
    std::_Exit(3);
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_SNAPSHOT_HASH, GPU-OWNED-PLAN L1-1): the refreshes of a 4 MiB
// Storage snapshot (Recorder::refreshDrawSnapshot through ReusableDrawSnapshot) over a watched
// block, under GpuMutex. Unset, today's compares and copies run with no memo. With the memo: the
// first refresh after KeepDrawSnapshot learns every block's memo (memcmp) and the next is decided by
// it; a CPU store's block is copied and rehashed from S (the next refresh decides it equal); a
// differing block with another holder of S erases the entry; and S written through its mapping
// behind its memo in two blocks (a hidden writer) is counted stale twice at =2, aborts at =3 (the
// last case; its ctest row matches the [snapshot-hash] line), and is found by =1's audit within one
// refresh per block, which copies the audited block and the other one (both differ in that refresh,
// as today's compare would copy them) and drops the entry's memo.
bool drawSnapshotRefreshTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t bytes = std::size_t{4} << 20u;
    constexpr std::size_t block = 65536;
    constexpr std::size_t count = bytes / block;
    void* memory = WriteWatched() ? AllocateWatched(bytes, block) : nullptr;
    if (memory == nullptr) {
        std::cout << "skipped, write watching unavailable\n";
        return false;
    }
    struct Release {
        void* memory;
        ~Release() { ReleaseWatched(memory, bytes); }
    } release{memory};
    auto* guest = static_cast<std::uint8_t*>(memory);
    for (std::size_t i = 0; i < bytes; ++i) guest[i] = static_cast<std::uint8_t>((i * 131u + (i >> 11u)) & 0xFFu);
    const auto address = reinterpret_cast<std::uint64_t>(memory);
    const auto registry = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    const int mode = SnapshotHashMode();
    // ReusableDrawSnapshot asserts the hold (APS5_ASSERT_GPU_LOCK=1); the compare pool's helpers never
    // take it.
    std::unique_lock gpu(GpuMutex());
    Recorder cache(device.GetContext());
    // A new entry holding a copy of guest memory; the entry holds the only reference to S, whose
    // mapped bytes are returned.
    const auto keep = [&] {
        auto buffer = std::make_shared<Buffer>(device.GetContext(), bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const auto held = buffer->Bytes();
        std::memcpy(held.data(), memory, bytes);
        BumpCollectEpoch();
        const auto generation = CollectWrites(address, bytes);
        Require(generation != 0, "the snapshot refresh fixture has no write generation");
        cache.KeepDrawSnapshot(address, bytes, generation, registry, std::move(buffer));
        return held;
    };
    // A lookup as PrepareDrawBindings makes it: the blocks stamped since the entry's generation are
    // compared. Whether the entry survived.
    const auto refresh = [&] {
        BumpCollectEpoch();
        const auto generation = CollectWrites(address, bytes);
        return cache.ReusableDrawSnapshot(address, bytes, Recorder::SnapshotUse::Storage, nullptr, generation) != nullptr;
    };
    const auto current = [&](std::span<std::byte> held) { return std::memcmp(held.data(), memory, bytes) == 0; };
    // At =2/=3 the [snapshot-stats] line prints at the process's first refresh (then every 10 s),
    // taking the memo's counts: a refresh of a first entry lets it print before the counted cases.
    keep();
    MarkWritten(address, bytes);
    Require(refresh(), "a refresh over unchanged bytes lost the snapshot");
    TakeSnapshotHashCounts();
    auto held = keep();
    // Memo learned, then used: a driver stamp over unchanged bytes has every block compared.
    MarkWritten(address, bytes);
    Require(refresh() && current(held), "a refresh over unchanged bytes lost the snapshot");
    auto counts = TakeSnapshotHashCounts();
    Require(counts.learned == (mode != 0 ? count : 0) && counts.decided == 0, mode != 0 ? "the first refresh did not learn every block's hash memo" : "a hash memo was used with the switch unset");
    MarkWritten(address, bytes);
    Require(refresh() && current(held), "a refresh over unchanged bytes lost the snapshot");
    counts = TakeSnapshotHashCounts();
    if (mode == 0) Require(counts.learned == 0 && counts.decided == 0 && counts.audited == 0, "a hash memo was used with the switch unset");
    else Require(counts.decided == count && counts.decidedEqual == count && counts.learned == 0 && counts.falseEqual == 0 && counts.stale == 0 && counts.raced == 0 && counts.audited == (mode == 1 ? 1 : 0) && counts.anomalies == 0, "the second refresh was not decided by the hash memo");
    // Copy and rehash: a CPU store's block alone is compared, differs, and is copied, then rehashed
    // from S, so the next refresh decides it equal.
    guest[5 * block + 123] ^= 0xFFu;
    Require(refresh() && current(held), "a refresh did not copy a CPU store's block");
    counts = TakeSnapshotHashCounts();
    if (mode != 0) Require(counts.decided == 1 && counts.decidedEqual == 0 && counts.rehashed == 1, "a CPU store's block was not decided differing by its memo, or not rehashed after its copy");
    MarkWritten(address, bytes);
    Require(refresh() && current(held), "a refresh over unchanged bytes lost the snapshot");
    counts = TakeSnapshotHashCounts();
    if (mode != 0) Require(counts.decided == count && counts.decidedEqual == count && counts.rehashed == 0, "a rehashed memo does not describe the copied block");
    // Not sole: with another holder of S, a differing block fails the refresh and erases the entry
    // (with its memo), and S is left as the other holder reads it.
    {
        const auto other = cache.ReusableDrawSnapshot(address, bytes);
        Require(other != nullptr, "the refreshed snapshot is not reusable");
        guest[9 * block] ^= 0xFFu;
        Require(!refresh(), "a refresh copied into a snapshot another holder still reads");
        Require(cache.ReusableDrawSnapshot(address, bytes) == nullptr, "a snapshot that could not be refreshed was kept");
        Require(std::memcmp(held.data() + 9 * block, guest + 9 * block, block) != 0, "a failed refresh wrote a snapshot another holder still reads");
    }
    TakeSnapshotHashCounts();
    // S written through its mapping behind its memo in blocks 7 and 11, guest memory unchanged. A
    // recreated entry starts without a memo, so it learns first.
    held = keep();
    MarkWritten(address, bytes);
    Require(refresh() && current(held), "a refresh over unchanged bytes lost the recreated snapshot");
    counts = TakeSnapshotHashCounts();
    Require(counts.learned == (mode != 0 ? count : 0) && counts.decided == 0, "a recreated snapshot did not start without a hash memo");
    held[7 * block + 40] ^= std::byte{0x5A};
    held[11 * block + 4000] ^= std::byte{0x5A};
    if (mode == 0) {
        MarkWritten(address, bytes);
        Require(refresh() && current(held), "the compare did not heal a snapshot written behind it");
    } else if (mode == 1) {
        std::size_t refreshes = 0;
        do {
            MarkWritten(address, bytes);
            Require(refresh(), "a refresh over unchanged bytes lost the snapshot");
            ++refreshes;
            counts = TakeSnapshotHashCounts();
            Require(counts.audited == 1, "a refresh decided by the hash memo audited no block");
        } while (counts.anomalies == 0 && refreshes < count);
        Require(counts.anomalies == 1 && counts.falseEqual == 0 && counts.raced == 0, "the audit did not find a block of S written behind its memo within one refresh per block");
        Require(counts.rehashed == 2 && current(held), "the audit's anomaly did not copy every memo-decided block of its refresh that differs");
        MarkWritten(address, bytes);
        Require(refresh() && current(held), "a refresh over unchanged bytes lost the snapshot");
        counts = TakeSnapshotHashCounts();
        Require(counts.learned == count - 2 && counts.decided == 2 && counts.decidedEqual == 2 && counts.anomalies == 0, "the audit's anomaly did not drop the entry's memo, or the copied blocks were not rehashed");
    } else if (mode == 2) {
        MarkWritten(address, bytes);
        Require(refresh() && current(held), "=2 did not use memcmp's verdict for a snapshot written behind its memo");
        counts = TakeSnapshotHashCounts();
        Require(counts.stale == 2 && counts.falseEqual == 0 && counts.raced == 0 && counts.rehashed == 2, "=2 did not count the stale memos of a snapshot written behind them");
    } else {
#ifdef _WIN32
        SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#else
        std::signal(SIGABRT, snapshotHashExitOnAbort);
#endif
        std::cout << "APS5_EXPERIMENT_SNAPSHOT_HASH=3: the next refresh must abort" << std::endl;
        MarkWritten(address, bytes);
        refresh();
        Require(false, "=3 did not abort on the stale memo of a snapshot written behind it");
    }
    return true;
}

void unwrittenSnapshotTests(const Device& device) {
    Recorder cache(device.GetContext());
    const auto buffer = std::make_shared<Buffer>(device.GetContext(), 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    constexpr std::uint64_t address = 0x10000000;
    const std::array<std::uint64_t, 3> stamp{1, 1, 1};
    const std::array<std::uint64_t, 3> next{1, 2, 1};
    cache.NoteUnwrittenDrawSnapshot(address, 16, stamp, buffer);
    Require(cache.UnwrittenDrawSnapshot(address, 16, stamp) == buffer, "a noted unwritten snapshot is not found under its stamp");
    cache.NoteUnwrittenDrawSnapshot(address + 8192, Recorder::DrawSnapshotBudget, stamp, buffer);
    Require(cache.UnwrittenDrawSnapshot(address + 8192, Recorder::DrawSnapshotBudget, stamp) == nullptr, "unwritten snapshots grew past the byte budget");
    cache.NoteUnwrittenDrawSnapshot(address, Recorder::DrawSnapshotBudget, stamp, buffer);
    Require(cache.UnwrittenDrawSnapshot(address, Recorder::DrawSnapshotBudget, stamp) == buffer, "replacing an unwritten snapshot kept the bytes of the old one");
    Require(cache.UnwrittenDrawSnapshot(address, Recorder::DrawSnapshotBudget, next) == nullptr && cache.UnwrittenDrawSnapshot(address, Recorder::DrawSnapshotBudget, stamp) == nullptr, "an unwritten snapshot outlived a newer stamp");
    for (std::size_t entry = 0; entry < Recorder::UnwrittenSnapshotEntries; ++entry) cache.NoteUnwrittenDrawSnapshot(address + entry * 16, 1, next, buffer);
    cache.NoteUnwrittenDrawSnapshot(address + Recorder::UnwrittenSnapshotEntries * 16, 1, next, buffer);
    Require(cache.UnwrittenDrawSnapshot(address, 1, next) == buffer && cache.UnwrittenDrawSnapshot(address + Recorder::UnwrittenSnapshotEntries * 16, 1, next) == nullptr, "unwritten snapshots grew past the entry cap");
    const std::array<std::uint64_t, 3> later{1, 3, 1};
    cache.NoteUnwrittenDrawSnapshot(address, 16, later, buffer, 256);
    VkDeviceSize offset = 0;
    Require(cache.UnwrittenDrawSnapshot(address, 16, later, &offset) == buffer && offset == 256, "an unwritten snapshot lost its offset in its buffer");
}

void drawSnapshotUploadTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    const auto limit = ShaderResources::DrawUploadLimit();
    if (limit < 256 || limit > 16384) {
        std::cout << "draw upload limit outside 256..16384 bytes: draw snapshot uploads not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "host imports or write watching unavailable: draw snapshot uploads not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    for (std::size_t index = 0; index < bytes; ++index) static_cast<std::uint8_t*>(block)[index] = static_cast<std::uint8_t>(index * 7u + 1u);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the watched block refused: draw snapshot uploads not tested\n";
        return;
    }
    const std::array<std::uint64_t, 3> elements{address + 4096, address + 8192, address + 32768};
    const std::array<std::size_t, 3> sizes{256, limit, limit + 256};
    ShaderRecompiler::RecompileResult program;
    for (std::uint32_t index = 0; index < elements.size(); ++index) {
        ShaderRecompiler::DescriptorBinding binding;
        binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
        binding.role = ShaderRecompiler::DescriptorRole::GuestBuffers;
        binding.descriptorSet = 0;
        binding.binding = index;
        binding.count = 1;
        binding.guestDescriptor = {static_cast<std::uint32_t>(elements[index]), static_cast<std::uint32_t>(elements[index] >> 32u) & 0xffffu, static_cast<std::uint32_t>(sizes[index]), 0x31000000u};
        binding.bufferWritten = {false};
        program.bindings.push_back(binding);
    }
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    {
        auto snapshotContext = context;
        DescriptorCache cache(snapshotContext);
        snapshotContext.descriptorCache = &cache;
        Recorder snapshotRecorder(snapshotContext);
        snapshotRecorder.Activate();
        ShaderResources resources(snapshotContext, compute);
        const auto alignment = std::max<VkDeviceSize>(snapshotContext.limits.minStorageBufferOffsetAlignment, 16);
        const auto bindings = resources.PrepareDrawBindings(snapshotRecorder);
        Require(bindings != nullptr && bindings->snapshots.size() == elements.size(), "read-only draw inputs were not snapshotted");
        for (std::size_t index = 0; index < elements.size(); ++index) {
            const auto& snapshot = bindings->snapshots[index];
            Require(snapshot.address == elements[index] && snapshot.bytes == sizes[index] && snapshot.offset % alignment == 0, "a draw snapshot is not an aligned copy of its input");
            Require(std::memcmp(snapshot.Bytes().data(), reinterpret_cast<const void*>(elements[index]), sizes[index]) == 0, "a draw snapshot does not hold the guest bytes of its draw");
        }
        Require(bindings->snapshots[0].buffer == bindings->snapshots[1].buffer && bindings->snapshots[0].offset != bindings->snapshots[1].offset, "draw inputs up to the upload limit do not share the recorder's upload chunk");
        Require(bindings->snapshots[2].buffer != bindings->snapshots[0].buffer && bindings->snapshots[2].offset == 0, "a draw input above the upload limit was copied into the upload chunk");
        const auto earlier = bindings->snapshots[0].Bytes();
        const std::vector<std::byte> before(earlier.begin(), earlier.end());
        std::memset(reinterpret_cast<void*>(elements[0]), 0x5a, sizes[0]);
        const auto again = resources.PrepareDrawBindings(snapshotRecorder);
        Require(again != nullptr && again->snapshots.size() == elements.size(), "read-only draw inputs were not snapshotted again");
        const auto copied = again->snapshots[0].Bytes();
        Require(again->snapshots[0].buffer != bindings->snapshots[0].buffer || again->snapshots[0].offset != bindings->snapshots[0].offset, "a draw input copied again took the slice an earlier draw reads");
        Require(std::all_of(copied.begin(), copied.end(), [](std::byte value) { return value == std::byte{0x5a}; }), "a draw snapshot copied again misses the CPU store");
        Require(std::equal(before.begin(), before.end(), earlier.begin()), "copying a draw input again changed the slice an earlier draw reads");
        const auto first = snapshotRecorder.DrawUpload(16);
        const auto second = snapshotRecorder.DrawUpload(16);
        Require(second.buffer == first.buffer && second.offset >= first.offset + 16 && second.offset % alignment == 0, "upload slices overlap or are not aligned");
        const auto whole = snapshotRecorder.DrawUpload(Recorder::DrawUploadChunk);
        Require(whole.buffer != second.buffer && whole.offset == 0, "a full upload chunk was not replaced");
        snapshotRecorder.Sync();
    }
    recorder.Activate();
}

void drawInputReuseTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    using Use = Recorder::SnapshotUse;
    constexpr std::size_t bytes = 65536;
    void* block = WriteWatched() ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "write watching unavailable: draw input reuse not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    const auto& context = device.GetContext();
    auto* words = static_cast<std::uint32_t*>(block);
    for (std::uint32_t i = 0; i < bytes / sizeof(std::uint32_t); ++i) words[i] = i * 3;
    const auto address = reinterpret_cast<std::uint64_t>(block);
    constexpr std::size_t size = 256;
    const auto equalsGuest = [&](const DrawInputCopy& copy) {
        const auto contents = copy.buffer->Bytes();
        return contents.size() == size && std::memcmp(contents.data(), block, size) == 0;
    };
    const auto first = CopyDrawInput(context, &recorder, address, size, 4, Use::Index32);
    Require(!first.reused && first.generation != 0 && equalsGuest(first), "the first draw input copy is wrong");
    KeepDrawInput(&recorder, address, first, Use::Index32, 189);
    const auto second = CopyDrawInput(context, &recorder, address, size, 4, Use::Index32);
    Require(second.reused && second.buffer == first.buffer && second.derived == 189, "an unchanged draw input was copied again");
    const auto vertex = CopyDrawInput(context, &recorder, address, size, 1, Use::Vertex);
    const auto narrow = CopyDrawInput(context, &recorder, address, size, 2, Use::Index16);
    Require(!vertex.reused && !narrow.reused && equalsGuest(vertex) && equalsGuest(narrow), "a draw input reused another use's copy");
    KeepDrawInput(&recorder, address, vertex, Use::Vertex, 0);
    Require(CopyDrawInput(context, &recorder, address, size, 1, Use::Vertex).buffer == vertex.buffer && CopyDrawInput(context, &recorder, address, size, 4, Use::Index32).buffer == first.buffer, "the uses' copies displaced each other");
    Require(!CopyDrawInput(context, nullptr, address, size, 4, Use::Index32).reused, "a draw input was reused without a recorder");
    words[5] = 0xdead;
    const auto stored = CopyDrawInput(context, &recorder, address, size, 4, Use::Index32);
    Require(!stored.reused && stored.buffer != first.buffer && equalsGuest(stored), "a draw input outlived a CPU store");
    KeepDrawInput(&recorder, address, stored, Use::Index32, 0xdead);
    Require(CopyDrawInput(context, &recorder, address, size, 4, Use::Index32).derived == 0xdead, "the new copy was not kept");
    MarkWritten(address + 128, 4);
    const auto marked = CopyDrawInput(context, &recorder, address, size, 4, Use::Index32);
    Require(!marked.reused && equalsGuest(marked), "a draw input outlived a stamped GPU store");
    KeepDrawInput(&recorder, address, marked, Use::Index32, 1);
    Require(CopyDrawInput(context, &recorder, address, size, 4, Use::Index32).reused, "the copy after the GPU store was not kept");
    alignas(64) static std::byte other[64];
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(other, sizeof(other), true, true, true);
    }
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(other);
    }
    Require(!CopyDrawInput(context, &recorder, address, size, 4, Use::Index32).reused, "a draw input outlived a registry mutation");
    const auto pool = address + 8192;
    const auto whole = CopyDrawInput(context, &recorder, pool, 12288, 1, Use::Vertex);
    Require(!whole.reused && std::memcmp(whole.buffer->Bytes().data(), reinterpret_cast<const void*>(pool), 12288) == 0, "the vertex pool copy is wrong");
    KeepDrawInput(&recorder, pool, whole, Use::Vertex, 0);
    const auto prefix = CopyDrawInput(context, &recorder, pool, 4096, 1, Use::Vertex);
    Require(prefix.reused && prefix.buffer == whole.buffer, "a shorter vertex read did not reuse the longer snapshot");
    Require(!CopyDrawInput(context, &recorder, pool, 16384, 1, Use::Vertex).reused, "a longer vertex read reused a shorter snapshot");
    Require(!CopyDrawInput(context, &recorder, pool + 4, 4096, 1, Use::Vertex).reused, "a vertex read at another address reused the snapshot");
    Require(!CopyDrawInput(context, &recorder, pool, 4096, 4, Use::Index32).reused, "an index read reused a longer vertex snapshot");
    const auto shorter = CopyDrawInput(context, &recorder, pool, 8192, 4, Use::Index32);
    KeepDrawInput(&recorder, pool, shorter, Use::Index32, 3);
    Require(!CopyDrawInput(context, &recorder, pool, 4096, 4, Use::Index32).reused, "an index read reused a longer index snapshot");
    words[(8192 + 8192 + 16) / 4] = 0xbeef;
    const auto prefixAfter = CopyDrawInput(context, &recorder, pool, 4096, 1, Use::Vertex);
    Require(std::memcmp(prefixAfter.buffer->Bytes().data(), reinterpret_cast<const void*>(pool), 4096) == 0, "a shorter vertex read after a store got other bytes");
    const auto after = CopyDrawInput(context, &recorder, pool, 12288, 1, Use::Vertex);
    Require(!after.reused && std::memcmp(after.buffer->Bytes().data(), reinterpret_cast<const void*>(pool), 12288) == 0, "a vertex read over the store reused the old bytes");
    KeepDrawInput(&recorder, pool, after, Use::Vertex, 0);
    const auto small = CopyDrawInput(context, &recorder, pool, 4096, 1, Use::Vertex);
    Require(small.reused && small.buffer == after.buffer, "the new vertex snapshot does not serve shorter reads");
    recorder.Sync();
}

void drawInputInPlaceTests(const Device& device, Recorder& recorder, bool copied = false) {
    auto context = device.GetContext();
    constexpr std::size_t bytes = 65536;
    void* block = context.hostImportAlignment != 0 ? AllocateWatched(bytes, 65536) : nullptr;
    if (block == nullptr) {
        std::cout << "host imports or write watching unavailable: in-place draw inputs not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } unregister{context, block};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the watched block refused: in-place draw inputs not tested\n";
        return;
    }
    context.recorder = &recorder;
    recorder.Sync();
    const auto vertices = address + 4096;
    context.drawInputsInPlace = !copied;
    if (copied) {
        Require(InPlaceDrawInput(context, vertices, 256, 4) == nullptr, "an imported draw input was bound in place where draw inputs are copied");
        Require(!recorder.PendingReadOverlaps(vertices, 256), "a copied draw input's read was noted as an in-place read");
        return;
    }
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_DRAW_INPUTS_VRAM=1|2, ctest agc_draw_inputs_in_place_vram1
    // and _vram2): InPlaceDrawInput refuses every range without a pending GPU write (the draw binds
    // a copy) and still binds a vertex range under one in place, behind it. Unset or 0 such a free
    // range binds in place; `freeRange` checks whichever the switch expects.
    const bool refused = DrawInputsVram() != 0;
    const auto freeRange = [refused](const HostImport* bound) { return refused ? bound == nullptr : bound != nullptr; };
    if (refused) std::cout << "APS5_EXPERIMENT_DRAW_INPUTS_VRAM=" << DrawInputsVram() << ": draw inputs free of pending GPU writes are expected to be refused\n";
    const auto* import = refused ? HostImportFor(context, vertices, 256) : InPlaceDrawInput(context, vertices, 256, 4);
    if (refused) {
        Require(import != nullptr && InPlaceDrawInput(context, vertices, 256, 4) == nullptr && !recorder.PendingReadOverlaps(vertices, 256), "an imported draw input was bound in place under APS5_EXPERIMENT_DRAW_INPUTS_VRAM");
    } else {
        Require(import != nullptr && import->base <= vertices && vertices + 256 <= import->base + import->bytes, "an imported draw input was not bound in place");
        const auto read = recorder.DescribePendingRead(vertices, 256);
        Require(read.has_value() && read->kind == Recorder::ReadKind::DrawInput && read->open, "an in-place draw input's read was not noted on the open batch");
    }
    Require(InPlaceDrawInput(context, vertices + 2, 256, 4) == nullptr, "a draw input misaligned for its format was bound in place");
    Require(freeRange(InPlaceDrawInput(context, vertices + 2, 256, 2)), refused ? "an aligned 16-bit draw input was bound in place under APS5_EXPERIMENT_DRAW_INPUTS_VRAM" : "an aligned 16-bit draw input was not bound in place");
    recorder.NotePendingWrite(address + 8192, 16);
    const auto* afterWrite = InPlaceDrawInput(context, address + 8192, 256, 4);
    if (refused) {
        // Under the switch an index range is refused before any check: the flush that completes the
        // write is its copy's (CopyDrawInput runs the flush hook first), made here as the draw would.
        Require(afterWrite == nullptr && recorder.PendingWriteOverlaps(address + 8192, 256), "an index range under APS5_EXPERIMENT_DRAW_INPUTS_VRAM was bound in place or ran the flush hook");
        AgcDriver::GuestMemory::FlushGpuWrites(address + 8192, 256);
    }
    Require(freeRange(afterWrite) && !recorder.PendingWriteOverlaps(address + 8192, 256), refused ? "a draw input after a completed GPU write was bound in place under APS5_EXPERIMENT_DRAW_INPUTS_VRAM, or the write was not completed" : "a draw input under a pending GPU write was bound before the write completed");
    bool behind = false;
    Require(freeRange(InPlaceDrawInput(context, address + 20480, 256, 4, &behind)) && !behind, refused ? "a vertex input free of pending GPU writes was bound in place under APS5_EXPERIMENT_DRAW_INPUTS_VRAM or marked as ordered behind one" : "a vertex input free of pending GPU writes was marked as ordered behind one");
    recorder.NotePendingWrite(address + 20480, 16);
    Require(InPlaceDrawInput(context, address + 20480, 256, 4, &behind) != nullptr && behind && recorder.PendingWriteOverlaps(address + 20480, 256), "a vertex input under a pending GPU write was not bound in place behind it");
    const std::array<std::byte, 4> label{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    recorder.RecordStore(import->buffer, address + 12288 - import->base, label, address + 12288);
    Require(recorder.QueuedStoreOverlaps(address + 12288, 4), "the label store was not queued");
    Require(InPlaceDrawInput(context, address + 12288, 256, 4) == nullptr, "a draw input under a queued label store was bound in place");
    behind = false;
    Require(InPlaceDrawInput(context, address + 12288, 256, 4, &behind) == nullptr, "a vertex input under a queued label store was bound in place");
    Require(freeRange(InPlaceDrawInput(context, address + 16384, 256, 4)), refused ? "a disjoint draw input beside a queued label store was bound in place under APS5_EXPERIMENT_DRAW_INPUTS_VRAM" : "a queued label store refused a disjoint draw input");
    alignas(64) static std::array<std::byte, 256> unregistered{};
    Require(InPlaceDrawInput(context, reinterpret_cast<std::uint64_t>(unregistered.data()), unregistered.size(), 4) == nullptr, "memory without a host import was bound in place");
    auto unrecorded = context;
    unrecorded.recorder = nullptr;
    Require(InPlaceDrawInput(unrecorded, vertices, 256, 4) == nullptr, "a draw without a recorder bound its input in place");
    recorder.Sync();
    Require(!recorder.PendingReadOverlaps(vertices, 256), "an in-place draw input's read outlived its batch");
    if (!AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "host imports are compared, not watched: kept highest indices not tested\n";
        return;
    }
    const auto indices = address + 20480;
    constexpr std::size_t kept = Recorder::KeptIndexRangeIndices * 2;
    std::vector<std::uint16_t> first(Recorder::KeptIndexRangeIndices * 2, 1);
    first[0] = 3;
    first[1] = 9;
    first[2] = 0xffff;
    std::memcpy(reinterpret_cast<void*>(indices), first.data(), kept * 2);
    auto range = InPlaceIndexRange(recorder, indices, kept, 2, false);
    Require(!range.reused && range.highest == 0xffffu, "an in-place index range was not scanned at its first use");
    range = InPlaceIndexRange(recorder, indices, kept, 2, false);
    Require(range.reused && range.highest == 0xffffu, "an unchanged in-place index range was scanned again");
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(!range.reused && range.highest == 9u, "an in-place index range served a scan made under the other restart rule");
    Require(!InPlaceIndexRange(recorder, indices, kept + 2, 2, true).reused, "an in-place index range served a scan of another size");
    const std::uint16_t raised = 12;
    std::memcpy(reinterpret_cast<void*>(indices + 2), &raised, sizeof(raised));
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range kept its highest index over a guest write");
    range = InPlaceIndexRange(recorder, indices, kept, 2, true);
    Require(range.reused && range.highest == 12u, "a rescanned in-place index range was not kept");
    range = InPlaceIndexRange(recorder, indices, kept - 2, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range below the kept size was not scanned");
    range = InPlaceIndexRange(recorder, indices, kept - 2, 2, true);
    Require(!range.reused && range.highest == 12u, "an in-place index range below the kept size was kept");
    range = InPlaceIndexRange(recorder, indices, kept * 2 - 4, 4, false);
    Require(!range.reused && range.highest == 0x000c0003u, "a 32-bit in-place index range below the kept size was not scanned");
    Require(!InPlaceIndexRange(recorder, indices, kept * 2 - 4, 4, false).reused, "a 32-bit in-place index range below the kept size was kept");
    Require(!InPlaceIndexRange(recorder, indices, kept * 2, 4, false).reused, "a 32-bit in-place index range was not scanned at its first use");
    range = InPlaceIndexRange(recorder, indices, kept * 2, 4, false);
    Require(range.reused && range.highest == 0x000c0003u, "an unchanged 32-bit in-place index range was scanned again");
    const std::vector<std::uint16_t> restarts(Recorder::KeptIndexRangeIndices, 0xffff);
    std::memcpy(reinterpret_cast<void*>(indices + 20480), restarts.data(), kept);
    Require(!InPlaceIndexRange(recorder, indices + 20480, kept, 2, true).highest.has_value(), "an index range of restart indices has a highest index");
    range = InPlaceIndexRange(recorder, indices + 20480, kept, 2, true);
    Require(range.reused && !range.highest.has_value(), "an empty in-place index range was scanned again");
}

// Unit shadows (UnitShadow.hpp) over a host import of write-watched arena memory: a retile piece's
// slab destination and its seeds, freshness from the tracker (a publish never stamps, a CPU write
// makes the unit stale), the scopes, the slab boundary, and the retire publish. With
// APS5_UNIT_SHADOW_MIB=8 (one slab) the second slab's allocation evicts the first with a publish;
// with APS5_NO_UNIT_SHADOW=1 every primitive is inert.
bool unitShadowTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    if (!UnitShadowEnabled()) {
        Require(!AnyShadowedOverlaps(0x10000, 16) && PublishShadow(0x10000, 16, PublishScope::Whole, PublishReason::Hook) == 0, "unit shadows are off but not inert");
        std::cout << "unit shadows off (APS5_NO_UNIT_SHADOW, or no write watching): primitives inert\n";
        return true;
    }
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: unit shadows not tested\n";
        return false;
    }
    constexpr std::uint64_t unit = 65536;
    // Three slabs at the default 8 MiB: units 0..127, 128..255, 256..271.
    constexpr std::size_t bytes = (17u << 20u);
    void* block = AllocateWatched(bytes, 65536);
    Require(block != nullptr, "unit shadows are on without write watching");
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        bool armed = true;
        ~Unregister() {
            if (!armed) return;
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the shadow test block refused: unit shadows not tested\n";
        return false;
    }
    if (!Watched(address, bytes)) {
        std::cout << "host imports are compared, not watched: unit shadows not tested\n";
        return false;
    }
    Require(import->base == address && import->bytes == bytes, "the import does not cover the block");
    CollectWritesUncached(address, bytes);
    const auto* words = static_cast<const std::uint8_t*>(block);
    const auto unit3 = address + 3 * unit;
    const auto unit4 = unit3 + unit;
    const auto unit5 = unit4 + unit;
    // A half-unit piece seeds its unit from the import; a whole-unit piece does not; both land in
    // one slab at one offset.
    auto half = ShadowDestinationFor(context, *import, unit3, unit3 + unit / 2);
    Require(half.has_value() && half->seedUnits.size() == 1 && half->seedUnits[0].first == unit3 && half->seedUnits[0].second == unit4, "a half-unit retile piece does not seed its unit");
    auto whole = ShadowDestinationFor(context, *import, unit3, unit4);
    Require(whole.has_value() && whole->seedUnits.empty() && whole->slab == half->slab && whole->offset == half->offset && whole->buffer == half->buffer, "a whole-unit retile piece seeds, or lands elsewhere");
    Require(!AnyShadowedOverlaps(unit3, unit), "a destination alone counts as shadowed");
    const auto firstSlab = half->slab;
    Require(firstSlab->pins.load() == 2 && half->pin != nullptr, "destinations do not pin their slab");
    // The retile as writeBackWindows records it: the seed, then the piece's bytes over its half.
    const auto copyBuffer = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    const auto retile = [&](const ShadowDestination& destination, std::uint64_t begin, std::uint64_t length, std::uint8_t value, bool seed) {
        auto pattern = std::make_shared<Buffer>(context, static_cast<std::size_t>(length), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memset(pattern->Bytes().data(), value, pattern->Bytes().size());
        const auto commands = recorder.Commands();
        recorder.Keep(pattern);
        recorder.Keep(destination.slab);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        if (seed) {
            const VkBufferCopy seedCopy{begin - import->base, SlabOffset(*import, *destination.slab, begin), unit};
            copyBuffer(commands, import->buffer, destination.buffer, 1, &seedCopy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        const VkBufferCopy piece{0, SlabOffset(*import, *destination.slab, begin), length};
        copyBuffer(commands, pattern->Handle(), destination.buffer, 1, &piece);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        recorder.MarkCovered(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const ShadowedRange range{begin & ~(unit - 1), (begin & ~(unit - 1)) + unit, destination.slab};
        MarkShadowed(*import, std::span(&range, 1), TrackerGeneration());
    };
    retile(*half, unit3, unit / 2, 0x22, true);
    Require(AnyShadowedOverlaps(unit3, 1) && AnyShadowedOverlaps(unit4 - 1, 1) && AnyShadowedOverlaps(unit3 + 1000, 64), "a fresh unit is not seen as shadowed");
    Require(!AnyShadowedOverlaps(unit3 - 1, 1) && !AnyShadowedOverlaps(unit4, unit) && !AnyShadowedOverlaps(address, 3 * unit), "a range outside the unit is seen as shadowed");
    {
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 3> outside{{{unit3 - 8, unit3}, {unit4, unit4 + unit}, {address, address + 3 * unit}}};
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 3> lastInside{{{unit3 - 8, unit3}, {unit4, unit4 + 8}, {unit4 - 4, unit4}}};
        Require(!AnyShadowedOverlaps(outside) && AnyShadowedOverlaps(lastInside), "a list of runs is not shadowed exactly when one of its runs is");
    }
    {
        // An upload's runs over units 2..4 split at unit 3's freshness; the fresh piece reads the slab.
        const std::pair<std::uint64_t, std::uint64_t> run{2 * unit, 5 * unit};
        const auto sources = ShadowSources(context, *import, address, std::span(&run, 1), {});
        Require(sources.size() == 3, "an upload run is not split at the fresh unit");
        Require(!sources[0].shadow && sources[0].begin == 2 * unit && sources[0].end == 3 * unit && sources[0].buffer == import->buffer && sources[0].offset == 2 * unit, "the import piece before the fresh unit is wrong");
        Require(sources[1].shadow && sources[1].begin == 3 * unit && sources[1].end == 4 * unit && sources[1].buffer == half->buffer && sources[1].offset == half->offset && sources[1].slab == half->slab, "the fresh unit's piece does not read the slab");
        Require(!sources[2].shadow && sources[2].begin == 4 * unit && sources[2].end == 5 * unit, "the import piece after the fresh unit is wrong");
    }
    // A whole publish: the untouched half keeps the seed (the import's bytes), the other the
    // retile's; the copy stamps nothing, and a second publish copies nothing.
    const auto pre = CollectWritesUncached(address, bytes);
    Require(PublishShadow(unit3, unit, PublishScope::Whole, PublishReason::Hook) == 1, "the whole publish did not copy the unit");
    Require(recorder.PendingWriteOverlaps(unit3, unit), "the publish noted no pending write");
    recorder.Sync();
    Require(words[3 * unit] == 0x22 && words[3 * unit + unit / 2 - 1] == 0x22 && words[3 * unit + unit / 2] == 0x11 && words[4 * unit - 1] == 0x11 && words[3 * unit - 1] == 0x11 && words[4 * unit] == 0x11, "the published bytes are wrong");
    Require(!AnyShadowedOverlaps(unit3, unit), "a published unit still counts as shadowed");
    Require(PublishShadow(unit3, unit, PublishScope::Whole, PublishReason::Hook) == 0, "a second publish copied the unit again");
    Require(UnchangedSince(unit3, unit, pre), "the publish stamped the unit");
    {
        // The published unit still reads from the slab (both hold the bytes).
        const std::pair<std::uint64_t, std::uint64_t> run{3 * unit, 4 * unit};
        const auto sources = ShadowSources(context, *import, address, std::span(&run, 1), {});
        Require(sources.size() == 1 && sources[0].shadow, "a published unit does not read from the slab");
    }
    // Partial-unit scope over unit 3 (partly, fresh again) and unit 4 (wholly, fresh): unit 3 only.
    retile(*whole, unit3, unit, 0x33, false);
    auto four = ShadowDestinationFor(context, *import, unit4, unit5);
    Require(four.has_value() && four->seedUnits.empty() && four->slab == half->slab, "unit 4's destination is not in the same slab");
    retile(*four, unit4, unit, 0x44, false);
    Require(AnyShadowedOverlaps(unit3, unit) && AnyShadowedOverlaps(unit4, unit), "re-marked units are not shadowed");
    Require(PublishShadow(unit3 + 100, static_cast<std::size_t>(2 * unit - 100), PublishScope::PartialUnits, PublishReason::Fill) == 1, "the partial publish did not copy exactly the partly covered unit");
    Require(!AnyShadowedOverlaps(unit3, unit) && AnyShadowedOverlaps(unit4, unit), "the partial publish copied the wrong unit");
    recorder.Sync();
    Require(words[3 * unit] == 0x33 && words[4 * unit - 1] == 0x33 && words[4 * unit] == 0x11, "the partial publish's bytes are wrong");
    // A CPU write into a page of unit 4 makes it stale: the publish collects the unit itself (no
    // walk saw the store before it), copies nothing and drops it.
    static_cast<void>(CollectWritesUncached(unit4, unit));
    *static_cast<volatile std::uint8_t*>(static_cast<void*>(static_cast<std::uint8_t*>(block) + 4 * unit + 4096)) = 0x55;
    Require(PublishShadow(unit4, unit, PublishScope::Whole, PublishReason::Hook) == 0, "a stale unit was published");
    Require(!AnyShadowedOverlaps(unit4, unit), "a stale unit still counts as shadowed after its publish");
    recorder.Sync();
    Require(words[4 * unit + 4096] == 0x55 && words[4 * unit] == 0x11, "a stale unit's publish overwrote the CPU's bytes");
    // The slab boundary: a piece over units 127 and 128 is refused, each alone lands in its slab.
    const auto unit127 = address + 127 * unit;
    const auto unit128 = unit127 + unit;
    Require(SlabBoundary(*import, unit127) == unit128 && SlabBoundary(*import, address) == unit128 && SlabBoundary(*import, unit128) == unit128 + 128 * unit, "the slab boundary is wrong");
    Require(!ShadowDestinationFor(context, *import, unit127, unit128 + unit).has_value(), "a piece across a slab boundary was accepted");
    auto low = ShadowDestinationFor(context, *import, unit127, unit128);
    Require(low.has_value() && low->slab == firstSlab && low->offset == 127 * unit, "unit 127 does not land in the first slab");
    retile(*low, unit127, unit, 0x66, false);
    auto high = ShadowDestinationFor(context, *import, unit128, unit128 + unit);
    const char* budgetText = std::getenv("APS5_UNIT_SHADOW_MIB");
    const auto budgetMiB = budgetText != nullptr ? std::strtoull(budgetText, nullptr, 10) : 1024ull;
    if (budgetMiB < 16) {
        // One slab fits: while destinations pin the first slab the second is refused (a
        // write-back in progress must keep its slab); with the pins released, the second slab's
        // allocation evicts the first after publishing its fresh unit 127 (the eviction publish is
        // recorded, so the bytes land at the sync).
        Require(!high.has_value(), "a pinned slab was evicted for a second slab");
        half.reset();
        whole.reset();
        four.reset();
        low.reset();
        Require(firstSlab->pins.load() == 0, "a released destination left its pin");
        high = ShadowDestinationFor(context, *import, unit128, unit128 + unit);
        Require(high.has_value() && high->slab != firstSlab && high->offset == 0, "unit 128 does not land in a second slab under the one-slab budget");
        recorder.Sync();
        Require(words[127 * unit] == 0x66 && !AnyShadowedOverlaps(unit127, unit), "the evicted slab's fresh unit was not published");
        std::cout << "unit shadow eviction under APS5_UNIT_SHADOW_MIB=" << budgetMiB << " verified\n";
    } else {
        Require(high.has_value() && high->slab != firstSlab && high->offset == 0 && AnyShadowedOverlaps(unit127, unit), "unit 128 does not land in a second slab beside the first");
        half.reset();
        whole.reset();
        four.reset();
        low.reset();
    }
    retile(*high, unit128, unit, 0x77, false);
    Require(AnyShadowedOverlaps(unit128, unit), "the second slab's unit is not shadowed");
    high.reset();
    retile(*ShadowDestinationFor(context, *import, unit5, unit5 + unit), unit5, unit, 0x88, false);
    retile(*ShadowDestinationFor(context, *import, address + 2 * unit, unit3), address + 2 * unit, unit, 0x99, false);
    Require(AnyShadowedOverlaps(unit5, unit) && AnyShadowedOverlaps(address + 2 * unit, unit), "units 2 and 5 are not shadowed before the retire");
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        mutation.Add(block, static_cast<std::size_t>(5 * unit), true, true, true);
    }
    Require(HostImportFor(context, address, bytes) == nullptr, "a shrunk registration still imports the old range");
    Require(!AnyShadowedOverlaps(address, bytes), "the retired import's shadow survived");
    recorder.Sync();
    Require(words[2 * unit] == 0x99 && words[3 * unit - 1] == 0x99, "the retire did not publish the unit still registered");
    Require(words[5 * unit] == 0x88 && words[5 * unit + unit - 1] == 0x88, "the retire did not publish a unit covered by the old registration");
    Require(words[127 * unit] == 0x66 && words[128 * unit] == 0x77, "the retire did not publish the second slab's units covered by the old registration");
    return true;
}

void idlePublishWaitTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    constexpr std::uint64_t unit = 65536;
    constexpr std::size_t bytes = 8 * unit;
    void* block = AllocateWatched(bytes, 65536);
    Require(block != nullptr, "unit shadows are on without write watching");
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr || !Watched(address, bytes)) {
        std::cout << "host import of the idle publish test block refused or not watched: idle publish wait not tested\n";
        return;
    }
    recorder.Sync();
    CollectWritesUncached(address, bytes);
    auto* words = static_cast<std::uint8_t*>(block);
    const auto copyBuffer = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    const auto retile = [&](std::uint64_t begin, std::uint8_t value) {
        auto destination = ShadowDestinationFor(context, *import, begin, begin + unit);
        Require(destination.has_value() && destination->seedUnits.empty(), "a whole-unit retile piece has no destination");
        auto pattern = std::make_shared<Buffer>(context, static_cast<std::size_t>(unit), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memset(pattern->Bytes().data(), value, pattern->Bytes().size());
        const auto commands = recorder.Commands();
        recorder.Keep(pattern);
        recorder.Keep(destination->slab);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkBufferCopy piece{0, destination->offset, unit};
        copyBuffer(commands, pattern->Handle(), destination->buffer, 1, &piece);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        recorder.MarkCovered(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const ShadowedRange range{begin, begin + unit, destination->slab};
        MarkShadowed(*import, std::span(&range, 1), TrackerGeneration());
        Require(AnyShadowedOverlaps(begin, unit), "the retiled unit is not shadowed");
    };
    retile(address + 2 * unit, 0x22);
    retile(address + 3 * unit, 0x33);
    StorageTexture::FlushAllPending("idle");
    Require(!AnyShadowedOverlaps(address + 2 * unit, static_cast<std::size_t>(2 * unit)), "the idle flush did not publish the shadowed units");
    Require(!recorder.PendingWriteOverlaps(address + 2 * unit, static_cast<std::size_t>(2 * unit)), "the idle flush returned before its publish's copies completed");
    words[2 * unit] = 0x44;
    words[4 * unit - 1] = 0x45;
    recorder.Sync();
    Require(words[2 * unit] == 0x44 && words[4 * unit - 1] == 0x45, "a CPU store made after the idle flush returned was overwritten by its publish");
    Require(words[2 * unit + 1] == 0x22 && words[3 * unit - 1] == 0x22 && words[3 * unit] == 0x33 && words[1 * unit] == 0x11 && words[4 * unit] == 0x11, "the idle publish's bytes are wrong");
    retile(address + 5 * unit, 0x55);
    StorageTexture::FlushAllPending("label");
    Require(recorder.PendingWriteOverlaps(address + 5 * unit, unit), "a publish of a flush other than the idle one was waited");
    recorder.Sync();
    Require(words[5 * unit] == 0x55 && words[6 * unit - 1] == 0x55, "the label flush's publish did not land");
    std::cout << "idle publish wait verified\n";
}

void storageRefreshTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: storage refresh not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = nullptr;
    if (watched) {
        block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: storage refresh in watched memory not tested\n";
            return;
        }
    } else {
#ifdef _WIN32
        block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        block = std::aligned_alloc(65536, bytes);
#endif
    }
    Require(block != nullptr, "cannot allocate the storage refresh block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    const auto* import = HostImportFor(base, address, bytes);
    if (import == nullptr) {
        std::cout << "host import of the storage refresh block refused: storage refresh not tested\n";
        return;
    }
    if (watched && !AgcDriver::GuestMemory::Watched(address, bytes)) std::cout << "host imports are compared, not watched: storage refresh in watched memory runs as unwatched\n";
    watched = watched && AgcDriver::GuestMemory::Watched(address, bytes);
    Require(watched || !ShadowDestinationFor(base, *import, address, address + 65536).has_value(), "a unit shadow was offered for memory the write tracker does not watch");
    if (!AgcDriver::GuestMemory::WriteWatched()) Require(!UnitShadowEnabled(), "unit shadows are on without write watching");
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    Require(DescribeSurface(resource).guestBytes == surfaceBytes, "the test surface has an unexpected size");
    DccKeyProof proof;
    Require(ProvedClearKeys(resource, surfaceBytes, proof) == DccKeys::Clear0000, "0x00 keys are not a 0000 clear");
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto draw = [&](VkClearColorValue value) {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        };
        const auto holds = [&](std::array<std::uint8_t, 4> texel) {
            Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                if (std::to_integer<std::uint8_t>(pixels[i]) != texel[i % 4]) return false;
            }
            return true;
        };
        const auto memoryHolds = [&](std::array<std::uint8_t, 4> texel, std::size_t first = 0, std::uint8_t other = 0) {
            std::vector<std::byte> read(surfaceBytes);
            if (watched) AgcDriver::GuestMemory::Read(address, read);
            else std::memcpy(read.data(), texels, surfaceBytes);
            for (std::size_t i = 0; i < surfaceBytes; ++i) {
                if (std::to_integer<std::uint8_t>(read[i]) != (i < first ? other : texel[i % 4])) return false;
            }
            return true;
        };
        const auto holdsMixed = [&](std::array<std::uint8_t, 4> texel, std::size_t count, std::uint8_t other) {
            Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            std::size_t others = 0;
            for (std::size_t i = 0; i < pixels.size(); i += 4) {
                bool isOther = true, isTexel = true;
                for (std::size_t c = 0; c < 4; ++c) {
                    const auto value = std::to_integer<std::uint8_t>(pixels[i + c]);
                    isOther = isOther && value == other;
                    isTexel = isTexel && value == texel[c];
                }
                if (isOther) ++others;
                else if (!isTexel) return false;
            }
            return others == count;
        };
        Require(holds({0, 0, 0, 0}), "a surface under 0000 clear keys was not cleared");
        draw({{1.0f, 0.0f, 0.0f, 1.0f}});
        image->Refresh();
        Require(holds({255, 0, 0, 255}), "a refresh under the image's own clear keys dropped the draw's results");
        if (watched) {
            Require(memoryHolds({255, 0, 0, 255}), "the refresh's store of the results did not reach watched guest memory");
        } else {
            Require(StorageTexture::FindPending(address, surfaceBytes) == image, "an unchanged untracked refresh discarded the pending image");
            Require(StorageTexture::FlushPending(address, surfaceBytes, nullptr, "storage refresh visibility test"), "the pending untracked image was not flushed");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(memoryHolds({255, 0, 0, 255}), "an explicit flush did not make pending results visible in untracked guest memory");
        }
        if (watched) {
            draw({{0.0f, 0.0f, 1.0f, 1.0f}});
            std::memset(texels, 0x33, 65536);
            image->Refresh();
            Require(holdsMixed({0, 0, 255, 255}, 65536 / 4, 0x33), "a CPU store into a unit with results pending was not seen by the refresh");
            Require(memoryHolds({0, 0, 255, 255}, 65536, 0x33), "guest memory lost the CPU store or the other units' results");
            image->Refresh();
            Require(holdsMixed({0, 0, 255, 255}, 65536 / 4, 0x33), "an unchanged surface changed at a refresh");
        }
        draw({{0.0f, 1.0f, 0.0f, 1.0f}});
        std::memset(keys, 0x40, keyCount);
        image->Refresh();
        Require(holds({0, 0, 0, 255}), "new 0001 clear keys did not clear the image");
        Require(watched ? memoryHolds({0, 0, 255, 255}, 65536, 0x33) : memoryHolds({255, 0, 0, 255}), "results dead under new clear keys were stored");
    }
    recorder.Sync();
}

bool writtenBufferOrderTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: written buffer order not tested\n";
        return false;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t bytes = side * side * 4;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: written buffer order not tested\n";
        return false;
    }
    std::memset(block, 0x55, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the written buffer order block refused: written buffer order not tested\n";
        return false;
    }
    std::cout << "written buffer order: host imports are " << (AgcDriver::GuestMemory::Watched(address, bytes) ? "watched" : "compared") << "\n";
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    Require(DescribeSurface(resource).guestBytes == bytes, "the written buffer order surface has an unexpected size");
    const auto settle = [&] {
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    };
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
        {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        }
        constexpr std::size_t written = 40800;
        GuestBufferMemory memory(context);
        memory.AcquireRegistered();
        memory.AddWritable(address, written, false, false);
        memory.Upload(true);
        std::uint32_t adjustment = 0;
        const auto view = memory.Descriptor(address, written, adjustment);
        Require(adjustment == 0, "the written buffer view is not aligned to its element");
        {
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, view.buffer, view.offset, written, 0xa5a5a5a5u);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        memory.RecordCopyBacks(recorder);
        memory.MarkDirectWrites();
        settle();
        StorageTexture::FlushAllPending("written buffer order test");
        settle();
        std::vector<std::byte> read(bytes);
        AgcDriver::GuestMemory::Read(address, read);
        Require(std::all_of(read.begin(), read.begin() + written, [](std::byte value) { return value == std::byte{0xa5}; }), "storage results pending under a buffer a later use wrote were stored over the buffer's contents");
        for (std::size_t i = written; i < bytes; ++i) Require(std::to_integer<std::uint8_t>(read[i]) == (i % 4 == 1 || i % 4 == 2 ? 0 : 255), "the storage results outside the written buffer did not reach guest memory");
    }
    recorder.Sync();
    return true;
}

void hostImportUnmapTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: import unmap lifetime not tested\n";
        return;
    }
    const auto alignment = static_cast<std::size_t>(std::max<VkDeviceSize>(65536u, context.hostImportAlignment));
    constexpr std::size_t RequestedBytes = 17u << 20u;
    const auto bytes = ((RequestedBytes + alignment - 1u) / alignment) * alignment;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(alignment, bytes);
#endif
    Require(block != nullptr, "cannot allocate the host import unmap block");
    struct Block {
        void* address;
        std::size_t bytes;
        bool registered = true;
        bool released = false;
        void Release() {
#ifdef _WIN32
            Require(VirtualFree(address, 0, MEM_RELEASE) != 0, "cannot release the host import unmap block");
#else
            std::free(address);
#endif
            released = true;
        }
        ~Block() {
            if (registered) {
                try {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(address);
                } catch (...) {
                    return;
                }
            }
            if (!released) {
#ifdef _WIN32
                VirtualFree(address, 0, MEM_RELEASE);
#else
                std::free(address);
#endif
            }
        }
    } allocation{block, bytes};
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    const HostImport* imported = nullptr;
    {
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(context, address, bytes);
    }
    if (imported == nullptr) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        allocation.registered = false;
        allocation.Release();
        std::cout << "host import refused: import unmap lifetime not tested\n";
        return;
    }
    bool retiredBeforeRelease = false;
    {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(block, bytes, [&](const void*, std::size_t, const void*, bool) {
            retiredBeforeRelease = !HostImportCovers(context, address, bytes);
            if (retiredBeforeRelease) allocation.Release();
        });
        allocation.registered = false;
    }
    if (!retiredBeforeRelease) {
        {
            std::lock_guard gpu(GpuMutex());
            static_cast<void>(HostImportFor(context, address, bytes));
            recorder.Sync();
        }
        allocation.Release();
    }
    Require(retiredBeforeRelease, "guest unmap released pages before retiring their host import");
    std::cout << "host import unmap lifetime: ok\n";
}

void parkedImportUnmapTests(const Device& device) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: parked import unmap not tested\n";
        return;
    }
    const auto alignment = static_cast<std::size_t>(std::max<VkDeviceSize>(65536u, context.hostImportAlignment));
    const auto bytes = alignment * 4u;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(alignment, bytes);
#endif
    Require(block != nullptr, "cannot allocate the parked import unmap block");
    const auto release = [&] {
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    };
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    std::unique_ptr<Device> replacement;
    bool imported = false;
    {
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(context, address, bytes) != nullptr;
    }
    if (imported) {
        replacement = std::make_unique<Device>();
        std::lock_guard gpu(GpuMutex());
        imported = HostImportFor(replacement->GetContext(), address, bytes) != nullptr;
    }
    if (!imported) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        if (replacement) ClearHostImports(replacement->GetContext().device);
        release();
        std::cout << "host import refused: parked import unmap not tested\n";
        return;
    }
    bool unmapped = false;
    try {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(block, bytes, [&](const void*, std::size_t, const void*, bool) { unmapped = true; });
    } catch (const std::exception& error) {
        ClearHostImports(replacement->GetContext().device);
        ClearHostImports(context.device);
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
        release();
        throw std::runtime_error(std::string("guest unmap after a device replacement: ") + error.what());
    }
    ClearHostImports(replacement->GetContext().device);
    ClearImageMirrors(replacement->GetContext().device);
    replacement.reset();
    release();
    Require(unmapped, "guest unmap of a range a replaced device imported did not run");
    std::cout << "parked import unmap: ok\n";
}

void importWatchTests(const Device& device) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
#ifdef _WIN32
    Require(PrepareImportWatch(context) == ImportWatch::Unwatch, "Windows host imports stayed watched by default");
    std::cout << "import watch decisions: Windows host imports through views use comparisons\n";
    if (context.hostImportAlignment == 0 || !WriteWatched()) {
        std::cout << "host imports or write watching unavailable: the Windows import probe not tested\n";
        return;
    }
    if (context.settleQueue == VK_NULL_HANDLE || context.settleQueueMutex == nullptr) {
        Require(PrepareImportWatch(context) == ImportWatch::Unwatch, "(w) a device without a settle queue kept host imports watched");
        std::cout << "no settle queue: the Windows import probe skipped\n";
        return;
    }
    const auto probe = ProbeImportWriteProtection(context);
    Require(probe.failure == nullptr, std::string("(w) the Windows import probe failed at ") + (probe.failure != nullptr ? probe.failure : "") + " (" + std::to_string(static_cast<int>(probe.result)) + ")");
    std::cout << "Windows import probe: " << probe.writtenAtImport << " of " << probe.pages << " scratch pages reported written at the import, " << probe.writtenDuringSettle << " by its 4-byte settle read, " << probe.writtenAfterSettle << " after it, " << probe.writtenAfterSubmit << " after a full GPU read, " << probe.writtenAfterSecondSubmit << " after a second, " << probe.writtenAfterFill << " after a GPU write, " << probe.writtenByCpu << " after a CPU store\n";
    Require(probe.writtenByCpu != 0, "(w) a CPU store into a host-imported MEM_WRITE_WATCH range was not collected");
    Require(probe.writtenAfterSettle == 0 || PrepareImportWatch(context) == ImportWatch::Unwatch, "(w) a settled import kept reporting writes and stayed watched");
#else
    if (context.hostImportAlignment == 0 || !WriteWatched()) {
        std::cout << "host imports or write watching unavailable: import watch decisions not tested\n";
        return;
    }
    const auto probe = ProbeImportWriteProtection(context);
    Require(probe.failure == nullptr, std::string("(u) the import probe failed at ") + (probe.failure != nullptr ? probe.failure : "") + " (" + std::to_string(static_cast<int>(probe.result)) + ")");
    std::cout << "import probe: " << probe.writtenAfterSubmit << " of " << probe.pages << " scratch pages written after a GPU read, " << probe.writtenAfterFill << " after a GPU write, " << probe.writtenAtImport << " after the import\n";
    Require(probe.writtenByCpu != 0, "(u) a CPU store after the import probe's GPU read was not collected");
    const int udmabuf = context.dmaBufImport ? open("/dev/udmabuf", O_RDWR | O_CLOEXEC) : -1;
    if (udmabuf >= 0) close(udmabuf);
    if (context.dmaBufImport && udmabuf < 0) {
        std::cout << "/dev/udmabuf is not accessible: the dma-buf import probe not tested\n";
    } else if (context.dmaBufImport) {
        const auto dmaBuf = ProbeDmaBufImportWriteProtection(context);
        Require(dmaBuf.failure == nullptr, std::string("(u) the dma-buf import probe failed at ") + (dmaBuf.failure != nullptr ? dmaBuf.failure : "") + " (" + std::to_string(static_cast<int>(dmaBuf.result)) + ")");
        Require(dmaBuf.writtenByCpu != 0, "(u) a CPU store into a dma-buf imported scratch range was not collected");
        std::cout << "dma-buf import probe: " << dmaBuf.writtenAfterSubmit << " of " << dmaBuf.pages << " scratch pages written after a GPU read, " << dmaBuf.writtenAtImport << " after the import, " << dmaBuf.writtenByCpu << " after a CPU store\n";
    } else {
        std::cout << "dma-buf imports unavailable: the dma-buf import probe not tested\n";
    }
    const auto decided = PrepareImportWatch(context);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{context, decided};
    constexpr std::size_t unit = 65536;
    const auto remap = [](void* block, std::size_t bytes) {
        munmap(block, bytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
        Require(mmap(block, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == block, "(u) cannot map the test block again");
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
    };
    const auto registerRange = [](void* block, std::size_t bytes) {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    };
    const auto unregisterRange = [](void* block) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(block);
    };
    SetImportWatch(context, ImportWatch::Unwatch);
    {
        constexpr std::size_t bytes = 3 * unit;
        void* block = AllocateWatched(bytes, unit);
        const auto address = reinterpret_cast<std::uint64_t>(block);
        std::memset(block, 0x11, bytes);
        const auto before = CollectWrites(address, bytes);
        Require(before != 0, "(u) a watched block is not collected");
        BumpCollectEpoch();
        Require(CollectWrites(address, 2 * unit) != 0, "(u) the memoized collect of a watched block failed");
        registerRange(block, 2 * unit);
        const auto* import = HostImportFor(context, address, 2 * unit);
        Require(import != nullptr && import->unwatched, "(u) an import made under the unwatch decision is not marked unwatched");
        Require(!Watched(address, 2 * unit) && !Watched(address + unit, 4096), "(u) an imported range stays watched");
        Require(CollectWrites(address, 2 * unit) == 0 && CollectWrites(address + 4096, 4096) == 0, "(u) a collect memoized before the import answers for the unwatched range");
        Require(CollectWrites(address + 2 * unit - 4096, 8192) == 0, "(u) a range partly imported is collected");
        Require(Watched(address + 2 * unit, unit), "(u) the unimported neighbour left the watch");
        const auto neighbour = CollectWrites(address + 2 * unit, unit);
        Require(neighbour != 0, "(u) the unimported neighbour is not collected");
        static_cast<volatile std::uint8_t*>(block)[4096] = 0x22;
        Require(!UnchangedSince(address, 2 * unit, before) && !UnchangedSince(address, 4096, TrackerGeneration()) && !UnchangedSinceCollected(address, 4096, TrackerGeneration()), "(u) an unwatched range reports unchanged");
        const std::array<std::uint64_t, 2> generations{before, TrackerGeneration()};
        std::array<std::uint8_t, 2> changed{};
        std::array<std::uint8_t, 2> cpu{};
        Require(!ChangedBlocks(address, 2 * unit, generations, changed, cpu) && changed[0] != 0 && changed[1] != 0, "(u) the stamps of an unwatched range are offered as tracked");
        Require(MarkWritten(address, unit) == 0, "(u) a GPU write into an unwatched range claims a generation");
        Require(MarkWritten(address + 2 * unit, 4096) != 0 && !UnchangedSince(address + 2 * unit, unit, neighbour), "(u) a GPU write into the watched neighbour is not stamped");
        unregisterRange(block);
        remap(block, 2 * unit);
        Require(Watched(address, 2 * unit), "(u) a remapped range is not watched");
        registerRange(block, 2 * unit);
        const auto* kept = HostImportFor(context, address, 2 * unit);
        Require(kept != nullptr && kept->unwatched && !Watched(address, 2 * unit) && CollectWrites(address, 2 * unit) == 0, "(u) an import kept over a remap left the range watched");
        unregisterRange(block);
        remap(block, 2 * unit);
        registerRange(block, unit);
        const auto* again = HostImportFor(context, address, unit);
        Require(again != nullptr && again->unwatched && again->bytes == unit && !Watched(address, unit), "(u) a re-import after an unmap did not follow the decision");
        Require(Watched(address + unit, unit) && CollectWrites(address + unit, unit) != 0, "(u) the unregistered rest of a remapped range is not watched");
        unregisterRange(block);
        Require(HostImportFor(context, address, unit) == nullptr, "(u) an unregistered range still imports");
        ReleaseWatched(block, bytes);
    }
    SetImportWatch(context, ImportWatch::Watch);
    {
        constexpr std::size_t bytes = unit;
        void* block = AllocateWatched(bytes, unit);
        const auto address = reinterpret_cast<std::uint64_t>(block);
        registerRange(block, bytes);
        const auto* import = HostImportFor(context, address, bytes);
        Require(import != nullptr && !import->unwatched && Watched(address, bytes), "(u) an import made under the watch decision left the watch");
        const auto generation = CollectWrites(address, bytes);
        Require(generation != 0 && UnchangedSince(address, bytes, generation), "(u) a watched import is not collected");
        Require(MarkWritten(address, 4096) != 0 && !UnchangedSince(address, bytes, generation), "(u) a GPU write into a watched import is not stamped");
        unregisterRange(block);
        HostImportFor(context, address, bytes);
        ReleaseWatched(block, bytes);
    }
#endif
}

void staleGenerationTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
#ifdef _WIN32
    static_cast<void>(recorder);
    static_cast<void>(base);
    std::cout << "a range leaving the watch: Linux write watch only\n";
#else
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: a range leaving the watch not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: a range leaving the watch not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    const auto decided = PrepareImportWatch(base);
    SetImportWatch(base, ImportWatch::Unwatch);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{base, decided};
    auto* texels = static_cast<std::uint8_t*>(block);
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(texels + surfaceBytes, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    {
        Require(AgcDriver::GuestMemory::Watched(address, bytes), "(s) the block is not watched before the storage image takes its generation");
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto commands = recorder.Commands();
        recorder.Keep(image);
        Require(image->Generation() != 0, "(s) the storage image took no generation while its memory was watched");
        const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        image->MarkDirty();
        const auto* import = HostImportFor(base, address, bytes);
        if (import == nullptr) {
            std::cout << "host import of the stale generation block refused: a range leaving the watch not tested\n";
            recorder.Sync();
            return;
        }
        Require(!AgcDriver::GuestMemory::Watched(address, bytes), "(s) the imported range stayed watched");
        image->Refresh();
        Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto copyCommands = recorder.Commands();
        RecordMemoryBarrier(context, copyCommands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {side, side, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(copyCommands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, copyCommands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        Require(StorageTexture::FindPending(address, surfaceBytes) == image, "(s) the image is not pending after the refresh");
        Require(StorageTexture::FlushPending(address, surfaceBytes, nullptr, "stale generation test"), "(s) the pending image was not flushed");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        const auto pixels = readback.Bytes();
        constexpr std::array<std::uint8_t, 4> expected{255, 0, 0, 255};
        for (std::size_t i = 0; i < pixels.size(); ++i) Require(std::to_integer<std::uint8_t>(pixels[i]) == expected[i % 4], "(s) results pending when their memory left the watch were dropped");
        for (std::size_t i = 0; i < surfaceBytes; ++i) Require(texels[i] == expected[i % 4], "(s) results pending when their memory left the watch did not reach it");
    }
    recorder.Sync();
#endif
}

// A write-back stores the texels of an image and leaves every other byte of the surface's memory
// as it was: the padding of the tile blocks the surface covers only partly (its last block column
// and row), the tail block of a mip chain, the bytes between the mips and the pitch padding of a
// linear surface. Another surface over the memory, or the CPU, reads those bytes. Each case clears
// every mip of an image over memory holding a pattern no texel equals, stores it and reads the
// memory back (watched memory through the flush hook, after a store into the import's unit shadow):
// each 4-byte word is the clear texel or the pattern, and the texels are exactly the surface's.
// The 64 KiB-aligned tiled cases store through block windows, the others whole layers.
void writeBackPaddingTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: write-back padding not tested\n";
        return;
    }
    struct Case {
        const char* name;
        TextureTileMode tileMode;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t mipCount;
        std::uint64_t offset;
        bool direct = false;
    };
    constexpr Case cases[] = {
        {"64 KiB R_X 200x150", TextureTileMode::kR64KBX, 200, 150, 1, 0},
        {"64 KiB R_X 200x150, 3 mips", TextureTileMode::kR64KBX, 200, 150, 3, 0},
        {"64 KiB R_X 520x260, 6 mips", TextureTileMode::kR64KBX, 520, 260, 6, 0, true},
        {"64 KiB R_X 200x150 off a 64 KiB boundary, 3 mips", TextureTileMode::kR64KBX, 200, 150, 3, 4096},
        {"4 KiB S 100x70, 2 mips", TextureTileMode::kStandard4KB, 100, 70, 2, 0},
        {"linear 200x150", TextureTileMode::kLinear, 200, 150, 1, 0},
        {"64 KiB R_X 512x512, 3 mips, no padding below the tail", TextureTileMode::kR64KBX, 512, 512, 3, 0, true},
        {"64 KiB R_X 1000x700", TextureTileMode::kR64KBX, 1000, 700, 1, 0, true},
        {"64 KiB R_X 1900x1100, across two 8 MiB slabs", TextureTileMode::kR64KBX, 1900, 1100, 1, 0, true},
    };
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    constexpr std::array<std::array<std::uint8_t, 4>, 2> colours{{{255, 0, 0, 255}, {0, 255, 0, 255}}};
    std::string failures;
    for (const auto& test : cases) {
        const std::string what = std::string(watched ? "(watched) " : "") + test.name;
        GuestTextureResource resource{};
        resource.width = test.width;
        resource.height = test.height;
        resource.mipCount = test.mipCount;
        resource.tileMode = test.tileMode;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        const auto surfaceBytes = static_cast<std::size_t>(geometry.guestBytes);
        std::size_t texelCount = 0;
        for (const auto& mip : geometry.mips) texelCount += static_cast<std::size_t>(mip.width) * mip.height;
        // The surface, then a 64 KiB unit past it that no store may reach.
        const auto bytes = (static_cast<std::size_t>(test.offset) + surfaceBytes + 65535) / 65536 * 65536 + 65536;
        void* block = nullptr;
        if (watched) {
            block = AllocateWatched(bytes, 65536);
            if (block == nullptr) {
                std::cout << "no write watching: write-back padding in watched memory not tested\n";
                return;
            }
        } else {
#ifdef _WIN32
            block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
            block = std::aligned_alloc(65536, bytes);
#endif
        }
        Require(block != nullptr, "cannot allocate the write-back padding block");
        auto* memory = static_cast<std::uint8_t*>(block);
        for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + test.offset;
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(block, bytes, true, true, true);
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            bool watched;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                if (watched) ReleaseWatched(block, bytes);
            }
        } unregister{base, block, blockAddress, bytes, watched};
        if (HostImportFor(base, blockAddress, bytes) == nullptr) {
            std::cout << "host import of the write-back padding block refused: write-back padding not tested\n";
            return;
        }
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        resource.baseAddress = address;
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, test.mipCount, 0, 1};
            for (std::size_t pass = 0; pass < (test.direct ? colours.size() : 1u); ++pass) {
                const auto& colour = colours[pass];
                const auto commands = recorder.Commands();
                recorder.Keep(image);
                const VkClearColorValue value{{colour[0] / 255.0f, colour[1] / 255.0f, colour[2] / 255.0f, colour[3] / 255.0f}};
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                const auto slabRetilesBefore = StorageTexture::SlabRetiles();
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                const auto slabRetiled = StorageTexture::SlabRetiles() - slabRetilesBefore;
                if (test.direct && watched && UnitShadowEnabled() && AgcDriver::GuestMemory::Watched(address, surfaceBytes) && slabRetiled == 0) failures += what + ": no window was retiled straight into its unit-shadow slab\n";
                std::vector<std::byte> read(bytes);
                if (watched) AgcDriver::GuestMemory::Read(blockAddress, read);
                else std::memcpy(read.data(), memory, bytes);
                std::size_t texels = 0, garbage = 0, outside = 0;
                for (std::size_t i = 0; i < bytes; i += 4) {
                    const bool inside = i >= test.offset && i < test.offset + surfaceBytes;
                    bool isTexel = inside, isPattern = true;
                    for (std::size_t c = 0; c < 4; ++c) {
                        const auto byte = std::to_integer<std::uint8_t>(read[i + c]);
                        isTexel = isTexel && byte == colour[c];
                        isPattern = isPattern && byte == pattern(i + c);
                    }
                    if (isTexel) ++texels;
                    else if (!isPattern) ++(inside ? garbage : outside);
                }
                if (garbage != 0 || outside != 0 || texels != texelCount) {
                    failures += what + ", write-back " + std::to_string(pass + 1) + ": stored " + std::to_string(texels) + " texels of " + std::to_string(texelCount) + " and changed " + std::to_string(garbage) + " words no texel holds (" + std::to_string(outside) + " past the surface)\n";
                }
            }
        }
        recorder.Sync();
    }
    if (!failures.empty()) throw std::runtime_error("write-back padding:\n" + failures);
    std::cout << "write-back padding" << (watched ? " (watched)" : "") << ": ok\n";
}

void splitWriteBackTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0 || 4096 % base.hostImportAlignment != 0) {
        std::cout << "host imports unavailable: the split write-back not tested\n";
        return;
    }
    struct Case {
        const char* name;
        TextureTileMode tileMode;
        std::uint32_t mipCount;
        std::uint64_t offset;
        std::uint64_t split;
        bool gpu;
        bool direct;
        bool twoAllocations;
        bool secondStore;
    };
    const Case cases[] = {
        {"64 KiB R_X 200x150, 3 mips, ranges split at 128 KiB", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, false, false},
        {"64 KiB R_X 200x150, 3 mips, off a 64 KiB boundary, ranges split at 132 KiB", TextureTileMode::kR64KBX, 3, 4096, 135168, true, true, false, false},
        {"linear 200x150, ranges split at 64 KiB", TextureTileMode::kLinear, 1, 0, 65536, true, true, false, false},
        {"64 KiB R_X 200x150, 3 mips, second range of another access", TextureTileMode::kR64KBX, 3, 0, 131072, false, false, false, false},
        {"64 KiB R_X 200x150, 3 mips, ranges split inside a 64 KiB unit", TextureTileMode::kR64KBX, 3, 0, 135168, true, !UnitShadowEnabled(), false, false},
        {"64 KiB R_X 200x150, 3 mips, two adjacent allocations split at 128 KiB", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, true, false},
        {"64 KiB R_X 200x150, 3 mips, ranges split at 128 KiB, a second store outside the write watch", TextureTileMode::kR64KBX, 3, 0, 131072, true, true, false, true},
        {"linear 200x150, ranges split at 64 KiB, a second store outside the write watch", TextureTileMode::kLinear, 1, 0, 65536, true, true, false, true},
    };
    SetHostImportWindow(64ull << 10u);
    struct Window {
        ~Window() { SetHostImportWindow(0); }
    } window;
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    std::string failures;
    for (const auto& test : cases) {
        GuestTextureResource resource{};
        resource.width = 200;
        resource.height = 150;
        resource.mipCount = test.mipCount;
        resource.tileMode = test.tileMode;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        const auto surfaceBytes = static_cast<std::size_t>(geometry.guestBytes);
        Require(test.offset < test.split && test.split < test.offset + surfaceBytes, "the split test's ranges do not divide its surface");
        std::size_t texelCount = 0;
        for (const auto& mip : geometry.mips) texelCount += static_cast<std::size_t>(mip.width) * mip.height;
        const auto bytes = (static_cast<std::size_t>(test.offset) + surfaceBytes + 65535) / 65536 * 65536 + 65536;
#ifdef _WIN32
        void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* block = std::aligned_alloc(65536, bytes);
#endif
        Require(block != nullptr, "cannot allocate the split write-back block");
        auto* memory = static_cast<std::uint8_t*>(block);
        for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + test.offset;
        {
            GuestAllocations::Mutation mutation;
            if (test.twoAllocations) {
                mutation.Add(block, static_cast<std::size_t>(test.split), true, true, true);
                mutation.Add(memory + test.split, static_cast<std::size_t>(bytes - test.split), true, true, test.gpu);
            } else {
                mutation.Add(block, bytes, true, true, true);
                mutation.Protect(memory + test.split, static_cast<std::size_t>(bytes - test.split), true, true, test.gpu, [] {});
            }
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            void* second;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                    if (second != nullptr) mutation.Remove(second);
                }
                HostImportFor(context, address, bytes);
#ifdef _WIN32
                VirtualFree(block, 0, MEM_RELEASE);
#else
                std::free(block);
#endif
            }
        } unregister{base, block, blockAddress, bytes, test.twoAllocations ? static_cast<void*>(memory + test.split) : nullptr};
        Require(HostImportFor(base, address, surfaceBytes) == nullptr, std::string(test.name) + ": one import covers both ranges");
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        resource.baseAddress = address;
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, test.mipCount, 0, 1};
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
            image->WriteBack();
            const bool direct = !image->GuestSnapshotValid();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            std::size_t texels = 0, garbage = 0, outside = 0;
            for (std::size_t i = 0; i < bytes; i += 4) {
                const bool inside = i >= test.offset && i < test.offset + surfaceBytes;
                bool isTexel = inside, isPattern = true;
                for (std::size_t c = 0; c < 4; ++c) {
                    isTexel = isTexel && memory[i + c] == red[c];
                    isPattern = isPattern && memory[i + c] == pattern(i + c);
                }
                if (isTexel) ++texels;
                else if (!isPattern) ++(inside ? garbage : outside);
            }
            if (direct != test.direct) failures += std::string(test.name) + ": the write-back took the " + (direct ? "GPU-direct" : "CPU") + " path\n";
            if (direct && !(HostImportCovers(base, blockAddress, static_cast<std::size_t>(test.split)) && HostImportCovers(base, blockAddress + test.split, static_cast<std::size_t>(bytes - test.split)))) failures += std::string(test.name) + ": a range has no import after the GPU-direct write-back\n";
            if (garbage != 0 || outside != 0 || texels != texelCount) failures += std::string(test.name) + ": the write-back stored " + std::to_string(texels) + " texels of " + std::to_string(texelCount) + " and changed " + std::to_string(garbage) + " words no texel holds (" + std::to_string(outside) + " past the surface)\n";
            else if (test.secondStore) {
                const auto cpuPage = static_cast<std::size_t>(test.split) + 16384;
                Require(cpuPage + 4096 <= test.offset + surfaceBytes, std::string(test.name) + ": the CPU's page lies past the surface");
                if (AgcDriver::GuestMemory::Watched(address, surfaceBytes)) failures += std::string(test.name) + ": the surface is write-watched\n";
                if (!image->GuestSnapshotValid()) failures += std::string(test.name) + ": the first write-back left no saved bytes\n";
                for (std::size_t i = cpuPage; i < cpuPage + 4096; ++i) memory[i] = static_cast<std::uint8_t>(~pattern(i));
                const auto again = recorder.Commands();
                const VkClearColorValue greenValue{{0.0f, 1.0f, 0.0f, 1.0f}};
                RecordMemoryBarrier(context, again, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(again, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &greenValue, 1, &range);
                RecordMemoryBarrier(context, again, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                constexpr std::array<std::uint8_t, 4> green{0, 255, 0, 255};
                std::size_t kept = 0, greenBelow = 0, greenAbove = 0;
                for (std::size_t i = cpuPage; i < cpuPage + 4096; ++i) kept += memory[i] == static_cast<std::uint8_t>(~pattern(i)) ? 1 : 0;
                for (std::size_t i = test.offset; i < test.offset + surfaceBytes; i += 4) {
                    if (i >= cpuPage && i < cpuPage + 4096) continue;
                    if (std::equal(green.begin(), green.end(), memory + i)) ++(i < test.split ? greenBelow : greenAbove);
                }
                if (kept != 4096) failures += std::string(test.name) + ": the second write-back overwrote " + std::to_string(4096 - kept) + " of the 4096 bytes the CPU wrote after the first\n";
                if (greenBelow == 0 || greenAbove == 0) failures += std::string(test.name) + ": the second write-back stored " + std::to_string(greenBelow) + " texels below the split and " + std::to_string(greenAbove) + " above it\n";
            }
        }
        recorder.Sync();
    }
    if (!failures.empty()) throw std::runtime_error("split write-back:\n" + failures);
    std::cout << "split write-back: ok\n";
}

void writeBackBlockTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0 || 65536 % base.hostImportAlignment != 0) {
        std::cout << "host imports unavailable: write-back blocks not tested\n";
        return;
    }
    constexpr std::size_t piece = 65536;
    constexpr std::size_t window = std::size_t{8} << 20u;
    struct Case {
        const char* name;
        std::size_t offset;
        std::size_t registeredEnd;
        bool alternate;
        bool direct;
        std::size_t imports;
    };
    const Case cases[] = {
        {"a 4 MiB image over 64 adjacent 64 KiB pieces", 0, 4u << 20u, false, true, 1},
        {"a 4 MiB image at 64 KiB over 66 adjacent 64 KiB pieces", piece, (4u << 20u) + 2 * piece, false, true, 2},
        {"a 4 MiB image over 64 KiB pieces of which every other one is not GPU mapped", 0, 4u << 20u, true, false, 0},
    };
    SetHostImportWindow(window);
    struct Window {
        ~Window() { SetHostImportWindow(0); }
    } restore;
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    std::string failures;
    for (const auto& test : cases) {
        GuestTextureResource resource{};
        resource.width = 1024;
        resource.height = 1024;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kLinear;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto surfaceBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
        Require(surfaceBytes == (4u << 20u) && test.offset + surfaceBytes <= test.registeredEnd, std::string(test.name) + ": the surface does not fit its pieces");
#ifdef _WIN32
        void* reserved = VirtualAlloc(nullptr, 2 * window, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* reserved = std::aligned_alloc(window, 2 * window);
#endif
        Require(reserved != nullptr, "cannot allocate the write-back block");
        auto* memory = static_cast<std::uint8_t*>(reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(reserved) + window - 1) / window * window));
        for (std::size_t i = 0; i < window; ++i) memory[i] = pattern(i);
        std::vector<std::size_t> pieces;
        for (std::size_t offset = 0; offset < test.registeredEnd; offset += piece) pieces.push_back(offset);
        {
            GuestAllocations::Mutation mutation;
            for (const auto offset : pieces) mutation.Add(memory + offset, piece, true, true, !test.alternate || offset / piece % 2 == 0);
        }
        ClearHostImports(base.device);
        const auto address = reinterpret_cast<std::uint64_t>(memory) + test.offset;
        const auto before = HostImportCounters();
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        resource.baseAddress = address;
        std::size_t texels = 0;
        bool direct = false;
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
            image->WriteBack();
            direct = !image->GuestSnapshotValid();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            for (std::size_t i = test.offset; i < test.offset + surfaceBytes; i += 4) texels += std::equal(red.begin(), red.end(), memory + i) ? 1 : 0;
        }
        const auto after = HostImportCounters();
        std::size_t made = 0;
        std::size_t segments = 0;
        for (std::size_t index = 0; index < ImportRequesterCount; ++index) made += after.requesters[index].made - before.requesters[index].made;
        segments = after.requesters[static_cast<std::size_t>(ImportRequester::Segment)].made - before.requesters[static_cast<std::size_t>(ImportRequester::Segment)].made;
        std::printf("%s: %zu imports made (%zu by write-back segments), %s write-back, %zu of %zu texels stored\n", test.name, made, segments, direct ? "GPU-direct" : "CPU", texels, surfaceBytes / 4);
        if (made != test.imports) failures += std::string(test.name) + ": the write-back made " + std::to_string(made) + " imports, expected " + std::to_string(test.imports) + "\n";
        if (direct != test.direct) failures += std::string(test.name) + ": the write-back took the " + (direct ? "GPU-direct" : "CPU") + " path\n";
        if (texels != surfaceBytes / 4) failures += std::string(test.name) + ": the write-back stored " + std::to_string(texels) + " of " + std::to_string(surfaceBytes / 4) + " texels\n";
        {
            GuestAllocations::Mutation mutation;
            for (const auto offset : pieces) mutation.Remove(memory + offset);
        }
        ClearHostImports(base.device);
        recorder.Sync();
#ifdef _WIN32
        VirtualFree(reserved, 0, MEM_RELEASE);
#else
        std::free(reserved);
#endif
    }
    if (!failures.empty()) throw std::runtime_error("write-back blocks:\n" + failures);
    std::cout << "write-back blocks: ok\n";
}

void singlePassTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0 || !base.singlePassStorage) {
        std::cout << "host imports or storage access without a format unavailable: single-pass storage moves not tested\n";
        return;
    }
    struct Case {
        const char* name;
        std::uint32_t format;
        TextureTileMode tileMode;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t mipCount;
        std::uint32_t pipeBankXor = 0;
    };
    constexpr Case cases[] = {
        {"R8 64 KiB R_X 200x150", 1, TextureTileMode::kR64KBX, 200, 150, 1},
        {"R16 64 KiB R_X 200x150, 3 mips", 7, TextureTileMode::kR64KBX, 200, 150, 3},
        {"RGBA8 64 KiB R_X 520x260, 6 mips", 56, TextureTileMode::kR64KBX, 520, 260, 6},
        {"RGBA16F 64 KiB R_X 384x256", 71, TextureTileMode::kR64KBX, 384, 256, 1},
        {"RGBA16F 64 KiB R_X 256x200, 4 mips", 71, TextureTileMode::kR64KBX, 256, 200, 4},
        {"RGBA32F 64 KiB R_X 136x72, 2 mips", 77, TextureTileMode::kR64KBX, 136, 72, 2},
        {"RGBA8 64 KiB S 300x170", 56, TextureTileMode::kStandard64KB, 300, 170, 1},
        {"RGBA16F 4 KiB S 100x70, 2 mips", 71, TextureTileMode::kStandard4KB, 100, 70, 2},
        {"R8 256 B S 90x40", 1, TextureTileMode::kStandard256B, 90, 40, 1},
        {"RGBA8 64 KiB R_X 520x260, 2 mips, pipe/bank XOR 5", 56, TextureTileMode::kR64KBX, 520, 260, 2, 5},
    };
    std::string failures;
    std::size_t tested = 0;
    for (const auto& test : cases) {
        if (!StorageFormatAvailable(base, test.format)) continue;
        const std::string what = std::string(watched ? "(watched) " : "") + test.name;
        GuestTextureResource resource{};
        resource.width = test.width;
        resource.height = test.height;
        resource.mipCount = test.mipCount;
        resource.tileMode = test.tileMode;
        resource.dimension = TextureDimension::k2D;
        resource.format = test.format;
        resource.pipeBankXor = test.pipeBankXor;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        const auto elementBytes = BytesPerElement(test.format);
        const auto bytes = (static_cast<std::size_t>(geometry.guestBytes) + 65535) / 65536 * 65536 + 65536;
        struct Result {
            std::vector<std::uint8_t> uploaded, memory, reuploaded;
            std::uint64_t moves = 0;
        };
        std::array<Result, 2> results;
        bool skipped = false;
        for (int mode = 0; mode < 2 && !skipped; ++mode) {
            auto context = base;
            context.singlePassStorage = mode == 1;
            void* block = nullptr;
            if (watched) {
                block = AllocateWatched(bytes, 65536);
            } else {
#ifdef _WIN32
                block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
                block = std::aligned_alloc(65536, bytes);
#endif
            }
            if (block == nullptr) {
                std::cout << "no write watching: single-pass storage moves in watched memory not tested\n";
                return;
            }
            auto* memory = static_cast<std::uint8_t*>(block);
            std::uint32_t seed = 0x13579bdfu;
            const auto random = [&] {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                return static_cast<std::uint8_t>(seed >> 9);
            };
            for (std::size_t i = 0; i < bytes; ++i) memory[i] = random();
            const auto address = reinterpret_cast<std::uint64_t>(block);
            {
                GuestAllocations::Mutation mutation;
                mutation.Add(block, bytes, true, true, true);
            }
            struct Unregister {
                const Context& context;
                void* block;
                std::uint64_t address;
                std::size_t bytes;
                bool watched;
                ~Unregister() {
                    {
                        GuestAllocations::Mutation mutation;
                        mutation.Remove(block);
                    }
                    HostImportFor(context, address, bytes);
                    if (watched) ReleaseWatched(block, bytes);
#ifdef _WIN32
                    else VirtualFree(block, 0, MEM_RELEASE);
#else
                    else std::free(block);
#endif
                }
            } unregister{base, block, address, bytes, watched};
            if (HostImportFor(base, address, bytes) == nullptr) {
                std::cout << "host import refused: single-pass storage moves not tested\n";
                return;
            }
            TextureDetiler detiler(base);
            context.detiler = &detiler;
            resource.baseAddress = address;
            std::size_t linearBytes = 0;
            for (const auto& mip : geometry.mips) linearBytes += static_cast<std::size_t>(mip.width) * mip.height * elementBytes;
            const auto readImage = [&](const StorageTexture& image) {
                Buffer readback(context, linearBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                const auto commands = recorder.Commands();
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                std::vector<VkBufferImageCopy> copies;
                std::size_t at = 0;
                for (std::uint32_t level = 0; level < geometry.mips.size(); ++level) {
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = at;
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                    copy.imageExtent = {geometry.mips[level].width, geometry.mips[level].height, 1};
                    copies.push_back(copy);
                    at += static_cast<std::size_t>(geometry.mips[level].width) * geometry.mips[level].height * elementBytes;
                }
                context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), static_cast<std::uint32_t>(copies.size()), copies.data());
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                const auto read = readback.Bytes();
                return std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(read.data()), reinterpret_cast<const std::uint8_t*>(read.data()) + linearBytes);
            };
            std::vector<std::uint8_t> texels(linearBytes);
            for (auto& value : texels) value = random();
            const auto movesBefore = StorageTexture::SinglePassMoves();
            {
                auto image = StorageTexture::Create(context, detiler, resource, 0);
                recorder.Keep(image);
                results[mode].uploaded = readImage(*image);
                Buffer staging(context, linearBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                std::memcpy(staging.Bytes().data(), texels.data(), linearBytes);
                const auto commands = recorder.Commands();
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                std::vector<VkBufferImageCopy> copies;
                std::size_t at = 0;
                for (std::uint32_t level = 0; level < geometry.mips.size(); ++level) {
                    VkBufferImageCopy copy{};
                    copy.bufferOffset = at;
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                    copy.imageExtent = {geometry.mips[level].width, geometry.mips[level].height, 1};
                    copies.push_back(copy);
                    at += static_cast<std::size_t>(geometry.mips[level].width) * geometry.mips[level].height * elementBytes;
                }
                context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, staging.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(copies.size()), copies.data());
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                auto again = StorageTexture::Create(context, detiler, resource, 0);
                recorder.Keep(again);
                results[mode].reuploaded = readImage(*again);
                std::vector<std::byte> read(bytes);
                if (watched) AgcDriver::GuestMemory::Read(address, read);
                else {
                    StorageTexture::FlushPending(address, bytes, nullptr, "test", PublishScope::Whole);
                    recorder.Submit();
                    device.WaitQueue();
                    recorder.Sync();
                    std::memcpy(read.data(), memory, bytes);
                }
                results[mode].memory.assign(reinterpret_cast<const std::uint8_t*>(read.data()), reinterpret_cast<const std::uint8_t*>(read.data()) + bytes);
            }
            recorder.Sync();
            results[mode].moves = StorageTexture::SinglePassMoves() - movesBefore;
            if (results[mode].reuploaded != texels) failures += what + (mode == 1 ? " (single pass)" : " (copies)") + ": the second image does not hold the texels written back\n";
        }
        if (results[0].moves != 0 || results[1].moves < 3) failures += what + ": single-pass moves " + std::to_string(results[0].moves) + " with the copies and " + std::to_string(results[1].moves) + " in single pass (want 0 and at least 3)\n";
        if (results[0].uploaded != results[1].uploaded) failures += what + ": the first upload differs between the copies and single pass\n";
        if (results[0].memory != results[1].memory) {
            std::size_t differ = 0;
            for (std::size_t i = 0; i < results[0].memory.size(); ++i) differ += results[0].memory[i] != results[1].memory[i];
            failures += what + ": guest memory after the write-back differs in " + std::to_string(differ) + " bytes between the copies and single pass\n";
        }
        ++tested;
    }
    if (!failures.empty()) throw std::runtime_error("single-pass storage moves:\n" + failures);
    std::cout << "single-pass storage moves" << (watched ? " (watched)" : "") << ": ok (" << tested << " surfaces)\n";
}

class StorageUploadProbe {
public:
    StorageUploadProbe(Context& context, bool failView) : context(context), resolver(context.deviceProc), failView(failView) {
        Require(active == nullptr, "nested storage upload probe");
        active = this;
        context.deviceProc = resolve;
    }
    ~StorageUploadProbe() {
        for (const auto view : destroyedViews) function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
        if (imageDestroyed) function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
        if (memoryFreed) function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        context.deviceProc = resolver;
        active = nullptr;
    }
    VkImage Image() const { return image; }
    bool Destroyed() const { return imageDestroyed || memoryFreed; }
    bool Released() const { return imageDestroyed && memoryFreed; }
    bool FailedView() const { return failedView; }
    std::uint32_t Clears() const { return clears; }

private:
    template<typename TFunction>
    TFunction function(const char* name) const { return reinterpret_cast<TFunction>(resolver(context.device, name)); }
    static VKAPI_ATTR VkResult VKAPI_CALL createImage(VkDevice device, const VkImageCreateInfo* info, const VkAllocationCallbacks* callbacks, VkImage* created) {
        const auto result = active->function<PFN_vkCreateImage>("vkCreateImage")(device, info, callbacks, created);
        if (result == VK_SUCCESS) active->image = *created;
        return result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL bindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset) {
        const auto result = active->function<PFN_vkBindImageMemory>("vkBindImageMemory")(device, image, memory, offset);
        if (result == VK_SUCCESS && image == active->image) active->memory = memory;
        return result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL createView(VkDevice device, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* callbacks, VkImageView* created) {
        if (info->image == active->image && active->failView && active->clears != 0) {
            active->failedView = true;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        const auto result = active->function<PFN_vkCreateImageView>("vkCreateImageView")(device, info, callbacks, created);
        if (result == VK_SUCCESS && info->image == active->image) active->views.push_back(*created);
        return result;
    }
    static VKAPI_ATTR void VKAPI_CALL destroyView(VkDevice device, VkImageView view, const VkAllocationCallbacks* callbacks) {
        if (std::find(active->views.begin(), active->views.end(), view) != active->views.end()) active->destroyedViews.push_back(view);
        else active->function<PFN_vkDestroyImageView>("vkDestroyImageView")(device, view, callbacks);
    }
    static VKAPI_ATTR void VKAPI_CALL destroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* callbacks) {
        if (image == active->image) active->imageDestroyed = true;
        else active->function<PFN_vkDestroyImage>("vkDestroyImage")(device, image, callbacks);
    }
    static VKAPI_ATTR void VKAPI_CALL freeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
        if (memory == active->memory) active->memoryFreed = true;
        else active->function<PFN_vkFreeMemory>("vkFreeMemory")(device, memory, callbacks);
    }
    static VKAPI_ATTR void VKAPI_CALL clearImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearColorValue* value, std::uint32_t count, const VkImageSubresourceRange* ranges) {
        if (image == active->image) ++active->clears;
        active->function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image, layout, value, count, ranges);
    }
    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL resolve(VkDevice device, const char* name) {
        if (std::strcmp(name, "vkCreateImage") == 0) return reinterpret_cast<PFN_vkVoidFunction>(createImage);
        if (std::strcmp(name, "vkBindImageMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(bindImageMemory);
        if (std::strcmp(name, "vkCreateImageView") == 0) return reinterpret_cast<PFN_vkVoidFunction>(createView);
        if (std::strcmp(name, "vkDestroyImageView") == 0) return reinterpret_cast<PFN_vkVoidFunction>(destroyView);
        if (std::strcmp(name, "vkDestroyImage") == 0) return reinterpret_cast<PFN_vkVoidFunction>(destroyImage);
        if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(freeMemory);
        if (std::strcmp(name, "vkCmdClearColorImage") == 0) return reinterpret_cast<PFN_vkVoidFunction>(clearImage);
        return active->resolver(device, name);
    }
    Context& context;
    PFN_vkGetDeviceProcAddr resolver;
    bool failView;
    bool failedView = false;
    bool imageDestroyed = false;
    bool memoryFreed = false;
    std::uint32_t clears = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::vector<VkImageView> views;
    std::vector<VkImageView> destroyedViews;
    static inline StorageUploadProbe* active = nullptr;
};

void storageUploadLifetimeTests(const Device& device, Recorder& previous, bool failView) {
    auto context = device.GetContext();
    context.hostImportAlignment = 0;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.width = 64;
    resource.height = 64;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    const auto guestBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
    std::vector<std::byte> texels(guestBytes + 256, std::byte{0x55});
    std::vector<std::byte> keys(guestBytes / 256, std::byte{0});
    resource.baseAddress = (reinterpret_cast<std::uintptr_t>(texels.data()) + 255) & ~std::uintptr_t{255};
    resource.dccAddress = reinterpret_cast<std::uintptr_t>(keys.data());
    DccKeyProof proof;
    Require(ProvedClearKeys(resource, guestBytes, proof) == DccKeys::Clear0000, "storage lifetime fixture has no DCC clear");
    StorageUploadProbe probe(context, failView);
    Buffer readback(device.GetContext(), resource.width * resource.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(readback.Bytes().data(), 0x55, readback.Bytes().size());
    std::weak_ptr<StorageTexture> lifetime;
    bool premature = false;
    bool retained = true;
    bool threw = false;
    {
        Recorder recorder(context);
        recorder.Activate();
        try {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            lifetime = image;
            image.reset();
            retained = !lifetime.expired();
        } catch (const std::exception& error) {
            if (!probe.FailedView() || std::string_view(error.what()).find("vkCreateImageView storage") == std::string_view::npos) throw;
            threw = true;
        }
        premature = probe.Destroyed();
        Require(probe.Clears() == 1 && recorder.Recording(), "storage creation did not record its initial clear");
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {resource.width, resource.height, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, probe.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Sync();
    }
    previous.Activate();
    Require(threw == failView && probe.FailedView() == failView, "storage view failure injection did not take the requested path");
    Require(std::all_of(readback.Bytes().begin(), readback.Bytes().end(), [](std::byte value) { return value == std::byte{0}; }), "recorded storage clear did not reach every texel");
    Require(lifetime.expired() && probe.Released(), "storage upload resources outlived recorder teardown");
    Require(!premature && retained, failView ? "failed storage creation released an image still referenced by its recorded clear" : "storage creation did not retain its image through the recorded clear");
    std::cout << "Storage upload lifetime" << (failView ? " after view failure" : " after caller release") << ": passed\n";
}

// LOCAL EXPERIMENT (GPU-OWNED-PLAN L2-1): false when nothing could be tested (--refresh-proof-only
// then reports a skip).
bool refreshProofTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: refresh proofs not tested\n";
        return false;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: refresh proofs not tested\n";
        return false;
    }
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0xff, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr || !AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "no watched host import: refresh proofs not tested\n";
        return false;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    const auto proved = [] { return StorageTexture::RefreshesProved(); };
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto draw = [&](VkClearColorValue value) {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        };
        const auto settle = [&] {
            image->Refresh();
            image->Refresh();
        };
        const auto expectProved = [&](const char* what) {
            const auto before = proved();
            Require(image->Refresh(), what);
            Require(proved() == before + 1, what);
        };
        const auto expectFull = [&](const char* what) {
            const auto before = proved();
            image->Refresh();
            Require(proved() == before, what);
        };
        settle();
        expectProved("an unchanged surface was not proved current");
        expectProved("a proved surface was not proved again");
        draw({{1.0f, 0.0f, 0.0f, 1.0f}});
        settle();
        expectProved("a target holding its own pending results was not proved current");
        draw({{0.0f, 1.0f, 0.0f, 1.0f}});
        expectProved("a draw into an already pending target broke its proof");
        texels[0] = 0x11;
        expectFull("a CPU write of the surface was answered from the proof");
        settle();
        expectProved("the proof was not taken again after a CPU write");
        AgcDriver::GuestMemory::MarkWritten(address + 65536, 4096);
        expectFull("a driver store into the surface was answered from the proof");
        settle();
        AgcDriver::GuestMemory::MarkWritten(resource.dccAddress, keyCount);
        expectFull("a key store over the surface's keys was answered from the proof");
        settle();
        expectProved("the proof was not taken again after a key store");
        auto other = resource;
        other.dccAddress = 0;
        auto pending = StorageTexture::Create(context, detiler, other, 0);
        recorder.Keep(pending);
        pending->MarkDirty();
        expectFull("another image's pending results over the surface were answered from the proof");
        settle();
        expectProved("the proof was not taken again after another image's results were stored");
        pending->MarkDirty();
        expectFull("another image's pending results were answered from the proof");
        Require(!AgcDriver::Graphics::PendingStorageOverlaps(address, surfaceBytes, image.get()), "the full refresh left another image's results pending");
        auto fresh = StorageTexture::Create(context, detiler, resource, 0);
        const auto before = proved();
        fresh->Refresh();
        Require(proved() == before, "a new image of the surface was answered from another image's proof");
        std::memset(keys, 0x00, keyCount);
        expectFull("a fast clear of the keys was answered from the proof");
        recorder.Sync();
    }
    recorder.Sync();
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_PROOF_UNITS, GPU-OWNED-PLAN L2-1, P1): refresh proofs while
// another image is pending over the target with no pending unit inside it. A 1 MiB watched host
// import; the target T is a 256 KiB R64KBX surface at 256 KiB with its DCC keys at 768 KiB, and O a
// block-unit image without DCC over the whole 1 MiB (16 units of 64 KiB: T lies in O's units 4-7,
// its keys in O's unit 12). What a refresh answers follows the switch: unset or 0 never proves T
// while O is pending, 1 proves it, 2 hands every predicted hit to today's refresh (verified, never
// proved). With APS5_NO_FLUSH_PRETEST=1 every overlapping pending image has a unit "inside", so no
// proof is ever unit-checked. False when nothing could be tested.
bool refreshProofUnitTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: unit-checked refresh proofs not tested\n";
        return false;
    }
    constexpr std::size_t bytes = std::size_t{1} << 20;
    constexpr std::size_t targetOffset = 256 * 1024;
    constexpr std::size_t targetBytes = 256 * 1024;
    constexpr std::size_t keyOffset = 768 * 1024;
    constexpr std::size_t keyCount = targetBytes / 256;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: unit-checked refresh proofs not tested\n";
        return false;
    }
    auto* memory = static_cast<std::uint8_t*>(block);
    std::memset(memory, 0x55, bytes);
    std::memset(memory + keyOffset, 0xff, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr || !AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "no watched host import: unit-checked refresh proofs not tested\n";
        return false;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    const auto surface = [](std::uint64_t at, std::uint32_t side, std::uint64_t keys) {
        GuestTextureResource resource{};
        resource.baseAddress = at;
        resource.width = side;
        resource.height = side;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kR64KBX;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        resource.dccAddress = keys;
        return resource;
    };
    const auto targetResource = surface(address + targetOffset, 256, address + keyOffset);
    const auto otherResource = surface(address, 512, 0);
    Require(DescribeSurface(targetResource).guestBytes == targetBytes && DescribeSurface(otherResource).guestBytes == bytes, "unit-checked proof fixture: unexpected surface sizes");
    const char* text = std::getenv("APS5_EXPERIMENT_PROOF_UNITS");
    const auto value = text != nullptr ? std::strtol(text, nullptr, 10) : 0;
    const int units = value == 0 ? 0 : value == 2 ? 2 : 1;
    const bool pretest = std::getenv("APS5_NO_FLUSH_PRETEST") == nullptr;
    // Without the pretest no proof is unit-checked: the refreshes answer as with the switch off.
    const int mode = pretest ? units : 0;
    const auto proved = [] { return StorageTexture::RefreshesProved(); };
    const auto counts = [] { return StorageTexture::ProofUnitTotals(); };
    const auto start = counts();
    {
        auto target = StorageTexture::Create(context, detiler, targetResource, 0);
        recorder.Keep(target);
        Require(target->BlockUnits() && target->TrackedUnits() == 4, "unit-checked proof fixture: the target is not four block units");
        const auto settle = [&] {
            target->Refresh();
            target->Refresh();
        };
        const auto expectProved = [&](bool expected, const char* what) {
            const auto before = proved();
            target->Refresh();
            Require(proved() == before + (expected ? 1 : 0), what);
        };
        settle();
        expectProved(true, "the target was not proved with nothing else pending");
        auto other = StorageTexture::Create(context, detiler, otherResource, 0);
        recorder.Keep(other);
        Require(other->BlockUnits() && other->TrackedUnits() == 16, "unit-checked proof fixture: the other image is not sixteen block units");

        // 1. O pending outside T and its keys: 1 converts T's proof at once and proves it from then
        // on, 2 predicts and verifies every refresh, 0 (and no pretest) never proves T.
        std::vector<bool> outside(16, false);
        outside[0] = outside[1] = true;
        other->MarkUnitsWritten(outside);
        Require(other->Dirty(), "the other image's units were not marked pending");
        auto before = counts();
        expectProved(mode == 1, mode == 1 ? "1: the proof was not converted to a unit-checked one" : "a proof held while another image was pending over the target");
        settle();
        expectProved(mode == 1, mode == 1 ? "1: the target was not proved after one settle with the other image pending outside it" : "the target was proved with another image pending over it");
        auto after = counts();
        if (mode == 1) Require(after.converted == before.converted + 1 && after.hits == before.hits + 4 && after.verified == before.verified, "1: the unit-checked proof was not converted once and used on every refresh");
        if (mode == 2) Require(after.converted == before.converted + 1 && after.taken == before.taken + 4 && after.verified == before.verified + 4 && after.hits == before.hits, "2: every predicted hit was not verified against today's refresh");
        // A CPU store into T: the next refresh uploads, the one after takes the proof unit-checked.
        memory[targetOffset] = 0x11;
        expectProved(false, "a CPU store into the target was answered from a unit-checked proof");
        before = counts();
        settle();
        expectProved(mode == 1, mode == 1 ? "1: no unit-checked proof was taken after a CPU store" : "the target was proved after a CPU store with another image pending over it");
        after = counts();
        if (mode != 0) Require(after.taken >= before.taken + 1, "no unit-checked proof was taken at the unchanged exit");

        // 2. A unit inside T marked in O, already pending: no pending serial moves (the premise of the
        // per-use check), and T's next refresh is not proved and stores the unit.
        const auto serial = StorageTexture::PendingSerial();
        auto inside = outside;
        inside[5] = true;
        other->MarkUnitsWritten(inside);
        Require(StorageTexture::PendingSerial() == serial, "a unit marked in an image already pending moved the pending serial");
        before = counts();
        expectProved(false, "a refresh was proved with another image's unit pending inside the target");
        if (mode != 0) Require(counts().refusedInside == before.refusedInside + 1, "the unit-checked proof was not refused for the unit inside");
        if (pretest) Require(!StorageTexture::FlushPending(targetResource.baseAddress, targetBytes, target.get(), "unit-checked proof test", PublishScope::None), "the full refresh left the other image's unit inside the target pending");

        // 3. An image of exactly T's surface pending: refused (the flush stores it).
        settle();
        auto same = targetResource;
        same.dccAddress = 0;
        auto alias = StorageTexture::Create(context, detiler, same, 0);
        recorder.Keep(alias);
        alias->MarkDirty();
        before = counts();
        expectProved(false, "a refresh was proved with an image of the same surface pending");
        if (mode != 0) Require(counts().refusedSameRange == before.refusedSameRange + 1, "the unit-checked proof was not refused for the same-range image");
        Require(!alias->Dirty(), "the full refresh left the same-range image pending");

        // 4. A unit of O inside T's DCC keys only: refused on use, and no proof is taken while it is
        // pending (stricter than today's proof, which reads no other image's keys).
        settle();
        auto keysUnit = outside;
        keysUnit[12] = true;
        other->MarkUnitsWritten(keysUnit);
        before = counts();
        expectProved(false, "a refresh was proved with another image's unit pending inside the target's keys");
        if (mode != 0) Require(counts().refusedKeyRange == before.refusedKeyRange + 1, "the unit-checked proof was not refused for the unit inside the keys");
        settle();
        expectProved(false, "a proof was taken with another image's unit pending inside the target's keys");

        // The unit scan over ranges around O's pending units (0, 1, 12): only a range meeting one
        // stores it (=2 checks every scan against the loop over every unit).
        if (pretest) {
            Require(!StorageTexture::FlushPending(address + 2 * 65536, 65536, nullptr, "unit-checked proof test", PublishScope::None), "a range of no pending unit stored the other image");
            Require(!StorageTexture::FlushPending(address + 11 * 65536 + 4096, 65536 - 8192, nullptr, "unit-checked proof test", PublishScope::None), "a range inside a unit that is not pending stored the other image");
            Require(StorageTexture::FlushPending(address + 2 * 65536 - 1, 1, nullptr, "unit-checked proof test", PublishScope::None), "the last byte of a pending unit did not store it");
            Require(StorageTexture::FlushPending(address + 12 * 65536 - 1, 2, nullptr, "unit-checked proof test", PublishScope::None), "a range straddling into a pending unit did not store it");
            Require(StorageTexture::FlushPending(address + 1, 3, nullptr, "unit-checked proof test", PublishScope::None), "a range inside the first pending unit did not store it");
        } else {
            StorageTexture::FlushPending(address, bytes, nullptr, "unit-checked proof test", PublishScope::None);
        }
        Require(!other->Dirty(), "the other image is still pending");
        // The stores of unit 12 over T's keys complete, so a key scan is kept as the proof again.
        recorder.Sync();
        settle();
        expectProved(true, "the target was not proved once nothing else was pending");
        recorder.Sync();
    }
    recorder.Sync();
    const auto end = counts();
    if (mode == 0) Require(end.taken == start.taken && end.converted == start.converted && end.hits == start.hits && end.verified == start.verified, "a proof was unit-checked with the switch off or without the flush pretest");
    if (mode == 2) Require(end.hits == start.hits && end.scansChecked > start.scansChecked, "2: a unit-checked proof answered a refresh, or no unit scan was checked against the loop");
    if (mode == 1) Require(end.scansChecked == start.scansChecked, "1: a unit scan was checked against the loop");
    Require(end.raced == start.raced && end.keysOnly == start.keysOnly, "a predicted hit was overtaken by a store or differed in the keys");
    std::cout << "Unit-checked refresh proofs (APS5_EXPERIMENT_PROOF_UNITS=" << units << (pretest ? "" : ", no flush pretest") << "): passed\n";
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_KEY_PIN, GPU-OWNED-PLAN 9.3, K1): refresh proofs over pinned key
// proofs. The refresh proof fixture with its keys a 0000 clear, in a 1 MiB watched host import (a
// smaller import of private memory is compared instead of watched on a device that settles imports,
// such as this one): T is a 256 KiB R64KBX surface at 0 with its keys at 256 KiB. A draw into T
// makes the driver store its uncompressed keys on the GPU (MarkDirty), pending in the open batch.
// With the switch unset or 0 no refresh of T is proved meanwhile (key proof empty); 1 proves them
// from the pin, moving the layer generations as today's refresh does; 2 hands each to today's
// refresh, which must agree. C10 (hand-over lists) has no case: they are filled only by
// clearHandOver inside a refresh whose overlapping pending images have units inside the surface,
// which then takes no proof (other pending), and the next refresh clears them before it takes one.
// C7 (a stale depth plane) needs a depth surface in the harness, as P1's refusal does. False when
// nothing could be tested.
bool refreshKeyPinTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: pinned refresh proofs not tested\n";
        return false;
    }
    if (!KeyFastPath()) {
        std::cout << "key fast path off: pinned refresh proofs not tested\n";
        return false;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = std::size_t{1} << 20;
    // A 64 KiB block neither T nor its keys touch: a stamp there only moves the tracker generation.
    constexpr std::size_t elsewhere = 6 * 65536;
    // The two-layer variant V of case 6, and its keys.
    constexpr std::size_t variantOffset = 8 * 65536;
    constexpr std::size_t variantKeyOffset = 15 * 65536;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: pinned refresh proofs not tested\n";
        return false;
    }
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    std::memset(texels, 0x55, bytes);
    std::memset(keys, 0x00, keyCount);
    std::memset(texels + variantKeyOffset, 0x00, 65536);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr || !AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "no watched host import: pinned refresh proofs not tested\n";
        return false;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    const char* text = std::getenv("APS5_EXPERIMENT_KEY_PIN");
    const auto value = text != nullptr ? std::strtol(text, nullptr, 10) : 0;
    const int mode = value == 0 ? 0 : value == 2 ? 2 : 1;
    const auto proved = [] { return StorageTexture::RefreshesProved(); };
    const auto delta = [](const KeyPinCounts& before, KeyPin kind) { return KeyPinTotals()[kind] - before[kind]; };
    const auto start = KeyPinTotals();
    recorder.Sync();
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const auto draw = [&](const std::shared_ptr<StorageTexture>& target) {
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
            const VkClearColorValue color{{1.0f, 0.0f, 0.0f, 1.0f}};
            const auto commands = recorder.Commands();
            recorder.Keep(target);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, target->Image(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            target->MarkDirty();
        };
        // The driver's key store as StoreUncompressedOnGpu makes it, without the command: a 0xff fill
        // noted in the open batch, the keys stamped, the memo entry with the fill's note.
        const auto store = [&] {
            recorder.NotePendingFill(resource.dccAddress, keyCount, 0xff);
            AgcDriver::GuestMemory::MarkWritten(resource.dccAddress, keyCount);
            NoteKeysFillOnGpu(resource.dccAddress, keyCount, DccKeys::Uncompressed);
        };
        const auto settle = [&] {
            image->Refresh();
            image->Refresh();
        };
        const auto expectProved = [&](bool expected, const char* what) {
            const auto before = proved();
            image->Refresh();
            Require(proved() == before + (expected ? 1 : 0), what);
        };
        // Today's unchanged exit moves every unit to the surface's collect, made during the refresh (a
        // pinned hit's in refreshProved): after a stamp elsewhere moved the tracker generation, every
        // unit lies between the generations read before and after the refresh, and all are equal.
        const auto expectAdvanced = [&](const char* what) {
            AgcDriver::GuestMemory::MarkWritten(address + elsewhere, 4096);
            const auto before = AgcDriver::GuestMemory::TrackerGeneration();
            image->Refresh();
            const auto after = AgcDriver::GuestMemory::TrackerGeneration();
            const auto first = image->LayerGeneration(0);
            Require(first >= before && first <= after, what);
            for (std::uint32_t unit = 1; unit < image->TrackedUnits(); ++unit) Require(image->LayerGeneration(unit) == first, what);
        };

        // 1. 0000 keys: proved as today. A draw makes the driver store uncompressed keys on the GPU.
        settle();
        expectProved(true, "K1 refresh (1): the cleared target was not proved");
        draw(image);
        Require(image->UploadedKeys() == DccKeys::Uncompressed, "K1 refresh (1): the first draw did not store the keys uncompressed");
        if (!recorder.PendingWriteOverlaps(resource.dccAddress, keyCount)) {
            std::cout << "the first draw stored the keys on the CPU: pinned refresh proofs not tested\n";
            recorder.Sync();
            return false;
        }

        // 2. The next refresh pins the keys (1 and 2), then three more: 1 proves each from the pin, 2
        // verifies each against today's refresh, 0 proves none. Each leaves today's generations.
        auto before = KeyPinTotals();
        image->Refresh();
        if (mode != 0) Require(delta(before, KeyPin::Taken) == 1 && image->KeyProof().pinNote != 0, "K1 refresh (2): the refresh after the draw did not pin the keys");
        else Require(image->KeyProof().generation == 0 && image->KeyProof().pinNote == 0, "K1 refresh (2): a key proof was kept under the pending store with the switch off");
        before = KeyPinTotals();
        const auto provedBefore = proved();
        for (int round = 0; round < 3; ++round) expectAdvanced("K1 refresh (2): a refresh left a unit off the generation of the surface's collect");
        Require(proved() == provedBefore + (mode == 1 ? 3 : 0), mode == 1 ? "K1 refresh (2) 1: the pinned target was not proved" : "K1 refresh (2): the target was proved under a pending key store");
        if (mode == 1) Require(delta(before, KeyPin::RefreshHit) == 3, "K1 refresh (2) 1: the pinned hits were not counted");
        if (mode == 2) Require(delta(before, KeyPin::RefreshVerified) == 3 && delta(before, KeyPin::RefreshHit) == 0, "K1 refresh (2) 2: the pinned predictions were not verified against today's refresh");

        // 3. A CPU store into T: no mode proves it; once settled, 1 proves again.
        texels[0] = 0x11;
        expectProved(false, "K1 refresh (3): a CPU store into the surface was answered from a pinned proof");
        settle();
        expectProved(mode == 1, mode == 1 ? "K1 refresh (3) 1: the pinned target was not proved after a CPU store" : "K1 refresh (3): the target was proved under a pending key store");

        // 4. A CPU store into the keys: today still reads them from the memo (no upload) and the pin is
        // taken again at the new collect; 1 proves after.
        keys[0] = 0x40;
        before = KeyPinTotals();
        Require(image->Refresh(), "K1 refresh (4): a CPU store into the keys under the pending store uploaded the surface");
        if (mode != 0) Require(delta(before, KeyPin::Taken) == 1, "K1 refresh (4): the pin was not taken again after a CPU store into the keys");
        expectProved(mode == 1, mode == 1 ? "K1 refresh (4) 1: the target was not proved after the pin was taken again" : "K1 refresh (4): the target was proved under a pending key store");

        // 5. Another image of T's surface pending: no mode proves the refresh (its flush stores it).
        // The driver's store is played again after (the flush may have run the batch).
        auto other = resource;
        other.dccAddress = 0;
        auto pending = StorageTexture::Create(context, detiler, other, 0);
        recorder.Keep(pending);
        pending->MarkDirty();
        expectProved(false, "K1 refresh (5): another image's pending results over the target were answered from a pinned proof");
        store();
        settle();
        settle();
        expectProved(mode == 1, mode == 1 ? "K1 refresh (5) 1: the target was not proved from a new pin" : "K1 refresh (5): the target was proved under a pending key store");

        // 6. RefreshLayers of a two-layer DCC variant V: a DCC surface's layer refresh is never scoped
        // (RefreshLayers refreshes it whole), so its pin answers (1) and C9 never refuses.
        auto variantResource = resource;
        variantResource.baseAddress = address + variantOffset;
        variantResource.width = 128;
        variantResource.height = 128;
        variantResource.dimension = TextureDimension::k2DArray;
        variantResource.depthOrLastArray = 1;
        variantResource.dccAddress = address + variantKeyOffset;
        const auto variantBytes = DescribeSurface(variantResource).guestBytes;
        Require(variantBytes != 0 && variantBytes <= variantKeyOffset - variantOffset && variantBytes / 256 <= 65536, "K1 refresh (6): the two-layer variant does not fit its fixture");
        auto variant = StorageTexture::Create(context, detiler, variantResource, 0);
        Require(variant->ImageLayers() == 2, "K1 refresh (6): the variant is not two layers");
        variant->Refresh();
        variant->Refresh();
        draw(variant);
        Require(variant->UploadedKeys() == DccKeys::Uncompressed, "K1 refresh (6): the variant's first draw did not store its keys uncompressed");
        variant->Refresh();
        before = KeyPinTotals();
        const auto variantProved = proved();
        Require(variant->RefreshLayers(0, 1), "K1 refresh (6): the variant's first layer was not current");
        Require(delta(before, KeyPin::Scoped) == 0, "K1 refresh (6): a DCC surface's layer refresh was refused as scoped");
        if (mode == 1) Require(proved() == variantProved + 1 && delta(before, KeyPin::RefreshHit) == 1, "K1 refresh (6) 1: the variant's layer refresh was not proved from its pin");
        if (mode != 1) Require(proved() == variantProved, "K1 refresh (6): the variant's layer refresh was proved under a pending key store");

        // 7. The batch runs (1 still proves: not reaped); the reap ends the pin, after which the bytes
        // (0xff, as the store left them) give a stable key proof and every mode proves.
        std::memset(keys, 0xff, keyCount);
        image->Refresh();
        expectProved(mode == 1, mode == 1 ? "K1 refresh (7) 1: the target was not proved after a CPU store into the keys" : "K1 refresh (7): the target was proved under a pending key store");
        recorder.Submit();
        device.WaitQueue();
        expectProved(mode == 1, mode == 1 ? "K1 refresh (7) 1: the target was not proved before the batch was reaped" : "K1 refresh (7): the target was proved under a key store not reaped");
        recorder.Sync();
        before = KeyPinTotals();
        image->Refresh();
        if (mode != 0) Require(delta(before, KeyPin::NotPending) >= 1, "K1 refresh (7): the reap did not end the pin");
        // Once as the refresh's refusal; its slow path's key read counts its own apart.
        if (mode != 0) Require(KeyPinTotals().Refreshes(KeyPin::NotPending) == before.Refreshes(KeyPin::NotPending) + 1, "K1 refresh (7): the refresh's refusal at the reap was not counted once as a refresh's");
        Require(image->KeyProof().generation != 0 && image->KeyProof().pinNote == 0, "K1 refresh (7): the reaped keys gave no stable key proof");
        expectProved(true, "K1 refresh (7): the target was not proved with a stable key proof");

        // 8. A newer writer nothing stamped over the keys after a new pin: not the newest any more.
        store();
        image->Refresh();
        expectProved(mode == 1, mode == 1 ? "K1 refresh (8) 1: the target was not proved from a new pin" : "K1 refresh (8): the target was proved under a pending key store");
        recorder.NotePendingFill(resource.dccAddress, keyCount, 0xff);
        before = KeyPinTotals();
        expectProved(false, "K1 refresh (8): a refresh was proved from a pin under a newer writer");
        if (mode != 0) Require(delta(before, KeyPin::NotNewest) >= 1, "K1 refresh (8): the pin was not refused under a newer writer");
        if (mode != 0) Require(KeyPinTotals().Refreshes(KeyPin::NotNewest) == before.Refreshes(KeyPin::NotNewest) + 1, "K1 refresh (8): the refresh's refusal under a newer writer was not counted once as a refresh's");
        recorder.Sync();

        // 9. A pin another key read replaced with a stable proof at the same generation: the refresh
        // proof is taken over a new pin, the batch is reaped with nothing stamped, and a key read
        // through the image's key proof (fastRevalidate's or ServesKeysAt's) refuses the pin (not
        // pending) and keeps the bytes' stable proof at the pin's generation. Today took no refresh
        // proof under the pin, so its next refresh runs the slow path and moves every unit to the
        // surface's collect; no mode may answer it from the proof taken over the pin (a plain hit
        // would leave the units at the pin's generation).
        store();
        image->Refresh();
        if (mode != 0) Require(image->KeyProof().pinNote != 0, "K1 refresh (9): the refresh after the store did not pin the keys");
        expectProved(mode == 1, mode == 1 ? "K1 refresh (9) 1: the target was not proved from the pin" : "K1 refresh (9): the target was proved under a pending key store");
        const auto pinGeneration = image->KeyProof().generation;
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        Require(ProvedClearKeys(resource, surfaceBytes, image->KeyProof()) == DccKeys::Uncompressed && image->KeyProof().generation != 0 && image->KeyProof().pinNote == 0, "K1 refresh (9): the reaped keys gave no stable key proof");
        // The exact state needs nothing stamped between the pin and the stable proof; where the fixture
        // stamps (the reap's write-backs), the stable proof is at a newer generation and the checks
        // below still hold, without reaching the keyPinned guard.
        if (mode != 0) std::cout << "K1 refresh (9): the stable proof is " << (image->KeyProof().generation == pinGeneration ? "at the pin's generation (keyPinned guard reached)" : "at a newer generation (keyPinned guard not reached)") << '\n';
        const auto provedReplaced = proved();
        expectAdvanced("K1 refresh (9): the refresh after a stable proof replaced the pin left a unit off the generation of the surface's collect");
        Require(proved() == provedReplaced, "K1 refresh (9): the refresh proof taken over a pin answered after a stable proof replaced the pin");
        expectProved(true, "K1 refresh (9): the target was not proved after today's refresh took the stable proof");
        recorder.Sync();
    }
    recorder.Sync();
    // Nothing counted with the switch off; no answer at 2, no verify at 1, no race.
    const auto end = KeyPinTotals();
    if (mode == 0) Require(end.target == start.target && end.other == start.other && end.refreshTarget == start.refreshTarget && end.refreshOther == start.refreshOther, "K1 refresh: a pin was taken or counted with the switch off");
    if (mode == 1) Require(end[KeyPin::KeysVerified] == start[KeyPin::KeysVerified] && end[KeyPin::RefreshVerified] == start[KeyPin::RefreshVerified], "K1 refresh 1: a pin was verified");
    if (mode == 2) Require(end[KeyPin::KeyAnswer] == start[KeyPin::KeyAnswer] && end[KeyPin::RefreshHit] == start[KeyPin::RefreshHit], "K1 refresh 2: a pin answered");
    Require(end[KeyPin::Raced] == start[KeyPin::Raced], "K1 refresh: a pinned prediction was overtaken in a single-threaded test");
    std::cout << "Pinned refresh proofs (APS5_EXPERIMENT_KEY_PIN=" << mode << "): passed\n";
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): a target lease's refresh
// (StorageTexture::RefreshForLease) on its own: cases (a)-(h) of the D4 design, the lease's own rows,
// and K1's two (a pinned hit applies the generation advance; a pinned prediction writes nothing). A 1
// MiB watched host import (a smaller import of private memory is compared instead of watched on a
// device that settles imports, such as this one): T is a 256 KiB R64KBX surface at 0 with its keys
// (0xff, uncompressed) at 256 KiB, O a block-unit image without DCC over the whole 1 MiB (16 units of
// 64 KiB: T lies in O's units 0-3, its keys in unit 4), and a twin of T. RefreshForLease has no switch
// of its own; its verdicts follow APS5_EXPERIMENT_PROOF_UNITS (unit-checked proofs) and
// APS5_EXPERIMENT_KEY_PIN (the pinned target). False when nothing could be tested.
bool targetLeaseRefreshTests(const Device& device, Recorder& recorder) {
    using Verdict = StorageTexture::LeaseVerdict;
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: lease refreshes not tested\n";
        return false;
    }
    if (!KeyFastPath()) {
        std::cout << "key fast path off: lease refreshes not tested\n";
        return false;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = std::size_t{1} << 20;
    // A 64 KiB block neither T nor its keys touch: a stamp there only moves the tracker generation.
    constexpr std::size_t elsewhere = 6 * 65536;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: lease refreshes not tested\n";
        return false;
    }
    auto* texels = static_cast<std::uint8_t*>(block);
    std::memset(texels, 0x55, bytes);
    std::memset(texels + surfaceBytes, 0xff, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr || !AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "no watched host import: lease refreshes not tested\n";
        return false;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    const auto surface = [](std::uint64_t at, std::uint32_t width, std::uint64_t keys) {
        GuestTextureResource resource{};
        resource.baseAddress = at;
        resource.width = width;
        resource.height = width;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kR64KBX;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        resource.dccAddress = keys;
        return resource;
    };
    const auto resource = surface(address, side, address + surfaceBytes);
    const auto otherResource = surface(address, 512, 0);
    Require(DescribeSurface(resource).guestBytes == surfaceBytes && DescribeSurface(otherResource).guestBytes == bytes, "lease refresh fixture: unexpected surface sizes");
    const auto switchMode = [](const char* name) {
        const char* text = std::getenv(name);
        const auto value = text != nullptr ? std::strtol(text, nullptr, 10) : 0;
        return value == 0 ? 0 : value == 2 ? 2 : 1;
    };
    const int units = switchMode("APS5_EXPERIMENT_PROOF_UNITS");
    const int pins = switchMode("APS5_EXPERIMENT_KEY_PIN");
    const auto proved = [] { return StorageTexture::RefreshesProved(); };
    const auto layers = [](const StorageTexture& target) {
        std::vector<std::uint64_t> generations;
        for (std::uint32_t unit = 0; unit < target.TrackedUnits(); ++unit) generations.push_back(target.LayerGeneration(unit));
        return generations;
    };
    recorder.Sync();
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        recorder.Keep(image);
        const auto lease = [&](bool apply, StorageTexture::LeaseRows rows = {}) {
            rows.apply = apply;
            rows.drain = apply;
            image->RefreshForLease(rows);
            return rows;
        };
        // One lease refresh with its expected verdict; only an applied hit counts as a proved refresh.
        const auto expectVerdict = [&](bool apply, Verdict expected, const std::string& what, StorageTexture::LeaseRows rows = {}) {
            const auto before = proved();
            rows = lease(apply, rows);
            Require(rows.verdict == expected, what + " (verdict " + std::to_string(static_cast<int>(rows.verdict)) + ", expected " + std::to_string(static_cast<int>(expected)) + ")");
            Require(proved() == before + (apply && expected == Verdict::Hit ? 1 : 0), what + ": the proved count moved otherwise than by a hit");
            return rows;
        };
        // Refreshes until one is proved (today's path takes the proof at its unchanged exit).
        const auto settle = [&](const std::shared_ptr<StorageTexture>& target, const char* what) {
            for (int round = 0; round < 4; ++round) {
                const auto before = proved();
                target->Refresh();
                if (proved() == before + 1) return;
            }
            Require(false, what);
        };

        // (a) A proved target: the lease's refresh holds over the surface's and the keys' rows, writing
        // nothing at apply=false; at true it is the proved route.
        settle(image, "lease refresh (a): the target was never proved");
        auto state = image->LeaseState();
        auto rows = expectVerdict(false, Verdict::Hit, "lease refresh (a): a proved target's lease refresh did not hold");
        const auto unapplied = image->LeaseState();
        Require(unapplied.proofGeneration == state.proofGeneration && unapplied.pendingSerial == state.pendingSerial && unapplied.generation == state.generation, "lease refresh (a): apply=false changed the refresh proof");
        Require(!rows.pinned && !rows.unitChecked && rows.surface.address == address && rows.surface.bytes == surfaceBytes && rows.keys.address == resource.dccAddress && rows.keys.bytes == keyCount && rows.surface.generation == state.generation, "lease refresh (a): the rows checked are not the surface's and its keys'");
        expectVerdict(true, Verdict::Hit, "lease refresh (a): apply=true did not hit");

        // (b) A driver store into the surface: stamped, at both settings.
        AgcDriver::GuestMemory::MarkWritten(address + 65536, 4096);
        expectVerdict(false, Verdict::Stamped, "lease refresh (b): a store into the surface did not refuse the lease");
        expectVerdict(true, Verdict::Stamped, "lease refresh (b): a store into the surface did not refuse the applied lease");
        settle(image, "lease refresh (b): the proof was not taken again after a store into the surface");

        // (c) A store over the keys (the DCC variant): stamped.
        AgcDriver::GuestMemory::MarkWritten(resource.dccAddress, keyCount);
        expectVerdict(false, Verdict::Stamped, "lease refresh (c): a store over the keys did not refuse the lease");
        settle(image, "lease refresh (c): the proof was not taken again after a key store");

        // The lease's own rows in the same lock: a target-key row (unchanged since) and a CMASK row (not
        // stored over since), each refusing once stored over.
        {
            const auto now = AgcDriver::GuestMemory::TrackerGeneration();
            StorageTexture::LeaseRows extra;
            extra.targetKeys = {address + elsewhere, 4096, now};
            extra.cmask = {address + elsewhere + 8192, 64, now};
            expectVerdict(false, Verdict::Hit, "lease refresh: unstored target-key and CMASK rows refused the lease", extra);
            AgcDriver::GuestMemory::MarkWritten(address + elsewhere + 8192, 64);
            expectVerdict(false, Verdict::Stamped, "lease refresh: a CMASK row stored over did not refuse the lease", extra);
            extra.cmask = {};
            expectVerdict(false, Verdict::Hit, "lease refresh: an unstamped target-key row refused the lease", extra);
            AgcDriver::GuestMemory::MarkWritten(address + elsewhere + 100, 4);
            expectVerdict(false, Verdict::Stamped, "lease refresh: a stamped target-key row did not refuse the lease", extra);
        }

        // (d) Another image pending with a unit inside T: other pending (the unit inside named where the
        // reason is read: P1 on).
        auto other = StorageTexture::Create(context, detiler, otherResource, 0);
        recorder.Keep(other);
        Require(other->BlockUnits() && other->TrackedUnits() == 16, "lease refresh fixture: the other image is not sixteen block units");
        const auto flushOther = [&] {
            StorageTexture::FlushPending(address, bytes, nullptr, "lease refresh test", PublishScope::None);
            recorder.Sync();
            Require(!other->Dirty(), "lease refresh: the other image is still pending");
        };
        std::vector<bool> inside(16, false);
        inside[1] = true;
        other->MarkUnitsWritten(inside);
        rows = expectVerdict(false, Verdict::OtherPending, "lease refresh (d): another image's unit pending inside the target did not refuse the lease");
        if (units != 0) Require(rows.other == StorageTexture::OtherPending::Inside, "lease refresh (d): the refusal did not name the unit inside");
        flushOther();
        settle(image, "lease refresh (d): the target was not proved once nothing else was pending");

        // (e) A unit inside T marked in O while O is pending already (no pending serial moves): the
        // unit-checked proof, taken with O pending outside T and its keys, refuses it (P1 on). Under
        // P1's verify the unit-checked proof is a miss; with P1 off no proof is kept with O pending.
        std::vector<bool> outside(16, false);
        outside[8] = outside[9] = true;
        other->MarkUnitsWritten(outside);
        image->Refresh();
        if (units == 1) {
            rows = expectVerdict(false, Verdict::Hit, "lease refresh (e) 1: the unit-checked proof did not hold with the other image pending outside");
            Require(rows.unitChecked, "lease refresh (e) 1: the hit was not unit-checked");
            const auto serial = StorageTexture::PendingSerial();
            auto both = outside;
            both[1] = true;
            other->MarkUnitsWritten(both);
            Require(StorageTexture::PendingSerial() == serial, "lease refresh (e): a unit marked in an image already pending moved the pending serial");
            rows = expectVerdict(false, Verdict::OtherPending, "lease refresh (e) 1: a unit inside marked without a serial move did not refuse the unit-checked proof");
            Require(rows.other == StorageTexture::OtherPending::Inside, "lease refresh (e) 1: the refusal did not name the unit inside");
        } else {
            expectVerdict(false, units == 2 ? Verdict::P1Verifying : Verdict::NoProof, units == 2 ? "lease refresh (e) 2: a unit-checked proof answered under P1's verify" : "lease refresh (e) 0: a proof was kept with another image pending over the target");
        }
        flushOther();
        // A store drops a unit-checked proof: the next one is taken plain, with nothing else pending.
        AgcDriver::GuestMemory::MarkWritten(address + 65536, 4096);
        settle(image, "lease refresh (e): the target was not proved once nothing else was pending");

        // (f) The pending serial moved with nothing over the target: a hit, whose apply leaves the new
        // serial in the proof as the twin's Refresh() does.
        auto twin = StorageTexture::Create(context, detiler, resource, 0);
        recorder.Keep(twin);
        settle(twin, "lease refresh (f): the twin was never proved");
        settle(image, "lease refresh (f): the target was not proved");
        Require(!image->LeaseState().unitChecked && !twin->LeaseState().unitChecked, "lease refresh (f): a proof is unit-checked with nothing else pending");
        StorageTexture::BumpPendingSerial();
        const auto bumped = StorageTexture::PendingSerial();
        rows = expectVerdict(false, Verdict::Hit, "lease refresh (f): a pending serial move with nothing over the target refused the lease");
        Require(rows.pendingSerial == bumped && rows.serial == bumped && image->LeaseState().pendingSerial != bumped, "lease refresh (f): apply=false wrote the pending serial, or the hit would not leave the new one");
        expectVerdict(true, Verdict::Hit, "lease refresh (f): apply=true did not hit");
        settle(twin, "lease refresh (f): the twin's refresh was not proved after the serial move");
        Require(image->LeaseState().pendingSerial == bumped && twin->LeaseState().pendingSerial == bumped, "lease refresh (f): the lease's apply and the twin's Refresh() left different pending serials");

        // (g) A completion label pending over the surface: the head of CompletePendingCpuWrites would
        // sync, so the lease misses before anything ran, and nothing synced.
        {
            const std::array<std::byte, 4> value{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
            static_cast<void>(recorder.Commands());
            recorder.NotePendingWrite(address + 64, value.size());
            recorder.AfterCompletions(address + 64, value, 1, 0, false);
            Require(CpuWritesWouldSync(context, address, surfaceBytes) == CpuWriteSync::Label, "lease refresh (g): the pending completion label over the surface would not make CompletePendingCpuWrites sync");
            expectVerdict(false, Verdict::CopiedWriter, "lease refresh (g): a pending completion label over the surface did not refuse the lease");
            expectVerdict(true, Verdict::CopiedWriter, "lease refresh (g): a pending completion label over the surface did not refuse the applied lease");
            Require(recorder.PendingWriteOverlaps(address + 64, value.size()) && Recorder::PendingCompletionLabels() != 0, "lease refresh (g): the lease's refresh synced the pending label");
            recorder.Sync();
        }
        settle(image, "lease refresh (g): the target was not proved after the label landed");
        settle(twin, "lease refresh (g): the twin was not proved after the label landed");

        // (h) apply=true leaves the refresh proof and the proved count as the twin's Refresh() does (the
        // proved route moves no layer).
        const auto layersBefore = layers(*image);
        expectVerdict(true, Verdict::Hit, "lease refresh (h): apply=true did not hit");
        const auto provedTwin = proved();
        twin->Refresh();
        Require(proved() == provedTwin + 1, "lease refresh (h): the twin's refresh was not proved");
        const auto mine = image->LeaseState();
        const auto theirs = twin->LeaseState();
        const auto serialNow = StorageTexture::PendingSerial();
        Require(mine.proofGeneration == mine.generation && theirs.proofGeneration == theirs.generation && mine.pendingSerial == serialNow && theirs.pendingSerial == serialNow && mine.unitChecked == theirs.unitChecked && mine.keyPinned == theirs.keyPinned, "lease refresh (h): the lease's apply and the twin's Refresh() left different refresh proofs");
        Require(layers(*image) == layersBefore, "lease refresh (h): a plain hit moved a layer generation");
        twin.reset();

        // K1: the driver's uncompressed key store pending (as StoreUncompressedOnGpu makes it, without
        // the command), so today's key read answers from the memo. 1: the pinned proof answers the
        // lease, apply=false leaves every layer and the proof alone, and apply=true moves every layer
        // to the generation read in the stamp check's lock. 2: a miss (K1's verify takes it). 0: no
        // proof under the pending store.
        static_cast<void>(recorder.Commands());
        recorder.NotePendingFill(resource.dccAddress, keyCount, 0xff);
        AgcDriver::GuestMemory::MarkWritten(resource.dccAddress, keyCount);
        NoteKeysFillOnGpu(resource.dccAddress, keyCount, DccKeys::Uncompressed);
        image->Refresh();
        if (pins == 1) {
            Require(image->KeyProof().pinNote != 0, "lease refresh (K1): the refresh after the key store did not pin the keys");
            settle(image, "lease refresh (K1) 1: the pinned target was never proved");
            const auto layersPinned = layers(*image);
            const auto statePinned = image->LeaseState();
            AgcDriver::GuestMemory::MarkWritten(address + elsewhere, 4096);
            rows = expectVerdict(false, Verdict::Hit, "lease refresh (K1) 1: the pinned target's lease refresh did not hold");
            const auto afterPredict = image->LeaseState();
            Require(rows.pinned && layers(*image) == layersPinned && afterPredict.generation == statePinned.generation && afterPredict.proofGeneration == statePinned.proofGeneration, "lease refresh (K1) 1: an apply=false run of a pinned target moved a layer or its proof");
            AgcDriver::GuestMemory::MarkWritten(address + elsewhere, 4096);
            const auto before = AgcDriver::GuestMemory::TrackerGeneration();
            rows = expectVerdict(true, Verdict::Hit, "lease refresh (K1) 1: the pinned target's applied lease refresh did not hit");
            const auto after = AgcDriver::GuestMemory::TrackerGeneration();
            const auto advanced = image->LeaseState();
            Require(rows.pinned && rows.current >= before && rows.current <= after && advanced.layersAt && advanced.generation == rows.current && advanced.proofGeneration == rows.current && advanced.keyPinned, "lease refresh (K1) 1: the pinned hit did not move every layer to the generation of its stamp check");
        } else if (pins == 2) {
            Require(image->KeyProof().pinNote != 0, "lease refresh (K1): the refresh after the key store did not pin the keys");
            image->Refresh();
            expectVerdict(false, Verdict::K1Verifying, "lease refresh (K1) 2: a pinned proof answered the lease under K1's verify");
        } else {
            expectVerdict(false, Verdict::NoProof, "lease refresh (K1) 0: a refresh proof was kept under a pending key store with the switch off");
        }
        recorder.Sync();
        other.reset();
    }
    recorder.Sync();
    std::cout << "Lease refreshes (APS5_EXPERIMENT_PROOF_UNITS=" << units << ", APS5_EXPERIMENT_KEY_PIN=" << pins << "): passed\n";
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): two draws into one
// resident DCC + CMASK target, each lookup made as prepareDraw's slot loop makes it
// (TargetLeaseLookup). Between the draws, each of these must make the second lookup a miss: a CPU
// store another walk stamps, a key store, a depth plane drawn over the target, its eviction under a
// tiny budget, an epoch bump, and a replaced cache entry (the full path then returns the new image);
// a CMASK proof of another address must still allow a hit, and without any of them the second lookup
// is a hit (=1) or a verified would-hit (=2). The target is 256x256 SW_64KB_R_X at 0 of a 1 MiB
// watched host import, its keys (0xff) at 256 KiB, its CMASK (0xff, proved expanded beforehand, as
// an earlier draw would have) at 320 KiB; the other target at 512 KiB. With the switch unset nothing
// is leased or counted. Needs the epoch coverage (APS5_DEBUG_COVERAGE=1) for a collect memo key.
// False when nothing could be tested.
bool targetLeaseDrawTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: lease draws not tested\n";
        return false;
    }
    if (!KeyFastPath()) {
        std::cout << "key fast path off: lease draws not tested\n";
        return false;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t bytes = std::size_t{1} << 20;
    constexpr std::size_t keyOffset = 256 * 1024;
    constexpr std::size_t cmaskOffset = 320 * 1024;
    constexpr std::size_t otherOffset = 512 * 1024;
    constexpr std::size_t otherCmaskOffset = 832 * 1024;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: lease draws not tested\n";
        return false;
    }
    auto* memory = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    ColorTarget color{};
    color.address = address;
    color.extent = {side, side};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.componentMapping = 0xe4u;
    color.tileMode = ColorTileMode::RenderTarget;
    color.elementBytes = 4;
    color.bytes = ColorTargetLayout(side, side, color.tileMode, color.elementBytes).Bytes();
    color.surfaceAddress = address;
    color.surfaceExtent = color.extent;
    auto otherColor = color;
    otherColor.address = address + otherOffset;
    otherColor.surfaceAddress = otherColor.address;
    color.dccAddress = address + keyOffset;
    color.cmaskAddress = address + cmaskOffset;
    color.cmaskBytes = CmaskLayout(side, side).Bytes();
    const auto keyCount = DccKeyBytes(color.bytes);
    Require(color.bytes <= keyOffset && color.bytes <= bytes - otherOffset && keyCount <= cmaskOffset - keyOffset && color.cmaskBytes <= 65536 && DescribeSurface(SurfaceForTarget(color)).guestBytes == color.bytes, "lease draw fixture: unexpected target sizes");
    std::memset(memory, 0x55, bytes);
    // The target's bytes hold depths in [0, 1], which the depth plane case reads as its plane.
    for (std::size_t offset = 0; offset + sizeof(float) <= color.bytes; offset += sizeof(float)) {
        const float depth = 0.5f;
        std::memcpy(memory + offset, &depth, sizeof(depth));
    }
    std::memset(memory + keyOffset, 0xff, keyCount);
    std::memset(memory + cmaskOffset, 0xff, color.cmaskBytes);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr || !AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "no watched host import: lease draws not tested\n";
        return false;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    context.nullDescriptors = device.NullDescriptors();
    TextureCache cache(context);
    context.textureCache = &cache;
    // The storage cache and the depth surfaces hold the images this test makes through the slot
    // loop's lookups (they refer to the detiler): let go of them while it lives.
    struct Cleanup {
        const Context& context;
        Recorder& recorder;
        ~Cleanup() {
            recorder.Sync();
            ClearDepthSurfaces(context.device);
            ClearCachedTextures(context.device);
            recorder.Sync();
        }
    } cleanup{context, recorder};
    recorder.Sync();
    AgcDriver::GuestMemory::CollectWritesUncached(address, bytes);
    NoteCmaskProof(color.cmaskAddress, color.cmaskBytes, AgcDriver::GuestMemory::CollectWrites(color.cmaskAddress, color.cmaskBytes));
    AgcDriver::GuestMemory::BumpCollectEpoch();
    if (!AgcDriver::GuestMemory::CurrentCollectMemoKey().Valid()) {
        std::cout << "no collect memo key (APS5_DEBUG_COVERAGE unset): lease draws not tested\n";
        return false;
    }
    const int mode = TargetLeases();
    const auto start = TargetLeaseTotals();
    std::shared_ptr<StorageTexture> resident;
    const auto lookup = [&](const ColorTarget& target) {
        static_cast<void>(recorder.Commands());
        auto image = TargetLeaseLookup(context, target);
        Require(image != nullptr, "lease draws: a target stayed non-resident");
        return image;
    };
    const auto answered = [&](const TargetLeaseCounts& before, const TargetLeaseCounts& after) {
        if (mode == 1) return after.hits == before.hits + 1;
        return mode == 2 && after.wouldHits == before.wouldHits + 1 && after.verified == before.verified + 1;
    };
    // Lookups until one is answered by the lease: the first lookups after a change walk, take the
    // refresh proof and make the lease. Three with the switch unset.
    const auto settle = [&](const char* what) {
        for (int round = 0; round < 8; ++round) {
            const auto before = TargetLeaseTotals();
            resident = lookup(color);
            if (mode == 0 ? round == 2 : answered(before, TargetLeaseTotals())) return;
        }
        Require(false, what);
    };
    const auto expectHit = [&](const char* what) {
        const auto before = TargetLeaseTotals();
        resident = lookup(color);
        if (mode != 0) Require(answered(before, TargetLeaseTotals()), what);
    };
    const auto expectMiss = [&](const char* what) {
        const auto before = TargetLeaseTotals();
        resident = lookup(color);
        const auto after = TargetLeaseTotals();
        Require(after.hits == before.hits && after.wouldHits == before.wouldHits, what);
    };

    settle("lease draws: the target's lookups were never answered by a lease");
    expectHit("lease draws: the second draw with nothing between was not answered by the lease");

    // A CPU store into the target, which another walk stamps.
    memory[100] ^= 1;
    AgcDriver::GuestMemory::CollectWritesUncached(address, color.bytes);
    expectMiss("lease draws: a stamped CPU store into the target was answered from the lease");
    settle("lease draws: no lease after a CPU store");

    // A store of the target's keys.
    AgcDriver::GuestMemory::MarkWritten(color.dccAddress, keyCount);
    expectMiss("lease draws: a key store was answered from the lease");
    settle("lease draws: no lease after a key store");

    // A depth plane drawn over the target: a mirrorable one (D32, swizzle mode Z64KBX, one sample),
    // which DepthSurfaceView marks stale over the target's bytes (not its keys). The slot loop's flush
    // then stores it over the target (a store the lease's stamp test sees) and retires the surface, so
    // the lookup misses. Not tested only where the harness cannot make the depth surface; once it is
    // made, a plane left fresh, a lookup answered from the lease or a plane not stored fails the test.
    {
        DepthTarget depth{address, 0, {side, side}, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
        depth.depthSwizzle = XorSwizzleMode(TextureTileMode::kZ64KBX);
        GuestTextureResource plane{};
        plane.baseAddress = address;
        plane.width = side;
        plane.height = side;
        plane.mipCount = 1;
        plane.allocatedMipCount = 1;
        plane.tileMode = TextureTileMode::kZ64KBX;
        plane.dimension = TextureDimension::k2D;
        plane.format = 22;
        Require(DescribeSurface(plane).guestBytes <= keyOffset, "lease draws fixture: the depth plane reaches the target's keys");
        const auto stores = DepthPlaneStoreCounts().stores;
        bool drawn = false;
        try {
            drawn = DepthSurfaceView(context, depth) != VK_NULL_HANDLE;
        } catch (const std::exception& error) {
            std::cout << "lease draws: a depth surface over the target could not be made here (" << error.what() << ")\n";
        }
        if (drawn) {
            Require(DepthPlanesStaleOver(address, color.bytes), "lease draws: a written depth plane over the target is not stale");
            expectMiss("lease draws: a depth plane over the target was answered from the lease");
            Require(DepthPlaneStoreCounts().stores > stores && !DepthPlanesStaleOver(address, color.bytes), "lease draws: the slot loop's flush did not store the depth plane over the target");
        } else {
            std::cout << "lease draws: the depth plane case not tested\n";
        }
        // ClearDepthSurfaces destroys the surfaces' images at once (it is the device teardown's, run
        // with nothing in flight): the work recorded over the surface (its creation, the slot loop's
        // plane store) must finish first, or the batch goes to the queue over a destroyed image and
        // loses the device.
        recorder.Sync();
        ClearDepthSurfaces(context.device);
        recorder.Sync();
    }
    settle("lease draws: no lease after a depth plane");

    // The target evicted under a tiny budget by another target's creation.
    OverrideCacheBudgetForTest(1);
    static_cast<void>(lookup(otherColor));
    OverrideCacheBudgetForTest(0);
    Require(!resident->Cached(), "lease draws: the tiny budget did not evict the target");
    expectMiss("lease draws: an evicted target was answered from the lease");
    settle("lease draws: no lease after an eviction");

    // An epoch bump.
    AgcDriver::GuestMemory::BumpCollectEpoch();
    expectMiss("lease draws: the first lookup of an epoch was answered from the lease");
    settle("lease draws: no lease after an epoch bump");

    // A CMASK proof of another address moves the proof serial but leaves this CMASK's proof: a hit.
    const auto serial = CmaskProofSerial();
    NoteCmaskProof(address + otherCmaskOffset, 64, AgcDriver::GuestMemory::CollectWrites(address + otherCmaskOffset, 64));
    if (mode != 0) Require(CmaskProofSerial() != serial, "lease draws: a CMASK proof did not move the proof serial");
    expectHit("lease draws: a CMASK proof of another address revoked the lease");

    // A replaced cache entry: the lookup misses and returns the cache's new image (the lookup's
    // Requires run on it).
    const auto* replaced = resident.get();
    ClearCachedTextures(context.device);
    const auto fresh = CachedStorageSurface(context, SurfaceForTarget(color));
    Require(fresh.get() != replaced, "lease draws: the replacement is the image it replaced");
    expectMiss("lease draws: a replaced cache entry was answered from the lease");
    Require(resident == fresh, "lease draws: the lookup after the replacement did not return the cache's new image");
    settle("lease draws: no lease after a replaced entry");

    // Which materializations an answered lookup skips. With the target keys' row in the hit's lock,
    // both: the register clear and, as nothing runs between the lock and it, the CMASK proof. K1
    // (APS5_EXPERIMENT_KEY_PIN=1): the driver's uncompressed key store pending (as
    // StoreUncompressedOnGpu makes it, without the command), so the key read answers from its memo and
    // the target keys' proof is never stable: the pinned hit has no target-key row and runs the
    // register clear live, and then the CMASK proof too, since the live call may sync, store or flush
    // over the CMASK after the lock. =2's prediction skips the same. Not tested with K1 off or
    // verifying (no pin), nor with the switch unset (no lease).
    const auto answeredUsing = [&](TargetLeaseUse& used, const char* what) {
        const auto before = TargetLeaseTotals();
        static_cast<void>(recorder.Commands());
        resident = TargetLeaseLookup(context, color, &used);
        const auto after = TargetLeaseTotals();
        Require(answered(before, after), what);
        return after.targetKeysLive - before.targetKeysLive;
    };
    const char* pinText = std::getenv("APS5_EXPERIMENT_KEY_PIN");
    const auto pins = pinText != nullptr ? std::strtol(pinText, nullptr, 10) : 0;
    if (mode != 0) {
        TargetLeaseUse used;
        const auto live = answeredUsing(used, "lease draws: a lookup with nothing between was not answered by the lease");
        const bool skipsRegisterClear = mode == 1 ? used.skipRegisterClear : used.prediction.skipRegisterClear;
        const bool skipsCmaskProof = mode == 1 ? used.skipCmaskProof : used.prediction.skipCmaskProof;
        Require(skipsRegisterClear && skipsCmaskProof && live == 0, "lease draws: a hit holding the target keys' row did not skip both materializations");
    }
    if (mode != 0 && pins == 1) {
        static_cast<void>(recorder.Commands());
        recorder.NotePendingFill(color.dccAddress, keyCount, 0xff);
        AgcDriver::GuestMemory::MarkWritten(color.dccAddress, keyCount);
        NoteKeysFillOnGpu(color.dccAddress, keyCount, DccKeys::Uncompressed);
        expectMiss("lease draws (K1): a key store pending on the GPU was answered from the lease");
        settle("lease draws (K1): the pinned target's lookups were never answered by a lease");
        const auto pinnedBefore = TargetLeaseTotals();
        TargetLeaseUse used;
        const auto live = answeredUsing(used, "lease draws (K1): the pinned target's second lookup was not answered by the lease");
        const auto pinnedAfter = TargetLeaseTotals();
        if (mode == 1) {
            Require(used.hit && pinnedAfter.pinned == pinnedBefore.pinned + 1 && live == 1, "lease draws (K1) 1: the pinned target's hit was not pinned, or held the target keys' row");
            Require(!used.skipRegisterClear && !used.skipCmaskProof && !used.registerClearWorked && !used.cmaskClearWorked, "lease draws (K1) 1: a hit that ran the register clear live skipped the CMASK proof, or a clear did work");
        } else {
            Require(used.prediction.hit && used.prediction.pinned && !used.prediction.skipRegisterClear && !used.prediction.skipCmaskProof, "lease draws (K1) 2: the pinned prediction held the target keys' row, or skipped the CMASK proof after a live register clear");
        }
        recorder.Sync();
        settle("lease draws (K1): no lease after the pending key store finished");
    }
    recorder.Sync();

    const auto end = TargetLeaseTotals();
    if (mode == 0) Require(end.hits == start.hits && end.wouldHits == start.wouldHits && end.created == start.created && end.refused == start.refused && end.misses == start.misses, "lease draws: a lease was made or counted with the switch unset");
    if (mode == 1) Require(end.wouldHits == start.wouldHits && end.verified == start.verified && end.created > start.created, "lease draws 1: a lease was verified, or none was made");
    if (mode == 2) Require(end.hits == start.hits && end.created > start.created, "lease draws 2: a lease answered a lookup, or none was made");
    Require(end.raced == start.raced, "lease draws: a prediction raced in a single-threaded test");
    std::cout << "Lease draws (APS5_EXPERIMENT_TARGET_LEASES=" << mode << "): passed\n";
    return true;
}


void unchangedCpuStampTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::uint64_t offset = 0x2000;
    constexpr std::size_t bytes = 6 * 65536;
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    const auto pattern = [](std::size_t i) { return static_cast<std::uint8_t>(i * 131u + 7u); };
    for (const bool imported : {false, true}) {
#ifdef _WIN32
        if (imported) {
            std::cout << "unchanged CPU stamps over a watched host import: Linux write watch only\n";
            continue;
        }
#endif
        if (imported && base.hostImportAlignment == 0) {
            std::cout << "host imports unavailable: unchanged CPU stamps over a watched host import not tested\n";
            continue;
        }
        const auto decided = PrepareImportWatch(base);
        if (imported) SetImportWatch(base, ImportWatch::Watch);
        struct Restore {
            const Context& context;
            ImportWatch decided;
            ~Restore() { SetImportWatch(context, decided); }
        } restore{base, decided};
        for (const bool changed : {false, true}) {
            const std::string what = std::string(changed ? "a CPU store that changed a byte" : "a CPU store that left the bytes as they were") + (imported ? " over a watched host import" : " over watched memory");
            void* block = AllocateWatched(bytes, 65536);
            if (block == nullptr) {
                std::cout << "no write watching: unchanged CPU stamps not tested\n";
                return;
            }
            auto* memory = static_cast<std::uint8_t*>(block);
            for (std::size_t i = 0; i < bytes; ++i) memory[i] = pattern(i);
            const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
            const auto address = blockAddress + offset;
            {
                GuestAllocations::Mutation mutation;
                mutation.Add(block, bytes, true, true, true);
            }
            struct Unregister {
                const Context& context;
                void* block;
                std::uint64_t address;
                std::size_t bytes;
                ~Unregister() {
                    {
                        GuestAllocations::Mutation mutation;
                        mutation.Remove(block);
                    }
                    HostImportFor(context, address, bytes);
                    ReleaseWatched(block, bytes);
                }
            } unregister{base, block, blockAddress, bytes};
            AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
            TextureDetiler detiler(base);
            auto context = base;
            if (!imported) context.hostImportAlignment = 0;
            context.detiler = &detiler;
            GuestTextureResource resource{};
            resource.baseAddress = address;
            resource.width = side;
            resource.height = side;
            resource.mipCount = 1;
            resource.tileMode = TextureTileMode::kLinear;
            resource.dimension = TextureDimension::k2D;
            resource.format = 56;
            resource.dstSelX = 4;
            resource.dstSelY = 5;
            resource.dstSelZ = 6;
            resource.dstSelW = 7;
            Require(DescribeSurface(resource).guestBytes == surfaceBytes, "(c) unexpected surface size");
            {
                auto image = StorageTexture::Create(context, detiler, resource, 0);
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                const auto commands = recorder.Commands();
                recorder.Keep(image);
                const VkClearColorValue value{{1.0f, 0.0f, 0.0f, 1.0f}};
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
                for (std::size_t i = 0; i < 4096; ++i) memory[offset + i] = pattern(offset + i);
                if (changed) memory[offset + 16] = static_cast<std::uint8_t>(~pattern(offset + 16));
                AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
                Require(AgcDriver::GuestMemory::Watched(address, surfaceBytes) && (HostImportFor(context, address, surfaceBytes) != nullptr) == imported, "(c) " + what + ": the image is not on the path under test");
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
                std::vector<std::byte> read(bytes);
                AgcDriver::GuestMemory::Read(blockAddress, read);
                const auto texel = [&](std::size_t at) {
                    for (std::size_t c = 0; c < 4; ++c) {
                        if (std::to_integer<std::uint8_t>(read[at + c]) != red[c]) return false;
                    }
                    return true;
                };
                const auto headEnd = static_cast<std::size_t>(65536);
                std::size_t stored = 0, total = 0;
                for (std::size_t at = offset; at < offset + surfaceBytes; at += 4) {
                    ++total;
                    if (texel(at)) ++stored;
                }
                if (!changed) {
                    if (stored != total) throw std::runtime_error(std::string("unchanged CPU stamps: ") + what + " dropped " + std::to_string(total - stored) + " of " + std::to_string(total) + " texels the GPU wrote");
                } else {
                    Require(std::to_integer<std::uint8_t>(read[offset + 16]) == static_cast<std::uint8_t>(~pattern(offset + 16)), "(c) " + what + ": the changed byte was overwritten by the GPU results");
                    for (std::size_t at = headEnd; at < offset + surfaceBytes; at += 4) Require(texel(at), "(c) " + what + ": GPU results outside the block the CPU changed were dropped");
                }
            }
            recorder.Sync();
        }
    }
    std::cout << "unchanged CPU stamps: ok\n";
}

void keptUnitSettleTests(const Device& device, Recorder& recorder) {
    Require(AgcDriver::GuestMemory::WriteWatched(), "(k) kept-unit settle tests require host write watching");
    const auto& base = device.GetContext();
    constexpr std::uint32_t width = 1024;
    constexpr std::uint32_t height = 512;
    constexpr std::size_t surfaceBytes = std::size_t{width} * height * 4;
    constexpr std::size_t unitBytes = 65536;
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    constexpr std::array<std::uint8_t, 4> blue{0, 0, 255, 255};
    for (const bool direct : {false, true}) {
        const std::string what = direct ? "GPU-mapped direct memory" : "watched memory";
        if (direct && base.hostImportAlignment == 0) {
            std::cout << "host imports unavailable: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        struct Backing {
            Recorder& recorder;
            bool direct;
            void* block = nullptr;
            std::int64_t physical = -1;
            ~Backing() {
                try {
                    recorder.Sync();
                } catch (const std::exception& error) {
                    std::cerr << "(k) the recorded work did not finish, the test memory stays mapped: " << error.what() << '\n';
                    return;
                }
                if (direct) {
                    if (block != nullptr) sceKernelMunmap(block, surfaceBytes);
                    if (physical >= 0) sceKernelReleaseDirectMemory(physical, surfaceBytes);
                    return;
                }
                if (block == nullptr) return;
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                ReleaseWatched(block, surfaceBytes);
            }
        } backing{recorder, direct};
        if (direct) {
            Require(sceKernelAllocateDirectMemory(0, 0x7fffffffffll, surfaceBytes, unitBytes, 0, &backing.physical) == 0, "(k) allocate the direct memory");
            Require(sceKernelMapDirectMemory(&backing.block, surfaceBytes, 0x33, 0, backing.physical, unitBytes) == 0, "(k) map the direct memory for the GPU");
        } else {
            backing.block = AllocateWatched(surfaceBytes, unitBytes);
            Require(backing.block != nullptr, "(k) no watched block although write watching is on");
            GuestAllocations::Mutation mutation;
            mutation.Add(backing.block, surfaceBytes, true, true, true);
        }
        auto* memory = static_cast<std::uint8_t*>(backing.block);
        std::memset(memory, 0x5a, surfaceBytes);
        const auto address = reinterpret_cast<std::uint64_t>(backing.block);
        AgcDriver::GuestMemory::CollectWritesUncached(address, surfaceBytes);
        TextureDetiler detiler(base);
        auto context = base;
        if (!direct) context.hostImportAlignment = 0;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = width;
        resource.height = height;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kR64KBX;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        Require(DescribeSurface(resource).guestBytes == surfaceBytes, "(k) unexpected surface size");
        if (direct && !AgcDriver::GuestMemory::Watched(address, surfaceBytes)) {
            std::cout << "GPU-mapped direct memory is not write-watched here: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        Require(AgcDriver::GuestMemory::Watched(address, surfaceBytes), "(k) " + what + " is not watched");
        if (direct && HostImportFor(context, address, surfaceBytes) == nullptr) {
            std::cout << "host import of the direct memory refused: units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        if (direct && !AgcDriver::GuestMemory::Watched(address, surfaceBytes)) {
            std::cout << "the host import of the direct memory took it out of the write watch (its imports are compared): units stored next to a unit kept for the CPU in " << what << " not tested\n";
            continue;
        }
        Require(direct || HostImportFor(context, address, surfaceBytes) == nullptr, "(k) watched memory was imported although host imports are off for it");
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const auto clear = [&](VkClearColorValue value) {
                const auto commands = recorder.Commands();
                recorder.Keep(image);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                image->MarkDirty();
            };
            const auto store = [&] {
                image->WriteBack();
                recorder.Submit();
                device.WaitQueue();
                recorder.Sync();
            };
            const auto unitHolds = [&](std::size_t unit, const std::array<std::uint8_t, 4>& texel) {
                std::vector<std::byte> read(unitBytes);
                AgcDriver::GuestMemory::Read(address + unit * unitBytes, read);
                for (std::size_t i = 0; i < unitBytes; ++i) {
                    if (std::to_integer<std::uint8_t>(read[i]) != texel[i % 4]) return false;
                }
                return true;
            };
            clear({{1.0f, 0.0f, 0.0f, 1.0f}});
            memory[16] = 0x77;
            AgcDriver::GuestMemory::CollectWritesUncached(address, surfaceBytes);
            store();
            std::vector<std::byte> head(unitBytes);
            AgcDriver::GuestMemory::Read(address, head);
            Require(std::to_integer<std::uint8_t>(head[16]) == 0x77, "(k) " + what + ": the CPU store into unit 0 was overwritten by results it was newer than");
            for (std::size_t unit = 1; unit < surfaceBytes / unitBytes; ++unit) Require(unitHolds(unit, red), "(k) " + what + ": the first write-back did not store unit " + std::to_string(unit));
            clear({{0.0f, 0.0f, 1.0f, 1.0f}});
            store();
            for (std::size_t unit = 1; unit < surfaceBytes / unitBytes; ++unit) {
                Require(unitHolds(unit, blue), "(k) " + what + ": unit " + std::to_string(unit) + " kept the results of the earlier write-back; that write-back's own store was taken for a CPU write because a neighbouring unit was kept for the CPU");
            }
            AgcDriver::GuestMemory::Read(address, head);
            Require(std::to_integer<std::uint8_t>(head[16]) == 0x77, "(k) " + what + ": the second write-back stored over unit 0, which the CPU wrote after that unit's generation");
        }
        std::cout << "units stored next to a unit kept for the CPU in " << what << ": ok\n";
    }
}

void settledBaselineTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
#ifdef _WIN32
    std::cout << "generation baseline over a pending store: Linux write watch only\n";
    return;
#endif
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: generation baseline over a pending store not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::uint64_t offset = 0x2000;
    constexpr std::size_t bytes = 6 * 65536;
    const auto decided = PrepareImportWatch(base);
    SetImportWatch(base, ImportWatch::Watch);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{base, decided};
    for (const bool pending : {false, true}) {
        void* block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: generation baseline over a pending store not tested\n";
            return;
        }
        std::memset(block, 0x5a, bytes);
        const auto blockAddress = reinterpret_cast<std::uint64_t>(block);
        const auto address = blockAddress + offset;
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(block, bytes, true, true, true);
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                ReleaseWatched(block, bytes);
            }
        } unregister{base, block, blockAddress, bytes};
        AgcDriver::GuestMemory::CollectWritesUncached(blockAddress, bytes);
        TextureDetiler detiler(base);
        auto context = base;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = side;
        resource.height = side;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kLinear;
        resource.dimension = TextureDimension::k2D;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto surfaceBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
        Require(HostImportFor(context, address, surfaceBytes) != nullptr && AgcDriver::GuestMemory::Watched(address, surfaceBytes), "(b) the surface is not a watched host import");
        static_cast<void>(recorder.Commands());
        if (pending) recorder.NotePendingWrite(address + 4096, 4096);
        const auto before = recorder.Submissions();
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            recorder.Keep(image);
            if (pending) Require(recorder.Submissions() == before, "(b) an upload over a pending recorded store submitted the open batch to read its generation baseline");
        }
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::cout << "generation baseline over a pending store: ok\n";
}

void importWindowTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& base = device.GetContext();
#ifdef _WIN32
    static_cast<void>(recorder);
    static_cast<void>(base);
    std::cout << "the import window: Linux write watch only\n";
#else
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: the import window not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
    void* block = AllocateWatched(bytes, 65536);
    if (block == nullptr) {
        std::cout << "no write watching: the import window not tested\n";
        return;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    Require(ProbeImportWriteProtection(base).failure == nullptr, "(w) the import probe failed");
    const auto decided = PrepareImportWatch(base);
    if (decided == ImportWatch::Unwatch) {
        std::cout << "host imports are compared, not watched: the import window not tested\n";
        return;
    }
    SetImportWatch(base, ImportWatch::Watch);
    struct Restore {
        const Context& context;
        ImportWatch decided;
        ~Restore() { SetImportWatch(context, decided); }
    } restore{base, decided};
    auto* texels = static_cast<std::uint8_t*>(block);
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(texels + surfaceBytes, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = address + surfaceBytes;
    {
        Require(!HostImportCovers(base, address, bytes), "(w) the block is imported before the cache takes its generation");
        const auto cached = CollectWritesUncached(address, surfaceBytes);
        Require(cached != 0 && UnchangedSince(address, surfaceBytes, cached), "(w) the cache's generation is not current before the import");
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto commands = recorder.Commands();
        recorder.Keep(image);
        const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        image->MarkDirty();
        const auto* import = HostImportFor(base, address, bytes);
        if (import == nullptr) {
            std::cout << "host import of the import window block refused: the import window not tested\n";
            recorder.Sync();
            return;
        }
        Require(!import->unwatched && Watched(address, bytes), "(w) an import under the watch decision left the watch");
        CollectWritesUncached(address, surfaceBytes);
        Require(UnchangedSince(address, surfaceBytes, cached), "(w) an import that stored nothing reads as a change to a cache taken before it");
        Require(image->Generation() != 0 && UnchangedSince(address, surfaceBytes, image->Generation()), "(w) an import that stored nothing reads as a change to the pending storage image");
        std::array<std::uint8_t, surfaceBytes / 65536> changed{};
        const std::array<std::uint64_t, surfaceBytes / 65536> generations{cached, cached, cached, cached};
        Require(ChangedBlocks(address, surfaceBytes, generations, changed) && std::all_of(changed.begin(), changed.end(), [](std::uint8_t value) { return value == BlockUnchanged; }), "(w) the pages the import resolved read as written");
        const auto after = CollectWritesUncached(address, surfaceBytes);
        Require(after != 0 && UnchangedSince(address, surfaceBytes, after), "(w) the imported range is not watched again after the import window");
        image->Refresh();
        Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto copyCommands = recorder.Commands();
        RecordMemoryBarrier(context, copyCommands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {side, side, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(copyCommands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, copyCommands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        const auto pixels = readback.Bytes();
        constexpr std::array<std::uint8_t, 4> expected{255, 0, 0, 255};
        for (std::size_t i = 0; i < pixels.size(); ++i) Require(std::to_integer<std::uint8_t>(pixels[i]) == expected[i % 4], "(w) results pending over the import window were dropped");
        std::vector<std::byte> memory(surfaceBytes);
        Read(address, memory);
        for (std::size_t i = 0; i < surfaceBytes; ++i) Require(std::to_integer<std::uint8_t>(memory[i]) == expected[i % 4], "(w) results pending over the import window did not reach guest memory");
        const auto stored = CollectWritesUncached(address, surfaceBytes);
        texels[65536 + 8] = 0x77;
        CollectWritesUncached(address, surfaceBytes);
        Require(WrittenSince(address + 65536, 65536, stored) && !WrittenSince(address, 65536, stored), "(w) a CPU store after the import window is not reported as one");
    }
    recorder.Sync();
#endif
}

void queuedWriteGroupTests() {
    using Write = Recorder::QueuedWrite;
    using Groups = std::vector<std::size_t>;
    const auto paint = [](const std::vector<Write>& writes, const Groups& groups, std::size_t bytes) {
        std::vector<std::uint8_t> memory(bytes, 0);
        std::size_t first = 0;
        for (const auto end : groups) {
            for (const bool computed : {false, true}) {
                for (auto i = first; i < end; ++i) {
                    if (writes[i].computed != computed) continue;
                    for (auto at = writes[i].begin; at < writes[i].end; ++at) memory[at] = static_cast<std::uint8_t>(i + 1);
                }
            }
            first = end;
        }
        return memory;
    };
    const auto programOrder = [](const std::vector<Write>& writes, std::size_t bytes) {
        std::vector<std::uint8_t> memory(bytes, 0);
        for (std::size_t i = 0; i < writes.size(); ++i) {
            for (auto at = writes[i].begin; at < writes[i].end; ++at) memory[at] = static_cast<std::uint8_t>(i + 1);
        }
        return memory;
    };
    const auto recorded = [](const std::vector<Write>& writes) {
        const auto order = Recorder::QueuedWriteOrder(writes);
        std::vector<Write> ordered;
        for (const auto index : order) ordered.push_back(writes[index]);
        const auto groups = Recorder::QueuedWriteGroups(ordered);
        std::vector<std::size_t> sequence;
        std::size_t first = 0;
        for (const auto end : groups) {
            for (const bool computed : {false, true}) {
                for (auto i = first; i < end; ++i) {
                    if (ordered[i].computed == computed) sequence.push_back(order[i]);
                }
            }
            first = end;
        }
        return std::pair{sequence, groups.size()};
    };
    Require(Recorder::QueuedWriteGroups({}).empty(), "no queued writes made a group");
    const std::vector<Write> flagged{{0x100, 0x200, false}, {0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x200, 0x208, false}, {0x400, 0x500, false}, {0x400, 0x4f8, true}, {0x408, 0x500, true}, {0x500, 0x508, false}, {0x800, 0x804, false}};
    Require(recorded(flagged) == std::pair{std::vector<std::size_t>{0, 4, 1, 2, 5, 6, 3, 7, 8}, std::size_t{2}}, "the dumps of a run with a flag after each query slot were not folded once, with the clears before them and the flags after them");
    const std::vector<Write> open{{0x50, 0x58, false}, {0x100, 0x200, false}, {0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x200, 0x208, false}, {0x400, 0x500, false}, {0x400, 0x4f8, true}};
    Require(recorded(open) == std::pair{std::vector<std::size_t>{0, 1, 5, 2, 3, 6, 4}, std::size_t{2}}, "the clear of a slot whose end dump the run does not hold was not moved ahead of the dumps");
    const std::vector<Write> overDump{{0x100, 0x200, false}, {0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x1f0, 0x410, false}, {0x400, 0x4f8, true}};
    Require(Recorder::QueuedWriteOrder(overDump) == std::vector<std::size_t>{0, 1, 2, 3, 4}, "a store over an earlier dump that the later dumps cover only in part was moved ahead of that dump");
    const std::vector<Write> overFlag{{0x100, 0x200, false}, {0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x500, 0x508, false}, {0x400, 0x508, false}, {0x400, 0x4f8, true}, {0x408, 0x500, true}};
    Require(Recorder::QueuedWriteOrder(overFlag) == std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6}, "a clear over an earlier store after a dump was moved ahead of that store");
    const std::vector<Write> query{{0x100, 0x200, false}, {0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x200, 0x300, false}, {0x200, 0x2f8, true}, {0x208, 0x300, true}};
    Require(Recorder::QueuedWriteGroups(query) == Groups{3, 6}, "the clear of the next query slot did not wait for the dumps before it");
    const std::vector<Write> reused{{0x100, 0x1f8, true}, {0x100, 0x200, false}, {0x100, 0x1f8, true}};
    Require(Recorder::QueuedWriteGroups(reused) == Groups{1, 3}, "a store over a computed write of its group did not start a new group");
    const std::vector<Write> apart{{0x100, 0x1f8, true}, {0x1f8, 0x200, false}, {0x300, 0x310, false}};
    Require(Recorder::QueuedWriteGroups(apart) == Groups{1, 3}, "stores after the computed writes joined their group");
    const std::vector<Write> label{{0x100, 0x1f8, true}, {0x108, 0x200, true}, {0x400, 0x408, false}};
    Require(Recorder::QueuedWriteGroups(label) == Groups{2, 3}, "a label after the dumps landed before them");
    const std::vector<Write> stores{{0x100, 0x110, false}, {0x100, 0x110, false}};
    Require(Recorder::QueuedWriteGroups(stores) == Groups{2}, "stores alone were split into groups");
    std::uint64_t seed = 0x9e3779b97f4a7c15ull;
    const auto next = [&](std::uint64_t bound) {
        seed ^= seed << 13u;
        seed ^= seed >> 7u;
        seed ^= seed << 17u;
        return seed % bound;
    };
    constexpr std::size_t bytes = 64;
    for (int round = 0; round < 20000; ++round) {
        std::vector<Write> writes;
        const auto count = 1 + next(12);
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto begin = next(bytes - 1);
            const auto end = begin + 1 + next(std::min<std::uint64_t>(16, bytes - begin));
            writes.push_back({begin, std::min<std::uint64_t>(end, bytes), next(2) == 0});
        }
        const auto groups = Recorder::QueuedWriteGroups(writes);
        Require(!groups.empty() && groups.back() == writes.size() && std::is_sorted(groups.begin(), groups.end()), "queued write groups do not end at the last write in order");
        Require(paint(writes, groups, bytes) == programOrder(writes, bytes), "recording each group's stores before its computed writes changed the final bytes of the program order");
        const auto sequence = recorded(writes).first;
        std::vector<std::size_t> position(writes.size(), writes.size());
        for (std::size_t i = 0; i < sequence.size(); ++i) position[sequence[i]] = i;
        Require(sequence.size() == writes.size() && std::find(position.begin(), position.end(), writes.size()) == position.end(), "the recorded order is not a permutation of the queued writes");
        std::vector<std::uint8_t> memory(bytes, 0);
        for (const auto index : sequence) {
            for (auto at = writes[index].begin; at < writes[index].end; ++at) memory[at] = static_cast<std::uint8_t>(index + 1);
        }
        Require(memory == programOrder(writes, bytes), "the recorded order changed the final bytes of the program order");
        for (std::size_t j = 0; j < writes.size(); ++j) {
            if (writes[j].computed) continue;
            std::vector<bool> covered(bytes);
            for (auto k = j + 1; k < writes.size(); ++k) {
                if (writes[k].computed) std::fill(covered.begin() + writes[k].begin, covered.begin() + writes[k].end, true);
            }
            if (std::any_of(covered.begin() + writes[j].begin, covered.begin() + writes[j].end, [](bool value) { return value; })) continue;
            for (std::size_t i = 0; i < j; ++i) {
                Require(position[i] < position[j], writes[i].computed ? "a store no later dump overlaps was recorded before a dump ahead of it" : "a store no later dump overlaps was recorded before a store ahead of it");
            }
        }
    }
}

constexpr std::uint64_t SampleReady = 1ull << 63u;

void queuedSampleDumpTests(Recorder& recorder, const HostImport& import, std::uint64_t* words, std::uint64_t address, VkDeviceAddress target) {
    constexpr std::uint64_t untouched = 0xaaaaaaaaaaaaaaaaull;
    const auto storeZero = [&](std::uint64_t at, std::size_t bytes) {
        const std::vector<std::byte> zero(bytes);
        recorder.RecordStore(import.buffer, at - import.base, zero, at);
    };
    recorder.Sync();
    const auto base = recorder.SamplesTotal();
    std::fill(words, words + 256, untouched);
    Require(recorder.DumpSamples(target, address), "a dump was not queued");
    Require(recorder.HasQueuedStores() && recorder.QueuedStoreOverlaps(address + 240, 8) && !recorder.QueuedStoreOverlaps(address + 248, 8), "a queued dump does not cover its 16 counters");
    recorder.FlushStoresOverlapping(address + 256, 64);
    Require(recorder.HasQueuedStores(), "a reader of other bytes recorded the queued dump");
    Require(words[0] == untouched, "a queued dump landed before anything recorded it");
    recorder.FlushStoresOverlapping(address + 16, 8);
    Require(!recorder.HasQueuedStores(), "a reader of the dumped bytes left the dump queued");
    recorder.Sync();
    Require(words[0] == (base | SampleReady) && words[2] == SampleReady && words[1] == untouched, "the dump a reader recorded stored the wrong counters");
    std::fill(words, words + 256, untouched);
    storeZero(address, 256);
    Require(recorder.DumpSamples(target, address), "a dump after a clear was not queued");
    storeZero(address + 512, 16);
    Require(recorder.DumpSamples(target + 8, address + 8), "the end dump was not queued");
    Require(recorder.DumpSamples(target + 256, address + 256), "a dump of another slot was not queued");
    storeZero(address + 256, 16);
    Require(recorder.DumpSamples(target + 256 + 8, address + 256 + 8), "a dump after a store over an earlier dump was not queued");
    recorder.Sync();
    Require(words[0] == (base | SampleReady) && words[1] == (base | SampleReady) && words[2] == SampleReady && words[3] == SampleReady, "a clear and its dumps landed out of order");
    Require(words[32] == 0 && words[34] == SampleReady, "a store after a dump of its bytes did not land last");
    Require(words[33] == (base | SampleReady) && words[35] == SampleReady, "a dump after a store over an earlier dump landed before it");
    Require(words[64] == 0 && words[65] == 0 && words[66] == untouched, "a store between dumps was lost");
}

class CoverPass {
public:
    explicit CoverPass(const Context& context) : context(context), target(context, ColorTarget{0, {8, 8}, VK_FORMAT_R8G8B8A8_UNORM, 8 * 8 * 4, 0}, false) {
        try {
            if (!context.graphicsPipelineLibrary) {
                VkAttachmentDescription attachment{};
                attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
                attachment.samples = VK_SAMPLE_COUNT_1_BIT;
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
                const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
                VkSubpassDescription subpass{};
                subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
                subpass.colorAttachmentCount = 1;
                subpass.pColorAttachments = &color;
                VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
                passInfo.attachmentCount = 1;
                passInfo.pAttachments = &attachment;
                passInfo.subpassCount = 1;
                passInfo.pSubpasses = &subpass;
                Check(context.Function<PFN_vkCreateRenderPass>("vkCreateRenderPass")(context.device, &passInfo, nullptr, &renderPass), "vkCreateRenderPass cover");
                const auto view = target.View();
                VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &view;
                framebufferInfo.width = 8;
                framebufferInfo.height = 8;
                framebufferInfo.layers = 1;
                Check(context.Function<PFN_vkCreateFramebuffer>("vkCreateFramebuffer")(context.device, &framebufferInfo, nullptr, &framebuffer), "vkCreateFramebuffer cover");
            }
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout cover");
            const auto module = [&](const std::uint32_t* code, std::size_t bytes) {
                VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                info.codeSize = bytes;
                info.pCode = code;
                VkShaderModule result = VK_NULL_HANDLE;
                Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &info, nullptr, &result), "vkCreateShaderModule cover");
                return result;
            };
            vertex = module(COVER_VERT_SPV, sizeof(COVER_VERT_SPV));
            fragment = module(COVER_FRAG_SPV, sizeof(COVER_FRAG_SPV));
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr};
            stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr};
            VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            const VkViewport viewport{0, 0, 8, 8, 0, 1};
            const VkRect2D scissor{{0, 0}, {8, 8}};
            VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;
            VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.lineWidth = 1;
            VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo blendState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blendState.attachmentCount = 1;
            blendState.pAttachments = &blend;
            VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
            pipelineInfo.pStages = stages.data();
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &assembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &raster;
            pipelineInfo.pMultisampleState = &multisample;
            pipelineInfo.pColorBlendState = &blendState;
            pipelineInfo.layout = layout;
            const VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
            VkPipelineRenderingCreateInfoKHR renderingFormats{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
            renderingFormats.colorAttachmentCount = 1;
            renderingFormats.pColorAttachmentFormats = &colorFormat;
            if (context.graphicsPipelineLibrary) pipelineInfo.pNext = &renderingFormats;
            else pipelineInfo.renderPass = renderPass;
            Check(context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateGraphicsPipelines cover");
        } catch (...) {
            release();
            throw;
        }
    }
    ~CoverPass() { release(); }
    CoverPass(const CoverPass&) = delete;
    CoverPass& operator=(const CoverPass&) = delete;

    bool Draw(Recorder& recorder) const {
        const auto start = recorder.StartDrawPass(PassKey, false, false, PassAccess{});
        const bool continued = start.continued;
        const auto commands = start.commands;
        if (!continued) {
            recorder.PrepareSampleSlot();
            if (context.graphicsPipelineLibrary) {
                if (!transitioned) {
                    transitioned = true;
                    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    barrier.image = target.Image();
                    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                }
                VkRenderingAttachmentInfoKHR attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
                attachment.imageView = target.View();
                attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfoKHR rendering{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
                rendering.renderArea = {{0, 0}, {8, 8}};
                rendering.layerCount = 1;
                rendering.colorAttachmentCount = 1;
                rendering.pColorAttachments = &attachment;
                context.Resolved(&DeviceFunctions::cmdBeginRendering, "vkCmdBeginRenderingKHR")(commands, &rendering);
            } else {
                VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                begin.renderPass = renderPass;
                begin.framebuffer = framebuffer;
                begin.renderArea = {{0, 0}, {8, 8}};
                context.Function<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass")(commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
            }
        }
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        recorder.NoteSampledDraw(commands);
        context.Function<PFN_vkCmdDraw>("vkCmdDraw")(commands, 3, 1, 0, 0);
        recorder.LeaveRenderPassOpen(PassKey, Recorder::NoTiming, false, PassAccess{});
        return continued;
    }

    static constexpr std::uint64_t Samples = 64;

private:
    static constexpr std::uint64_t PassKey = 0xc0de;
    void release() noexcept {
        const auto device = context.device;
        if (pipeline != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device, pipeline, nullptr);
        if (vertex != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device, vertex, nullptr);
        if (fragment != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device, fragment, nullptr);
        if (layout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device, layout, nullptr);
        if (framebuffer != VK_NULL_HANDLE) context.Function<PFN_vkDestroyFramebuffer>("vkDestroyFramebuffer")(device, framebuffer, nullptr);
        if (renderPass != VK_NULL_HANDLE) context.Function<PFN_vkDestroyRenderPass>("vkDestroyRenderPass")(device, renderPass, nullptr);
        pipeline = VK_NULL_HANDLE;
        vertex = fragment = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
        framebuffer = VK_NULL_HANDLE;
        renderPass = VK_NULL_HANDLE;
    }
    const Context& context;
    RenderTarget target;
    mutable bool transitioned = false;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule vertex = VK_NULL_HANDLE;
    VkShaderModule fragment = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

void inPassSampleDumpTests(Recorder& recorder, const Context& context, const CoverPass& cover, std::uint64_t* words, std::uint64_t address, VkDeviceAddress target) {
    const auto samples = [&](std::uint64_t word) { return word & ~SampleReady; };
    recorder.Sync();
    auto base = recorder.SamplesTotal();
    std::fill(words, words + 64, 0);
    Require(!cover.Draw(recorder), "the first draw of a batch continued a pass");
    Require(recorder.DumpSamples(target, address), "a dump inside a render pass was not made on the GPU");
    Require(cover.Draw(recorder), "an occlusion dump ended the render pass");
    Require(recorder.DumpSamples(target + 8, address + 8), "the second dump inside the pass was not made on the GPU");
    Require(cover.Draw(recorder), "the second occlusion dump ended the render pass");
    Require(words[0] == 0 && words[1] == 0, "a dump inside the pass landed before its batch ran");
    recorder.Sync();
    const auto first = samples(words[0]) - base, second = samples(words[1]) - base, total = recorder.SamplesTotal() - base;
    Require((words[0] & SampleReady) != 0 && (words[1] & SampleReady) != 0, "a dump inside a pass lost its ready bit");
    Require(first > 0 && second > first && total > second, "dumps inside one pass do not split its draws' samples in order");
    if (context.occlusionQueryPrecise) Require(first == CoverPass::Samples && second == 2 * CoverPass::Samples && total == 3 * CoverPass::Samples, "dumps inside one pass counted the wrong samples");
    base = recorder.SamplesTotal();
    Require(!cover.Draw(recorder), "the first draw of a new batch continued a pass");
    Require(recorder.DumpSamples(target, address), "a dump before a reader was not made on the GPU");
    recorder.FlushStoresOverlapping(address, 8);
    Require(!cover.Draw(recorder), "a reader of a queued dump recorded it inside the render pass");
    recorder.Sync();
    Require(samples(words[0]) > base && samples(words[0]) < recorder.SamplesTotal(), "the dump a reader recorded in the pass counted the wrong draws");
    base = recorder.SamplesTotal();
    constexpr std::size_t slots = 160;
    std::fill(words, words + slots * 32, 0);
    std::size_t continued = 0;
    for (std::size_t i = 0; i < slots; ++i) {
        if (cover.Draw(recorder)) ++continued;
        Require(recorder.DumpSamples(target + 256 * i, address + 256 * i), "a dump of many in one pass was not made on the GPU");
    }
    recorder.Sync();
    Require(continued + 2 >= slots, "draws between dumps did not keep their render pass");
    std::uint64_t previous = base;
    for (std::size_t i = 0; i < slots; ++i) {
        const auto value = samples(words[32 * i]);
        Require((words[32 * i] & SampleReady) != 0 && value > previous, "a dump past one query pool's slots counted the wrong draws");
        if (context.occlusionQueryPrecise) Require(value == base + (i + 1) * CoverPass::Samples, "a dump past one query pool's slots counted the wrong samples");
        previous = value;
    }
    base = recorder.SamplesTotal();
    constexpr std::size_t passes = 70;
    for (std::size_t i = 0; i < passes; ++i) {
        Require(!cover.Draw(recorder), "a draw after other work continued a pass");
        recorder.Commands();
    }
    recorder.Submit();
    Require(recorder.DumpSamples(target, address), "a dump after many passes was not made on the GPU");
    recorder.Sync();
    const auto folded = samples(words[0]);
    Require(folded > base && folded == recorder.SamplesTotal(), "segments folded at Submit without a dump were lost");
    if (context.occlusionQueryPrecise) Require(folded == base + passes * CoverPass::Samples, "segments folded at Submit without a dump counted the wrong samples");
}

bool pendingKeyStoreTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: pending key stores not tested\n";
        return false;
    }
    constexpr std::size_t surfaceBytes = 65536;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the pending key store block");
    auto* keys = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    if (HostImportFor(context, address, bytes) == nullptr) {
        std::cout << "host import of the pending key store block refused: pending key stores not tested\n";
        return false;
    }
    recorder.Sync();
    std::memset(keys, 0x00, bytes);
    MarkDccUncompressed(context, address, surfaceBytes);
    Require(recorder.PendingWriteOverlaps(address, keyCount) && keys[0] == 0x00, "the uncompressed key store did not stay pending");
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed, "the keys after a pending uncompressed store do not read as uncompressed");
    Require(recorder.PendingWriteOverlaps(address, keyCount), "reading keys the driver's own pending store wrote waited for the GPU");
    Require(CurrentDccKeys(address + 16, surfaceBytes / 2) == DccKeys::Uncompressed, "a key range inside the pending store does not read as uncompressed");
    recorder.NotePendingWrite(address + 16, 16);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a key read with a later writer over the pending store did not wait for it");
    Require(std::all_of(keys, keys + keyCount, [](std::uint8_t key) { return key == 0xff; }), "the uncompressed key store did not land");
    recorder.NotePendingWrite(address, keyCount);
    NoteKeysFillOnGpu(address, keyCount, DccKeys::Clear0001);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of a pending fill did not read as its keys without a wait");
    Require(KnownPendingDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of a pending fill were not known without a wait");
    Require(KnownPendingDccKeys(address + 16, surfaceBytes / 2) == DccKeys::Clear0001, "a key range inside a pending fill was not known");
    Require(!KnownPendingDccKeys(address, 2 * surfaceBytes).has_value() && recorder.PendingWriteOverlaps(address, keyCount), "a range wider than the pending fill was known, or waited for");
    recorder.NotePendingWrite(address, 2 * keyCount);
    Require(!KnownPendingDccKeys(address, surfaceBytes).has_value() && recorder.PendingWriteOverlaps(address, keyCount), "the keys under a later writer over a pending fill were known");
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a later wider writer over a pending fill was not waited for");
    Require(!KnownPendingDccKeys(address, surfaceBytes).has_value(), "keys nothing writes were known as pending keys");
    recorder.NotePendingFill(address, keyCount / 2, 0x40);
    recorder.NotePendingFill(address + keyCount / 2, keyCount / 2, 0x40);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of pending 0001 fills over both halves did not read as 0001 without a wait");
    recorder.NotePendingWrite(address + keyCount / 2, 16);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address, keyCount), "a later writer of unknown bytes over pending fills was not waited for");
    recorder.NotePendingFill(address, keyCount / 2, 0x00);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Mixed && recorder.PendingWriteOverlaps(address, keyCount), "a pending 0000 fill over half of uncompressed keys did not read as mixed without a wait");
    recorder.Sync();
#ifdef _WIN32
    auto* edgeBlock = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * bytes, MEM_RESERVE, PAGE_NOACCESS));
    Require(edgeBlock != nullptr && VirtualAlloc(edgeBlock, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr, "cannot reserve the key block past committed memory");
#else
    void* edgeMapping = mmap(nullptr, 2 * bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(edgeMapping != MAP_FAILED && mprotect(edgeMapping, bytes, PROT_READ | PROT_WRITE) == 0, "cannot reserve the key block past committed memory");
    auto* edgeBlock = static_cast<std::uint8_t*>(edgeMapping);
#endif
    const auto edge = reinterpret_cast<std::uint64_t>(edgeBlock) + bytes - keyCount / 2;
    Require(!AgcDriver::GuestMemory::Accessible(reinterpret_cast<const void*>(edge), keyCount), "keys past the end of committed memory read as accessible");
    recorder.NotePendingFill(edge, keyCount / 2, 0x40);
    Require(CurrentDccKeys(edge, surfaceBytes) == DccKeys::Unreadable && !recorder.PendingWriteOverlaps(edge, keyCount), "keys running past the end of committed memory, with a pending fill over their committed half, were not read as unreadable after a wait");
#ifdef _WIN32
    VirtualFree(edgeBlock, 0, MEM_RELEASE);
#else
    munmap(edgeBlock, 2 * bytes);
#endif
    constexpr std::uint64_t displayBytes = 15u * 9u * 65536u;
    constexpr std::size_t consoleKeys = 49152;
    const auto displayKeys = DccKeyCount(TextureTileMode::kR64KBX, 4, 1920, 1080, displayBytes);
    std::memset(keys, 0x00, consoleKeys);
    MarkDccUncompressed(context, address, displayBytes, displayKeys);
    Require(recorder.PendingWriteOverlaps(address, consoleKeys) && keys[consoleKeys - 1] == 0x00, "the uncompressed key store over a 1920x1080 surface did not stay pending");
    Require(CurrentDccKeys(address + displayBytes / 256, (consoleKeys - displayBytes / 256) * 256) == DccKeys::Uncompressed && recorder.PendingWriteOverlaps(address, consoleKeys), "the keys past one per 256 bytes of a pending 1920x1080 store did not read as uncompressed without a wait");
    Require(CurrentDccKeys(address, displayBytes, displayKeys) == DccKeys::Uncompressed && recorder.PendingWriteOverlaps(address, consoleKeys), "the keys of a pending 1920x1080 store did not read as uncompressed over the console's extent without a wait");
    recorder.NotePendingWrite(address + 40000, 1);
    Require(CurrentDccKeys(address, displayBytes, displayKeys) == DccKeys::Uncompressed && !recorder.PendingWriteOverlaps(address + 40000, 1), "a later writer of key 40000 over a pending 1920x1080 store was not waited for");
    Require(std::all_of(keys, keys + consoleKeys, [](std::uint8_t key) { return key == 0xff; }), "the uncompressed key store over a 1920x1080 surface did not land on all 49152 keys");
    recorder.Sync();
    return true;
}

bool shadowedKeyStoreTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    const auto& context = device.GetContext();
    if (!UnitShadowEnabled() || context.hostImportAlignment == 0) {
        std::cout << "unit shadows or host imports unavailable: keys under a unit shadow not tested\n";
        return false;
    }
    constexpr std::uint64_t unit = 65536;
    constexpr std::size_t bytes = 2 * unit;
    constexpr std::size_t surfaceBytes = 65536;
    constexpr std::size_t keyCount = surfaceBytes / 256;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, unit);
    Require(block != nullptr, "cannot reserve the shadowed key block");
    const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
    Require(section != nullptr, "cannot create the shadowed key block");
    GuestArena::GuestArenaMap_nid_postfix(block, bytes, section, 0, PAGE_READWRITE);
    CloseHandle(section);
#else
    void* block = AllocateWatched(bytes, unit);
    Require(block != nullptr, "cannot allocate the shadowed key block");
#endif
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, bytes); }
    } release{block};
    auto* keys = static_cast<std::uint8_t*>(block);
    std::memset(block, 0xff, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr || !Watched(address, bytes)) {
        std::cout << "the shadowed key block is not imported and watched: keys under a unit shadow not tested\n";
        return false;
    }
    recorder.Sync();
    CollectWritesUncached(address, bytes);
    const auto copyBuffer = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    const auto shadow = [&](std::uint8_t value) {
        auto destination = ShadowDestinationFor(context, *import, address, address + unit);
        Require(destination.has_value() && destination->seedUnits.empty(), "the key unit has no unit shadow destination");
        auto pattern = std::make_shared<Buffer>(context, static_cast<std::size_t>(unit), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memset(pattern->Bytes().data(), value, pattern->Bytes().size());
        const auto commands = recorder.Commands();
        recorder.Keep(pattern);
        recorder.Keep(destination->slab);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkBufferCopy piece{0, destination->offset, unit};
        copyBuffer(commands, pattern->Handle(), destination->buffer, 1, &piece);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        recorder.MarkCovered(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const ShadowedRange range{address, address + unit, destination->slab};
        MarkShadowed(*import, std::span(&range, 1), TrackerGeneration());
        Require(AnyShadowedOverlaps(address, keyCount), "the key unit is not shadowed");
    };
    recorder.NotePendingFill(address, unit, 0x00);
    MarkWritten(address, unit);
    shadow(0x40);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && !recorder.PendingWriteOverlaps(address, keyCount), "the keys under a fresh unit shadow were not read from its published bytes after a wait");
    Require(keys[0] == 0x40 && keys[unit - 1] == 0x40 && keys[unit] == 0xff && !AnyShadowedOverlaps(address, unit), "the fresh unit's publish did not land");
    recorder.Sync();
    shadow(0x22);
    MarkWritten(address, unit);
    recorder.NotePendingFill(address + unit, unit, 0x11);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear0001 && !AnyShadowedOverlaps(address, unit), "the keys under a stale unit shadow were not read from guest memory");
    Require(recorder.PendingWriteOverlaps(address + unit, unit), "a key read under a stale unit shadow waited for a write outside the keys");
    recorder.Sync();
    shadow(0x22);
    recorder.NotePendingFill(address, unit, 0xc0);
    MarkWritten(address, unit);
    Require(CurrentDccKeys(address, surfaceBytes) == DccKeys::Clear1111 && recorder.PendingWriteOverlaps(address, keyCount), "the keys of a pending fill over a stale unit shadow were not read as the fill's keys without a wait");
    Require(!AnyShadowedOverlaps(address, unit), "the stale unit still counts as shadowed after the key read");
    recorder.Sync();
    GuestTextureResource resource{};
    resource.baseAddress = address + unit;
    resource.format = 56;
    resource.dccAddress = address;
    shadow(0x22);
    recorder.NotePendingFill(address, unit, 0xc0);
    MarkWritten(address, unit);
    Require(!WaitForKeyWriters(resource, surfaceBytes).has_value() && recorder.PendingWriteOverlaps(address, keyCount), "the key writers over a stale unit shadow were waited for");
    recorder.Sync();
    Require(keys[0] == 0x40 && keys[unit - 1] == 0x40, "a stale unit's slab bytes were published");
    return true;
}

void sampleDumpTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (context.hostImportAlignment == 0 || !context.bufferDeviceAddress) {
        std::cout << "host imports or buffer device addresses unavailable: occlusion counter dumps on the GPU not tested\n";
        return;
    }
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the occlusion counter block");
    auto* words = static_cast<std::uint64_t*>(block);
    constexpr std::uint64_t untouched = 0xaaaaaaaaaaaaaaaaull;
    std::fill(words, words + 64, untouched);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{context, block, address};
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr || import->address == 0) {
        std::cout << "host import of the occlusion counter block refused: occlusion counter dumps on the GPU not tested\n";
        return;
    }
    const auto target = import->address + (address - import->base);
    recorder.Sync();
    constexpr std::uint64_t ready = 1ull << 63u;
    Require(recorder.DumpSamples(target, address), "the occlusion counters were not dumped on the GPU");
    Require(words[0] == untouched && !recorder.Idle(), "the occlusion counter dump waited for the GPU or landed before its batch ran");
    recorder.Sync();
    const auto begin = recorder.SamplesTotal();
    for (std::size_t db = 0; db < 16; ++db) {
        Require(words[db * 2] == ((db == 0 ? begin : 0) | ready), "an occlusion counter dump stored the wrong value");
        Require(words[db * 2 + 1] == untouched, "an occlusion counter dump stored over the next counter");
    }
    Require(recorder.DumpSamples(target + 8, address + 8), "the second occlusion counter dump was not made on the GPU");
    recorder.Submit();
    recorder.Sync();
    for (std::size_t db = 0; db < 16; ++db) {
        Require(words[db * 2 + 1] == ((db == 0 ? begin : 0) | ready), "the counters changed with nothing drawn between two dumps");
        Require(words[db * 2] == ((db == 0 ? begin : 0) | ready), "the second dump stored over the first");
    }
    Require(recorder.SamplesTotal() == begin, "the sample total moved with nothing drawn");
    for (int i = 0; i < 40; ++i) Require(recorder.DumpSamples(target, address), "a dump past one batch's query slots was not made on the GPU");
    recorder.Sync();
    Require(words[0] == (begin | ready) && recorder.SamplesTotal() == begin, "dumps past one batch's query slots moved the total");
    Require(recorder.QueuesSampleDumps(), "occlusion dumps are recorded at once, not queued with the batch's stores");
    queuedSampleDumpTests(recorder, *import, words, address, target);
    if (!device.Graphics()) {
        std::cout << "the test queue has no graphics: occlusion dumps inside a render pass not tested\n";
        return;
    }
    CoverPass cover(context);
    inPassSampleDumpTests(recorder, context, cover, words, address, target);
}

void cmaskPassTests(const Device& device, Recorder& recorder) {
    namespace GuestMemory = AgcDriver::GuestMemory;
    auto context = device.GetContext();
    context.hostImportAlignment = 0;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    constexpr std::size_t surfaceBytes = 256 * 256 * 4;
    constexpr std::size_t bytes = surfaceBytes + 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the CMASK pass block");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Release {
        const Context& context;
        void* block;
        ~Release() {
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } release{context, block};
    Require(HostImportFor(context, address, bytes) == nullptr, "CMASK copy-path test imported its memory");
    std::memset(block, 0x55, surfaceBytes);
    std::memset(static_cast<std::byte*>(block) + surfaceBytes, 0, 4096);
    ColorTarget color{};
    color.address = address;
    color.extent = {256, 256};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.bytes = surfaceBytes;
    color.componentMapping = 0xe4u;
    color.tileMode = ColorTileMode::RenderTarget;
    color.cmaskAddress = address + surfaceBytes;
    color.cmaskBytes = CmaskBytes(color.extent.width, color.extent.height);
    color.clearWords = {0x80402010u, 0};
    const auto run = [&] { RunColorMetadataPass(context, {ColorMetadataPass::Mode::EliminateFastClear, {color}}); };
    auto broken = context;
    broken.deviceProc = nullptr;
    bool refused = false;
    try {
        RunColorMetadataPass(broken, {ColorMetadataPass::Mode::EliminateFastClear, {color}});
    } catch (const std::runtime_error& error) {
        refused = std::string_view(error.what()).find("missing Vulkan device function resolver") != std::string_view::npos;
        if (!refused) throw;
    }
    Require(refused, "CMASK storage creation failure was silently ignored");
    Require(*static_cast<std::uint8_t*>(block) == 0x55 && ReadDccKeys(color.cmaskAddress, color.cmaskBytes * 256u) == DccKeys::Clear0000, "failed CMASK storage creation changed guest memory");
    run();
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr && *static_cast<std::uint8_t*>(block) == 0x55, "CMASK clear without host imports did not stay on the GPU");
    Require(ReadDccKeys(color.cmaskAddress, color.cmaskBytes * 256u) == DccKeys::Uncompressed, "resident CMASK clear left unresolved metadata");
    color.clearWords[0] = 0x12345678;
    run();
    std::vector<std::uint32_t> stored(surfaceBytes / 4);
    GuestMemory::Read(address, std::as_writable_bytes(std::span(stored)));
    Require(std::ranges::all_of(stored, [](auto word) { return word == 0x80402010u; }), "expanded CMASK changed a pending clear or copy write-back lost it");
    const std::vector<std::byte> clearKeys(4096);
    GuestMemory::Write(color.cmaskAddress, clearKeys);
    run();
    GuestMemory::Read(address, std::as_writable_bytes(std::span(stored)));
    Require(std::ranges::all_of(stored, [](auto word) { return word == 0x12345678u; }), "a second resident CMASK clear retained the first frame");
    recorder.Sync();
}

void metadataPassTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: resident metadata passes not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the metadata pass block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the metadata pass block refused: resident metadata passes not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    ColorTarget color{};
    color.address = address;
    color.extent = {side, side};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.bytes = surfaceBytes;
    color.componentMapping = 0xe4u;
    color.tileMode = ColorTileMode::RenderTarget;
    color.elementBytes = 4;
    color.dccAddress = address + surfaceBytes;
    color.dccPipeAligned = true;
    color.clearWords = {0x80402010u, 0};
    const ColorMetadataPass pass{ColorMetadataPass::Mode::EliminateFastClear, {color}};
    const auto memoryHolds = [&](std::array<std::uint8_t, 4> texel) {
        StorageTexture::FlushPending(address, surfaceBytes, nullptr, "test");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        std::vector<std::uint8_t> stored(surfaceBytes);
        AgcDriver::GuestMemory::Read(address, std::as_writable_bytes(std::span(stored)), 1);
        for (std::size_t i = 0; i < surfaceBytes; ++i) {
            if (stored[i] != texel[i % 4]) return false;
        }
        return true;
    };
    const auto keysUncompressed = [&] { return ReadDccKeys(color.dccAddress, surfaceBytes) == DccKeys::Uncompressed; };
    const auto extent = DccKeyCount(TextureTileMode::kR64KBX, 4, side, side, surfaceBytes);
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0x20, extent);
    RunColorMetadataPass(context, pass);
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr && texels[0] == 0x55, "the register fast clear eliminate did not stay in the resident image");
    Require(keysUncompressed(), "the register fast clear eliminate left the keys compressed");
    Require(CurrentDccKeys(color.dccAddress, surfaceBytes, extent) == DccKeys::Uncompressed, "the register fast clear eliminate did not store uncompressed keys over the target's console DCC extent");
    Require(memoryHolds({0x10, 0x20, 0x40, 0x80}), "the register fast clear eliminate did not store CB_COLOR_CLEAR_WORD");
    Require(std::all_of(keys, keys + extent, [](std::uint8_t key) { return key == 0xff; }), "the stored keys are not uncompressed");
    std::memset(keys, 0xc0, keyCount);
    RunColorMetadataPass(context, {ColorMetadataPass::Mode::DccDecompress, {color}});
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr, "the DCC decompress did not stay in the resident image");
    Require(keysUncompressed(), "the DCC decompress left the keys compressed");
    Require(memoryHolds({0xff, 0xff, 0xff, 0xff}), "the DCC decompress did not store the 1111 value");
    {
        const auto* import = HostImportFor(base, address, bytes);
        Require(import != nullptr, "the metadata pass block lost its import");
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, import->buffer, color.dccAddress - import->base, keyCount, 0x20202020u);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
        recorder.NotePendingWrite(color.dccAddress, keyCount);
        Require(keys[0] == 0xff, "the recorded key fill landed before its batch ran");
        Require(CurrentDccKeys(color.dccAddress, surfaceBytes) == DccKeys::ClearRegister, "the keys read past a pending fast-clear fill");
    }
    std::memset(texels, 0x77, surfaceBytes);
    RunColorMetadataPass(context, pass);
    Require(StorageTexture::FindPending(address, surfaceBytes) != nullptr && texels[0] == 0x77, "a fast clear recorded before the pass fell back to CPU write-back");
    Require(keysUncompressed() && memoryHolds({0x10, 0x20, 0x40, 0x80}), "a fast clear recorded before the pass was not eliminated");
    std::memset(texels, 0x66, surfaceBytes);
    RunColorMetadataPass(context, pass);
    Require(memoryHolds({0x66, 0x66, 0x66, 0x66}), "a pass over uncompressed keys changed the texels");
    ClearCachedTextures(context.device);
    GuestTextureResource plain{};
    plain.baseAddress = address;
    plain.width = side;
    plain.height = side;
    plain.mipCount = 1;
    plain.tileMode = TextureTileMode::kR64KBX;
    plain.dimension = TextureDimension::k2D;
    plain.format = 56;
    plain.dstSelX = 4;
    plain.dstSelY = 5;
    plain.dstSelZ = 6;
    plain.dstSelW = 7;
    const auto unkeyed = CachedStorageSurface(context, plain);
    std::memset(keys, 0xc0, keyCount);
    RunColorMetadataPass(context, pass);
    const auto keyed = StorageTexture::FindPending(address, surfaceBytes);
    Require(keyed != nullptr && keyed != unkeyed && keyed->Descriptor().dccAddress == color.dccAddress, "a pass over a resident image under other keys did not remake it under the target's keys");
    Require(keysUncompressed() && memoryHolds({0xff, 0xff, 0xff, 0xff}), "a pass over a remade resident image did not store the 1111 value");
    {
        const auto* import = HostImportFor(base, address, bytes);
        Require(import != nullptr, "the metadata pass block lost its import");
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, import->buffer, color.dccAddress - import->base, keyCount, 0x20202020u);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
        recorder.NotePendingWrite(color.dccAddress, keyCount);
        NoteKeysFillOnGpu(color.dccAddress, keyCount, DccKeys::ClearRegister);
    }
    Require(KnownPendingDccKeys(color.dccAddress, surfaceBytes) == DccKeys::ClearRegister, "the keys of a pending register clear store were not known");
    const std::array<std::uint32_t, 4> clearPattern{0x80402010u, 0x80402010u, 0x80402010u, 0x80402010u};
    const char* refusal = nullptr;
    Require(!keyed->FillClear(clearPattern, StorageTexture::WholeImage, refusal) && refusal != nullptr && std::strcmp(refusal, "keys pending") == 0, "a clear over pending keys was not refused without the proof");
    refusal = nullptr;
    Require(keyed->FillClear(clearPattern, StorageTexture::WholeImage, refusal, true), "a clear over known pending keys was refused");
    Require(recorder.PendingWriteOverlaps(color.dccAddress, keyCount) && keys[0] == 0xff, "the clear over known pending keys waited for the key store");
    MarkDccUncompressed(context, color.dccAddress, surfaceBytes);
    Require(memoryHolds({0x10, 0x20, 0x40, 0x80}), "the clear recorded behind a pending key store did not store CB_COLOR_CLEAR_WORD");
    Require(std::all_of(keys, keys + keyCount, [](std::uint8_t key) { return key == 0xff; }), "the keys after a clear behind a pending key store are not uncompressed");
}

// The data word positions of a dispatch-cache variant (Driver.cpp's data-only hits): leaves
// located among the runs' words, aliased, unaligned, out-of-run and mismatched ones skipped.
void dataWordPositionsTests() {
    using AgcDriver::DataWordPositions;
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> runs{{0x1000, 0x1010}, {0x2000, 0x2008}};
    const std::vector<std::uint32_t> words{1, 2, 3, 4, 5, 6};
    const std::vector<std::uint32_t> flattened{2, 6, 3};
    std::vector<std::uint32_t> positions, slots;
    {
        const std::vector<std::pair<std::uint32_t, std::uint64_t>> leaves{{1, 0x2004}, {0, 0x1004}, {2, 0x1008}};
        const std::vector<std::uint64_t> otherReads{0x1008};
        const auto counts = DataWordPositions(runs, leaves, otherReads, words, flattened, positions, slots);
        Require(positions == std::vector<std::uint32_t>{1, 5} && slots == std::vector<std::uint32_t>{0, 1}, "data word positions are wrong");
        Require(counts.aliased == 1 && counts.unmapped == 0 && counts.mismatched == 0, "an aliased leaf was not counted");
    }
    {
        const std::vector<std::pair<std::uint32_t, std::uint64_t>> leaves{{0, 0x1006}, {1, 0x3000}, {1, 0x2008}, {2, 0x1000}};
        const auto counts = DataWordPositions(runs, leaves, {}, words, flattened, positions, slots);
        Require(positions.empty() && slots.empty(), "a skipped leaf produced a position");
        Require(counts.unmapped == 3 && counts.mismatched == 1 && counts.aliased == 0, "skipped leaves were counted wrongly");
    }
    {
        // Two pure slots over one dword: both positions kept, in position order.
        const std::vector<std::pair<std::uint32_t, std::uint64_t>> leaves{{2, 0x1008}, {1, 0x2004}, {0, 0x1004}};
        static_cast<void>(DataWordPositions(runs, leaves, {}, words, flattened, positions, slots));
        Require(positions == std::vector<std::uint32_t>{1, 2, 5} && slots == std::vector<std::uint32_t>{0, 2, 1}, "positions are not sorted");
    }
}

std::array<std::uint32_t, 8> linearTextureWords(std::uint64_t address, std::uint32_t width, std::uint32_t height, std::uint32_t format) {
    const auto base40 = address >> 8u;
    std::array<std::uint32_t, 8> words{};
    words[0] = static_cast<std::uint32_t>(base40);
    words[1] = static_cast<std::uint32_t>((base40 >> 32u) & 0xffu) | (format << 20u) | (((width - 1u) & 3u) << 30u);
    words[2] = (((width - 1u) >> 2u) & 0xfffu) | ((height - 1u) << 14u);
    words[3] = 4u | (5u << 3u) | (6u << 6u) | (7u << 9u) | (9u << 28u);
    return words;
}

void imageMemoTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    TextureCache textureCache(context);
    context.textureCache = &textureCache;
    if (!device.NullDescriptors()) {
        std::cout << "nullDescriptor unavailable: image memo not tested\n";
        return;
    }
    context.nullDescriptors = true;
    constexpr std::uint32_t width = 64;
    constexpr std::uint32_t height = 4;
    constexpr std::size_t surfaceBytes = width * height * 4;
    constexpr std::size_t bytes = 2 * 65536;
    void* block = AllocateWatched(bytes, 65536);
    const bool watched = block != nullptr;
#ifdef _WIN32
    if (!watched) block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    if (!watched) block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the image memo block");
    auto* first = static_cast<std::uint8_t*>(block);
    auto* second = first + 65536;
    std::memset(first, 0x11, surfaceBytes);
    std::memset(second, 0x22, surfaceBytes);
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = 1;
    binding.count = 1;
    binding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
    binding.imageSamplers = {0u};
    program.bindings.push_back(binding);
    const CompiledShader compiled{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    std::vector<std::shared_ptr<Texture>> kept;
    const auto sample = [&](const std::uint8_t* surface) {
        const auto words = linearTextureWords(reinterpret_cast<std::uint64_t>(surface), width, height, 56);
        program.bindings[0].guestDescriptor.assign(words.begin(), words.end());
        ShaderResources resources(context, compiled);
        Require(resources.SampledTextures().size() == 1, "the build has no sampled texture");
        kept.push_back(resources.SampledTextures().front());
        return kept.back();
    };
    const auto made = sample(first);
    Require(sample(first) == made, "an unchanged sampled texture was made again");
    const auto other = sample(second);
    Require(other != made && sample(first) == made, "a texture of another descriptor answered for the first");
    ClearCachedTextures(context.device);
    const auto remade = sample(first);
    Require(remade != made && remade != other, "a texture the cache dropped was bound again");
    Require(sample(first) == remade, "the remade texture is not the cache's");
    const auto replaced = sample(second);
    Require(replaced != other && sample(second) == replaced && sample(first) == remade, "the cache does not answer with the replacements");
    const auto firstWords = linearTextureWords(reinterpret_cast<std::uint64_t>(first), width, height, 56);
    const auto secondWords = linearTextureWords(reinterpret_cast<std::uint64_t>(second), width, height, 56);
    const auto bindElements = [&](std::initializer_list<std::array<std::uint32_t, 8>> elements, std::vector<bool> depthCompare) {
        program.bindings[0].count = static_cast<std::uint32_t>(elements.size());
        program.bindings[0].guestDescriptor.clear();
        for (const auto& words : elements) program.bindings[0].guestDescriptor.insert(program.bindings[0].guestDescriptor.end(), words.begin(), words.end());
        program.bindings[0].imageDepthCompare = std::move(depthCompare);
        program.bindings[0].imageSamplers.assign(elements.size(), 0u);
        ShaderResources resources(context, compiled);
        Require(resources.SampledTextures().size() == elements.size(), "the build has the wrong number of sampled textures");
        kept.insert(kept.end(), resources.SampledTextures().begin(), resources.SampledTextures().end());
        return resources.SampledTextures();
    };
    const auto repeated = bindElements({firstWords, firstWords, secondWords, firstWords}, {});
    Require(repeated[0] == remade && repeated[1] == remade && repeated[2] == replaced && repeated[3] == remade, "repeated elements of one build do not bind the cache's textures");
    program.bindings[0].count = 1;
    program.bindings[0].imageDepthCompare.clear();
    program.bindings[0].imageSamplers.assign(1, 0u);
    recorder.Submit();
    device.WaitQueue();
    recorder.Sync();
    kept.clear();
    ClearCachedTextures(context.device);
    if (watched) ReleaseWatched(block, bytes);
#ifdef _WIN32
    else VirtualFree(block, 0, MEM_RELEASE);
#else
    else std::free(block);
#endif
}

// A template's data buffers refreshed by words from a patched compiled result (a data-only hit)
// and back: DataWordsHash() follows the buffers exactly, so a later recipe hit's hash compare
// (RecordedDispatch::DataRefresh::Hash) decides correctly in both directions.
void dataRefreshTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::FlattenedSrt;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.guestDescriptor = {1, 2, 3, 4};
    program.bindings.push_back(binding);
    auto patched = program;
    patched.bindings[0].guestDescriptor[2] = 0x33;
    const CompiledShader original{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    const CompiledShader live{ShaderRecompiler::ShaderStage::Compute, &patched, 0};
    ShaderResources resources(context, original);
    Require(resources.DataWordsHash() == ShaderResources::DataWordsHash(original) && resources.DataWordsHash() != ShaderResources::DataWordsHash(live), "a fresh template's data hash is not its words'");
    Require(!resources.DataWordsDiffer(original) && resources.DataWordsDiffer(live), "the per-word compare disagrees with the words");
    Require(resources.RefreshData(recorder.Commands(), live, &recorder), "a refresh with different words recorded nothing");
    Require(resources.DataWordsHash() == ShaderResources::DataWordsHash(live) && !resources.DataWordsDiffer(live), "the refreshed template's hash is not the patched words'");
    Require(resources.RefreshData(recorder.Commands(), original, &recorder), "the refresh back recorded nothing");
    Require(resources.DataWordsHash() == ShaderResources::DataWordsHash(original), "the refreshed template's hash is not the original words'");
    Require(!resources.RefreshData(recorder.Commands(), original, &recorder), "a refresh with equal words recorded");
    auto replaced = patched;
    replaced.bindings[0].role = ShaderRecompiler::DescriptorRole::GuestSamplers;
    replaced.bindings[0].kind = ShaderRecompiler::DescriptorKind::Sampler;
    const CompiledShader changedResources{ShaderRecompiler::ShaderStage::Compute, &replaced, 0};
    bool rejected = false;
    try {
        static_cast<void>(resources.RefreshData(recorder.Commands(), changedResources, &recorder));
    } catch (const std::exception& error) {
        rejected = std::string_view(error.what()).find("cannot replace bound resources") != std::string_view::npos;
    }
    Require(rejected && resources.DataWordsHash() == ShaderResources::DataWordsHash(original), "a data refresh changed the bound resources");
    recorder.Submit();
    device.WaitQueue();
    recorder.Sync();
}

void multisampleGuestAccessTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0 || !device.NullDescriptors()) {
        std::cout << "4x color targets or nullDescriptor unavailable: multisampled target guest access not tested\n";
        return;
    }
    constexpr std::uint32_t side = 64;
    constexpr std::size_t targetBytes = 65536u;
    constexpr std::size_t bytes = 5u * 65536u;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the multisampled target block");
    std::memset(block, 0, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            ClearMultisampleTargets(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } unregister{base, block};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    context.nullDescriptors = true;
    TextureCache cache(context);
    context.textureCache = &cache;
    const auto clearTargets = [&] {
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        ClearMultisampleTargets(context.device);
    };
    const auto refused = [](const std::function<void()>& access, const std::string& expected, const std::string& what) {
        std::string message;
        try {
            access();
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find(expected) != std::string::npos && message.find("multisampled color target") != std::string::npos, what + " of a resolved-only multisampled target's memory did not throw: " + message);
    };
    ColorTarget target{};
    target.address = address;
    target.surfaceAddress = address;
    target.extent = {side, side};
    target.surfaceExtent = target.extent;
    target.format = VK_FORMAT_R8G8B8A8_UNORM;
    target.bytes = targetBytes;
    target.samples = 4;
    const bool ordered = MayReorderSamples(context, target.samples) && SampleOrderStorageFormat(context, target.format, target.elementBytes, target.samples) != VK_FORMAT_UNDEFINED;
    {
        std::lock_guard gpu(GpuMutex());
        Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the multisampled target has no view");
        Require(MultisampleTargetStorable(context, target) == ordered, ordered ? "a multisampled target whose samples may be reordered was created without storage usage" : "a multisampled target nothing stored to was created with storage usage");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::array<std::byte, 64> texels{};
    refused([&] { AgcDriver::GuestMemory::Read(address + 256u, texels); }, "a CPU read", "a CPU read");
    refused([&] { AgcDriver::GuestMemory::ReadCommitted(address + 256u, texels); }, "a CPU read", "a committed CPU read");
    refused([&] { AgcDriver::GuestMemory::ReadCommittedUnsynced(address + 256u, texels); }, "a CPU read", "a committed CPU read without the flush hook");
    refused([&] { RequireNoMultisampleTarget(address, 64u, "a DMA copy"); }, "a DMA copy", "a DMA copy");
    const auto captureSrt = [](std::uint64_t srt, AgcDriver::ShaderMemory::PendingWriteQuery pendingWrite) {
        static constexpr std::array<std::uint32_t, 6> code{0xf4080100u, 0xfa000000u, 0xbf8cc07fu, 0xe0700000u, 0x80010000u, 0xbf810000u};
        const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u)};
        ShaderRecompiler::RecompileRequest request{};
        request.shader = {ShaderRecompiler::ShaderStage::Compute, 0x24000u, code, 0, {}};
        request.context.waveSize = 32u;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        request.context.compute = ShaderRecompiler::ShaderComputeStageInfo{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
        request.target.vulkanVersion = 0x00401000u;
        request.target.spirvVersion = 0x00010300u;
        request.target.subgroupSize = 32;
        request.target.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
        request.layout.pushConstantSizeBytes = 128;
        AgcDriver::ShaderMemory memory({}, pendingWrite);
        static_cast<void>(memory.Capture(request));
    };
    refused([&] { captureSrt(address + 256u, nullptr); }, "a CPU read", "a shader capture");
    captureSrt(address + 4u * 65536u + 256u, nullptr);
    refused([&] { captureSrt(address + 256u, nullptr); }, "a CPU read", "a shader capture on a pooled page");
    const auto raw = [](std::uint64_t, std::size_t, std::span<std::byte>) { return AgcDriver::ShaderMemory::PendingWrite::Raw; };
    const auto synced = [](std::uint64_t, std::size_t, std::span<std::byte>) { return AgcDriver::ShaderMemory::PendingWrite::Sync; };
    refused([&] { captureSrt(address + 256u, raw); }, "a CPU read", "a shader capture's raw word read");
    refused([&] { captureSrt(address + 256u, synced); }, "a CPU read", "a shader capture's synced word read");
    captureSrt(address + 4u * 65536u + 256u, raw);
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding;
    binding.count = 1;
    binding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
    binding.imageSamplers = {0};
    binding.guestDescriptor = {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (9u << 28u),
        0u,
        0u,
        0u,
        0u,
    };
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    refused([&] {
        std::lock_guard gpu(GpuMutex());
        ShaderResources resources(context, compute);
    }, "a texture read", "a texture read");
    const auto multisampledWords = [&](std::uint32_t width, std::uint32_t sampleShift, std::uint64_t base) {
        return std::vector<std::uint32_t>{
            static_cast<std::uint32_t>(base >> 8u),
            static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (56u << 20u) | (((width - 1u) & 3u) << 30u),
            ((width - 1u) >> 2u) | ((side - 1u) << 14u),
            0xfacu | (sampleShift << 16u) | (14u << 28u),
            0u,
            sampleShift << 4u,
            0u,
            0u,
        };
    };
    const auto sampledProgram = [&](ShaderRecompiler::DescriptorKind kind, std::vector<std::uint32_t> words, ShaderRecompiler::DescriptorImageShape shape = ShaderRecompiler::DescriptorImageShape::Image2DMultisample) {
        ShaderRecompiler::RecompileResult result;
        auto element = binding;
        element.kind = kind;
        if (kind == ShaderRecompiler::DescriptorKind::StorageImage) element.binding = ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
        element.imageShape = shape;
        element.guestDescriptor = std::move(words);
        result.bindings.push_back(std::move(element));
        return result;
    };
    const auto refusedSample = [&](ShaderRecompiler::DescriptorKind kind, std::vector<std::uint32_t> words, const std::string& expected, const std::string& what) {
        const auto refusedProgram = sampledProgram(kind, std::move(words));
        const CompiledShader refusedCompute{ShaderRecompiler::ShaderStage::Compute, &refusedProgram, 0};
        std::string message;
        try {
            std::lock_guard gpu(GpuMutex());
            ShaderResources resources(context, refusedCompute);
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find(expected) != std::string::npos, what + " did not throw: " + message);
    };
    {
        const auto sampled = sampledProgram(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address));
        const auto sampledArray = sampledProgram(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address), ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray);
        auto bothShapes = sampled;
        auto arrayBinding = sampledArray.bindings.front();
        arrayBinding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding + 1u;
        bothShapes.bindings.push_back(std::move(arrayBinding));
        const CompiledShader sampledCompute{ShaderRecompiler::ShaderStage::Compute, &sampled, 0};
        const CompiledShader sampledArrayCompute{ShaderRecompiler::ShaderStage::Compute, &sampledArray, 0};
        const CompiledShader bothShapesCompute{ShaderRecompiler::ShaderStage::Compute, &bothShapes, 0};
        std::lock_guard gpu(GpuMutex());
        ShaderResources first(context, sampledCompute);
        ShaderResources second(context, sampledCompute);
        ShaderResources arrayFirst(context, sampledArrayCompute);
        ShaderResources arraySecond(context, sampledArrayCompute);
        ShaderResources combined(context, bothShapesCompute);
        Require(first.SampledTextures().size() == 1 && first.SampledTextures().front() != nullptr, "the multisampled target was not bound as a sampled texture");
        Require(first.SampledTextures().front()->Layout() == VK_IMAGE_LAYOUT_GENERAL && first.SampledTextures().front()->ViewFormat() == target.format, "the sampled multisampled target is not a view of the target's image in its layout");
        Require(first.SampledTextures().front()->SampledViewRange(false).type == VK_IMAGE_VIEW_TYPE_2D, "Image2DMultisample did not bind a 2D image view");
        Require(second.SampledTextures().front() == first.SampledTextures().front(), "a second lookup of the multisampled target made another texture");
        Require(arrayFirst.SampledTextures().size() == 1 && arrayFirst.SampledTextures().front() != nullptr, "the multisampled array target was not bound as a sampled texture");
        Require(arrayFirst.SampledTextures().front()->SampledViewRange(false).type == VK_IMAGE_VIEW_TYPE_2D_ARRAY, "Image2DMultisampleArray did not bind a 2D array image view");
        Require(arrayFirst.SampledTextures().front()->View() != first.SampledTextures().front()->View(), "Image2DMultisampleArray reused the non-arrayed multisampled image view");
        Require(arraySecond.SampledTextures().front() == arrayFirst.SampledTextures().front(), "a second lookup of the multisampled array target made another texture");
        Require(combined.SampledTextures().size() == 2 && combined.SampledTextures()[0] == first.SampledTextures().front() && combined.SampledTextures()[1] == arrayFirst.SampledTextures().front(), "consecutive multisampled bindings of different shapes did not preserve their distinct views");
        Require(first.MultisampleSources() == std::vector<std::uint64_t>{address} && !first.Reusable(), "resources sampling a multisampled target are kept for reuse without the barrier their lookup records");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 3u, address), "as a 64x64 texture of 8 samples", "sampling a 4x target as an 8x texture");
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side / 2u, 2u, address), "as a 32x64 texture of 4 samples", "sampling a 4x target as a narrower texture");
    refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address + 4u * 65536u), "which no draw rendered to and whose tile mode 0 has no modeled multisampled color footprint", "sampling a linear multisampled texture no draw rendered to");
    if (context.storageImageMultisample) {
        const auto stored = sampledProgram(ShaderRecompiler::DescriptorKind::StorageImage, multisampledWords(side, 2u, address));
        const CompiledShader storedCompute{ShaderRecompiler::ShaderStage::Compute, &stored, 0};
        std::lock_guard gpu(GpuMutex());
        const auto rendered = MultisampleTargetView(context, target);
        Require(MultisampleTargetStorable(context, target) == ordered, "sampling the multisampled target changed its storage usage");
        ShaderResources resources(context, storedCompute);
        Require(MultisampleTargetStorable(context, target) && (MultisampleTargetView(context, target) != rendered) != ordered, "a storage binding did not move the multisampled target to an image with storage usage, or moved one that had it");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    } else {
        refusedSample(ShaderRecompiler::DescriptorKind::StorageImage, multisampledWords(side, 2u, address), "as a storage image at its sample count", "storage access to a multisampled target without shaderStorageImageMultisample");
    }
    refused([&] { AgcDriver::GuestMemory::Read(address + 3u * 65536u + 256u, texels); }, "a CPU read", "a CPU read in the padding of a later sample plane");
    AgcDriver::GuestMemory::Read(address + 4u * 65536u, texels);
    clearTargets();
    target.tileMode = ColorTileMode::RenderTarget;
    {
        std::lock_guard gpu(GpuMutex());
        Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the tiled multisampled target has no view");
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    refused([&] { AgcDriver::GuestMemory::Read(address + targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the tiled multisampled target");
    AgcDriver::GuestMemory::Read(address + targetBytes, texels);
    clearTargets();
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_2_BIT) != 0) {
        target.extent = {128, 64};
        target.surfaceExtent = target.extent;
        target.samples = 2;
        {
            std::lock_guard gpu(GpuMutex());
            Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the 2x tiled multisampled target has no view");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        refused([&] { AgcDriver::GuestMemory::Read(address + 2u * targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the 2x tile-height padding");
        AgcDriver::GuestMemory::Read(address + 2u * targetBytes, texels);
        clearTargets();
    }
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_8_BIT) != 0) {
        target.extent = {64, 32};
        target.surfaceExtent = target.extent;
        target.samples = 8;
        {
            std::lock_guard gpu(GpuMutex());
            Require(MultisampleTargetView(context, target) != VK_NULL_HANDLE, "the 8x tiled multisampled target has no view");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        refused([&] { AgcDriver::GuestMemory::Read(address + 2u * targetBytes - texels.size(), texels); }, "a CPU read", "a CPU read inside the 8x tile-height padding");
        AgcDriver::GuestMemory::Read(address + 2u * targetBytes, texels);
        clearTargets();
    }
    AgcDriver::GuestMemory::Read(address + 256u, texels);
    target.extent = {side, side};
    target.surfaceExtent = target.extent;
    target.samples = 4;
    VkImageView previous = VK_NULL_HANDLE;
    std::shared_ptr<Texture> previousTexture;
    for (const bool removeAllocation : {false, true}) {
        VkImageView view = VK_NULL_HANDLE;
        std::shared_ptr<Texture> texture;
        {
            const auto sampled = sampledProgram(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address));
            const CompiledShader sampledCompute{ShaderRecompiler::ShaderStage::Compute, &sampled, 0};
            std::lock_guard gpu(GpuMutex());
            view = MultisampleTargetView(context, target);
            Require(view != VK_NULL_HANDLE && view != previous, "a remapped allocation reused its retired multisampled image");
            ShaderResources resources(context, sampledCompute);
            texture = resources.SampledTextures().front();
            Require(texture != nullptr && texture != previousTexture, "a remapped allocation sampled its retired multisampled image");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        {
            GuestAllocations::Mutation mutation;
            mutation.Protect(block, bytes, true, true, true, [] {});
        }
        refused([&] { RequireNoMultisampleTarget(address, 64u, "a CPU read"); }, "a CPU read", "a read after protection changed");
        bool failed = false;
        try {
            GuestAllocations::Mutation mutation;
            mutation.Unmap(block, bytes, [](const void*, std::size_t, const void*, bool) { throw std::runtime_error("unmap failed"); });
        } catch (const std::runtime_error&) {
            failed = true;
        }
        Require(failed, "the failed host unmap did not throw");
        refused([&] { RequireNoMultisampleTarget(address, 64u, "a CPU read"); }, "a CPU read", "a read after a failed unmap");
        {
            GuestAllocations::Mutation mutation;
            if (removeAllocation) mutation.Remove(block);
            else mutation.Unmap(block, bytes, [](const void*, std::size_t, const void*, bool) {});
            mutation.Add(block, bytes, true, true, true);
        }
        RequireNoMultisampleTarget(address, bytes, "a texture read");
        AgcDriver::GuestMemory::Read(address + 256u, texels);
        refusedSample(ShaderRecompiler::DescriptorKind::SampledImage, multisampledWords(side, 2u, address), "which no draw rendered to", "sampling a multisampled target whose allocation was unmapped");
        previous = view;
        previousTexture = texture;
    }
    clearTargets();
    std::cout << "Multisampled target guest access tests passed\n";
}

void fmaskDecompressMsaaTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if ((base.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0 || base.hostImportAlignment == 0) {
        std::cout << "4x color targets or host imports unavailable: FMASK decompress not tested\n";
        return;
    }
    constexpr std::uint32_t side = 128;
    constexpr std::size_t targetBytes = 65536u;
    constexpr std::size_t cmaskBytes = 4096u;
    constexpr std::size_t destBytes = targetBytes;
    constexpr std::size_t bytes = 5u * 65536u + cmaskBytes + 2u * destBytes;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, (bytes + 65535u) & ~std::size_t{65535u});
#endif
    Require(block != nullptr, "cannot allocate the FMASK decompress block");
    std::memset(block, 0, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto cmaskAddress = address + 5u * 65536u;
    const auto destAddress = cmaskAddress + cmaskBytes;
    const auto destAddress2 = destAddress + destBytes;
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            ClearMultisampleTargets(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } unregister{base, block};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    ColorTarget source{};
    source.address = address;
    source.surfaceAddress = address;
    source.extent = {side, side};
    source.surfaceExtent = source.extent;
    source.format = VK_FORMAT_R8G8B8A8_UNORM;
    source.elementBytes = 4;
    source.bytes = targetBytes;
    source.samples = 4;
    source.cmaskAddress = cmaskAddress;
    source.cmaskBytes = cmaskBytes;
    source.cmaskFastClear = true;
    source.clearWords = {0x11223344u, 0};
    ColorTarget destination{};
    destination.address = destAddress;
    destination.surfaceAddress = destAddress;
    destination.extent = {side, side};
    destination.surfaceExtent = destination.extent;
    destination.format = VK_FORMAT_R8G8B8A8_UNORM;
    destination.elementBytes = 4;
    destination.bytes = destBytes;
    destination.samples = 1;
    destination.tileMode = ColorTileMode::RenderTarget;
    {
        std::lock_guard gpu(GpuMutex());
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::FmaskDecompress, {source}});
        auto recolored = source;
        recolored.clearWords = {0x55667788u, 0};
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination}, recolored});
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::vector<std::uint32_t> resolved(destBytes / 4);
    AgcDriver::GuestMemory::Read(destAddress, std::as_writable_bytes(std::span(resolved)));
    Require(std::ranges::all_of(resolved, [](auto word) { return word == 0x11223344u; }), "FMASK decompress did not materialize the seeded clear to the host multisampled image");
    source.clearWords = {0x55667788u, 0};
    ColorTarget destination2 = destination;
    destination2.address = destAddress2;
    destination2.surfaceAddress = destAddress2;
    {
        std::lock_guard gpu(GpuMutex());
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::FmaskDecompress, {source}});
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination2}, source});
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    std::vector<std::uint32_t> resolved2(destBytes / 4);
    AgcDriver::GuestMemory::Read(destAddress2, std::as_writable_bytes(std::span(resolved2)));
    Require(std::ranges::all_of(resolved2, [](auto word) { return word == 0x11223344u; }), "subsequent FMASK decompress re-cleared the multisampled target");
    std::cout << "FMASK decompress MSAA clear and resolve tests passed\n";
}

void tiledResolveTests(const Device& device, Recorder& recorder) {
    namespace GuestMemory = AgcDriver::GuestMemory;
    auto context = device.GetContext();
    if ((context.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0 || !device.Graphics()) {
        std::cout << "4x color targets or graphics unavailable: CB resolves into tiled targets not tested\n";
        return;
    }
    context.hostImportAlignment = 0;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    constexpr std::uint32_t side = 128;
    constexpr std::size_t targetBytes = side * side * 4u;
    constexpr std::size_t sourceBytes = targetBytes * 4u;
    constexpr std::size_t bytes = targetBytes + sourceBytes + 65536u;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the tiled resolve block");
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Release {
        const Context& context;
        void* block;
        ~Release() {
            ClearMultisampleTargets(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } release{context, block};
    ColorTarget source{};
    source.address = address + targetBytes;
    source.surfaceAddress = source.address;
    source.extent = {side, side};
    source.surfaceExtent = source.extent;
    source.format = VK_FORMAT_R8G8B8A8_UNORM;
    source.bytes = targetBytes;
    source.componentMapping = 0xe4u;
    source.tileMode = ColorTileMode::RenderTarget;
    source.samples = 4;
    source.cmaskAddress = address + targetBytes + sourceBytes;
    source.cmaskBytes = 4096;
    source.cmaskFastClear = true;
    const std::array<std::pair<ColorTileMode, std::uint32_t>, 3> modes{{{ColorTileMode::Standard4KB, 0x80402010u}, {ColorTileMode::Standard64KB, 0x10204080u}, {ColorTileMode::RenderTarget, 0x40801020u}}};
    for (const auto& [mode, clear] : modes) {
        const auto name = std::to_string(static_cast<std::uint32_t>(mode));
        std::memset(block, 0x55, targetBytes);
        std::memset(static_cast<std::byte*>(block) + targetBytes + sourceBytes, 0, 4096);
        ColorTarget destination{};
        destination.address = address;
        destination.surfaceAddress = address;
        destination.extent = {side, side};
        destination.surfaceExtent = destination.extent;
        destination.format = VK_FORMAT_R8G8B8A8_UNORM;
        destination.bytes = targetBytes;
        destination.componentMapping = 0xe4u;
        destination.tileMode = mode;
        source.clearWords = {clear, 0};
        RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination}, source});
        std::vector<std::uint32_t> resolved(targetBytes / 4);
        GuestMemory::Read(address, std::as_writable_bytes(std::span(resolved)));
        const auto texels = std::ranges::count(resolved, clear);
        Require(texels == static_cast<std::ptrdiff_t>(resolved.size()), "a CB resolve into a tile mode " + name + " target left " + std::to_string(resolved.size() - static_cast<std::size_t>(texels)) + " texels without the resolved color");
        recorder.Sync();
        ClearMultisampleTargets(context.device);
        ClearCachedTextures(context.device);
    }
    {
        ColorTarget destination{};
        destination.address = address;
        destination.extent = {side, side};
        destination.format = VK_FORMAT_R8G8B8A8_UNORM;
        destination.bytes = targetBytes;
        destination.tileMode = ColorTileMode::Linear;
        std::string message;
        try {
            RunColorMetadataPass(context, {ColorMetadataPass::Mode::Resolve, {destination}, source});
        } catch (const std::exception& error) {
            message = error.what();
        }
        Require(message.find("without a resident image is not implemented") != std::string::npos, "a CB resolve into a linear target did not throw: " + message);
    }
    recorder.Sync();
}

void depthSurfaceProofTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (!device.NullDescriptors()) {
        std::cout << "nullDescriptor unavailable: depth surface fast proof not tested\n";
        return;
    }
    constexpr std::uint32_t side = 64;
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the depth surface block");
    std::memset(block, 0, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        ~Unregister() {
            ClearDepthSurfaces(context.device);
            ClearCachedTextures(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
#ifdef _WIN32
            VirtualFree(block, 0, MEM_RELEASE);
#else
            std::free(block);
#endif
        }
    } unregister{base, block};
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    context.nullDescriptors = true;
    TextureCache cache(context);
    context.textureCache = &cache;
    const DepthTarget target{address, 0, {side, side}, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    Require(DepthSurfaceView(context, target) != VK_NULL_HANDLE, "the depth surface has no view");
    ShaderRecompiler::RecompileResult program;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding;
    binding.count = 1;
    binding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
    binding.imageSamplers = {0};
    binding.guestDescriptor = {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (22u << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (9u << 28u),
        0u,
        0u,
        0u,
        0u,
    };
    program.bindings.push_back(binding);
    const CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    bool reusable = false;
    std::array<bool, 2> proved{};
    std::array<ShaderResources::ProofReport, 2> reports{};
    {
        ShaderResources resources(context, compute);
        reusable = resources.Reusable();
        for (std::size_t use = 0; reusable && use < reports.size(); ++use) proved[use] = resources.Revalidate(compute, &reports[use]);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    Require(reusable, "a set sampling only a depth surface is not reusable");
    for (std::size_t use = 0; use < reports.size(); ++use) {
        const auto name = std::to_string(use + 2);
        Require(proved[use], "a set sampling an unchanged depth surface failed its proof on use " + name);
        Require(reports[use].path == ShaderResources::ProofPath::Fast, "a set sampling an unchanged depth surface left the fast proof for the full walk on use " + name);
    }
    const auto resource = DecodeTextureResource(binding.guestDescriptor);
    const VkComponentMapping identity{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    const auto texture = DepthSurfaceTexture(context, binding.guestDescriptor, resource, identity);
    Require(texture != nullptr && DepthSurfaceHolds(context, resource, texture.get()), "a 2D depth lookup is not fast-provable");
    auto arrayed = resource;
    arrayed.dimension = TextureDimension::k2DArray;
    Require(!DepthSurfaceHolds(context, arrayed, texture.get()), "a 2D-array depth lookup took the fast proof instead of the full walk");
    auto layered = resource;
    layered.baseArray = 1;
    Require(!DepthSurfaceHolds(context, layered, texture.get()), "a layered depth lookup took the fast proof instead of the full walk");
}

}

class SampleProgram {
public:
    SampleProgram(const Context& context, Recorder& recorder, std::span<const std::uint32_t> code = SAMPLE_LOD_SPV) : context(context), recorder(recorder), result(context, sizeof(float) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 2};
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &setLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &push;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = code.size_bytes();
        moduleInfo.pCode = code.data();
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        pipelineInfo.layout = pipelineLayout;
        Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        Check(context.Function<PFN_vkCreateSampler>("vkCreateSampler")(context.device, &samplerInfo, nullptr, &sampler), "vkCreateSampler");
        const VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = sizes;
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
    }
    ~SampleProgram() {
        context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        context.Function<PFN_vkDestroySampler>("vkDestroySampler")(context.device, sampler, nullptr);
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, pipelineLayout, nullptr);
        context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }
    SampleProgram(const SampleProgram&) = delete;
    SampleProgram& operator=(const SampleProgram&) = delete;

    float Red(VkImageView view, VkImageLayout layout, float lod, float layer = 0) {
        VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = pool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &setLayout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocateInfo, &set), "vkAllocateDescriptorSets");
        const VkDescriptorImageInfo image{sampler, view, layout};
        const VkDescriptorBufferInfo buffer{result.Handle(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        writes[0].dstSet = set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &image;
        writes[1].dstSet = set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &buffer;
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 2, writes, 0, nullptr);
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        const float parameters[]{lod, layer};
        context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(parameters), parameters);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, 1, 1, 1);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        recorder.Sync();
        Check(context.Function<PFN_vkFreeDescriptorSets>("vkFreeDescriptorSets")(context.device, pool, 1, &set), "vkFreeDescriptorSets");
        float texel[4];
        std::memcpy(texel, result.Bytes().data(), sizeof(texel));
        return texel[0];
    }

private:
    const Context& context;
    Recorder& recorder;
    Buffer result;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
};

void expectRed(float got, float want, const char* what) {
    if (std::abs(got - want) > 1.5f / 255.0f) throw std::runtime_error(std::string(what) + ": read " + std::to_string(got) + ", expected " + std::to_string(want));
}

void minLodTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    if (!context.imageViewMinLod) {
        std::cout << "VK_EXT_image_view_min_lod unavailable: minimum LOD clamp not tested\n";
        return;
    }
    TextureDetiler detiler(context);
    GuestTextureResource resource{};
    resource.baseAddress = 0x100000;
    resource.width = 8;
    resource.height = 8;
    resource.mipCount = 4;
    resource.baseLevel = 0;
    resource.lastLevel = 3;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    const auto geometry = DescribeSurface(resource);
    Require(geometry.mips.size() == 4, "the min LOD test surface has an unexpected mip count");
    const std::array<std::uint8_t, 4> levels{0x00, 0x40, 0x80, 0xff};
    std::vector<std::byte> snapshot(static_cast<std::size_t>(geometry.guestBytes));
    for (std::size_t level = 0; level < levels.size(); ++level) {
        const auto& mip = geometry.mips[level];
        std::fill_n(snapshot.begin() + static_cast<std::ptrdiff_t>(mip.tiledOffset), static_cast<std::size_t>(mip.tiledSize), std::byte{levels[level]});
    }
    SampleProgram program(context, recorder);
    const VkComponentMapping identity{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    const auto sample = [&](std::uint32_t minLod, float lod, std::uint32_t baseLevel = 0) {
        auto described = resource;
        described.minLod = minLod;
        described.baseLevel = baseLevel;
        Texture texture(context, detiler, described, identity, snapshot);
        return program.Red(texture.View(), texture.Layout(), lod);
    };
    const auto unorm = [&](std::size_t level) { return levels[level] / 255.0f; };
    expectRed(sample(0, 0.0f), unorm(0), "minimum LOD clamp: no clamp reads level 0");
    expectRed(sample(0x100, 0.0f), unorm(1), "minimum LOD clamp: MIN_LOD 1 reads level 1 at LOD 0");
    const auto fractional = sample(0x180, 0.0f);
    if (std::abs(fractional - unorm(1)) <= 1.5f / 255.0f) {
        std::cout << "minimum LOD clamp: the device takes the integer part of a fractional view minimum LOD\n";
    } else {
        expectRed(fractional, (unorm(1) + unorm(2)) / 2.0f, "minimum LOD clamp: MIN_LOD 1.5 blends levels 1 and 2");
    }
    expectRed(sample(0x100, 2.0f), unorm(2), "minimum LOD clamp: MIN_LOD 1 lowered LOD 2");
    expectRed(sample(0xfff, 0.0f), unorm(3), "minimum LOD clamp: MIN_LOD past the last level reads the last level");
    expectRed(sample(0x200, 0.0f, 1), unorm(2), "minimum LOD clamp: MIN_LOD 2 over a view from level 1 reads level 2");
    expectRed(sample(0x100, 0.0f, 1), unorm(1), "minimum LOD clamp: MIN_LOD at the view's base level reads its base level");
}

void singleCubeTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    SampleProgram program(context, recorder, SAMPLE_ARRAY_SPV);
    const VkComponentMapping identity{};
    for (const auto first : {0u, 5u, 6u}) {
        std::array<std::uint32_t, 8> words{0x1000u, (56u << 20u) | (3u << 30u), 15u | (63u << 14u), 0xb0000facu, first | (first << 16u), 0, 0, 0};
        auto resource = DecodeTextureResource(words);
        const auto geometry = DescribeSurface(resource);
        std::vector<std::byte> memory(static_cast<std::size_t>(geometry.guestBytes) + 256);
        auto* surface = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255});
        for (std::uint32_t layer = 0; layer < geometry.imageLayers; ++layer) {
            std::memset(surface + geometry.GuestLayerOffset(layer), 16 * (layer + 1), static_cast<std::size_t>(geometry.layerBytes));
        }
        resource.baseAddress = reinterpret_cast<std::uint64_t>(surface);
        const std::span<const std::byte> snapshot(surface, static_cast<std::size_t>(geometry.guestBytes));
        Texture texture(context, detiler, resource, identity, snapshot);
        auto storage = StorageTexture::Create(context, detiler, resource, 0);
        recorder.Keep(storage);
        Texture storageView(context, storage, resource, identity);
        for (std::uint32_t face = 0; face < 6; ++face) {
            const auto expected = 16.0f * (first + face + 1) / 255.0f;
            expectRed(program.Red(texture.View(), texture.Layout(), 0, face), expected, "single cube snapshot sampled the wrong face");
            expectRed(program.Red(storageView.View(), storageView.Layout(), 0, face), expected, "single cube storage view sampled the wrong face");
        }
    }
}

void cubeExtraSliceTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    SampleProgram program(context, recorder, SAMPLE_ARRAY_SPV);
    const VkComponentMapping identity{};
    std::array<std::uint32_t, 8> words{0x1000u, (56u << 20u) | (3u << 30u), 15u | (63u << 14u), 0xb0000facu, 6u, 0, 0, 0};
    auto resource = DecodeTextureResource(words);
    const auto geometry = DescribeSurface(resource);
    Require(geometry.imageLayers == 7u, "a cube with a slice past its last whole cube does not allocate it");
    std::vector<std::byte> memory(static_cast<std::size_t>(geometry.guestBytes) + 256);
    auto* surface = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255});
    for (std::uint32_t layer = 0; layer < geometry.imageLayers; ++layer) {
        std::memset(surface + geometry.GuestLayerOffset(layer), 16 * (layer + 1), static_cast<std::size_t>(geometry.layerBytes));
    }
    resource.baseAddress = reinterpret_cast<std::uint64_t>(surface);
    const std::span<const std::byte> snapshot(surface, static_cast<std::size_t>(geometry.guestBytes));
    Texture texture(context, detiler, resource, identity, snapshot);
    auto storage = StorageTexture::Create(context, detiler, resource, 0);
    recorder.Keep(storage);
    Texture storageView(context, storage, resource, identity);
    const auto wordCube = [&words](std::vector<std::byte>& backing, std::uint32_t format, const std::function<std::uint32_t(std::uint32_t)>& word) {
        auto cubeWords = words;
        cubeWords[1] = (format << 20u) | (3u << 30u);
        auto described = DecodeTextureResource(cubeWords);
        const auto layout = DescribeSurface(described);
        backing.assign(static_cast<std::size_t>(layout.guestBytes) + 256, std::byte{});
        auto* base = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(backing.data()) + 255) & ~std::uintptr_t{255});
        for (std::uint32_t layer = 0; layer < layout.imageLayers; ++layer) {
            const auto value = word(layer);
            for (std::uint64_t offset = 0; offset < layout.layerBytes; offset += sizeof(value)) std::memcpy(base + layout.GuestLayerOffset(layer) + offset, &value, sizeof(value));
        }
        described.baseAddress = reinterpret_cast<std::uint64_t>(base);
        return described;
    };
    std::vector<std::byte> depthMemory;
    const auto depth = wordCube(depthMemory, 22u, [](std::uint32_t layer) {
        const float value = static_cast<float>(layer + 1) / 8.0f;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    });
    auto depthStorage = StorageTexture::Create(context, detiler, depth, 0);
    recorder.Keep(depthStorage);
    Require(Texture::CanDepthCopyFrom(*depthStorage, depth), "a cube with a slice past its last whole cube cannot take a depth copy");
    Texture depthCopy(context, depthStorage, depth, identity, Texture::DepthCopyTag{});
    depthCopy.RefreshDepthCopy();
    constexpr std::array<std::uint32_t, 7> unsignedFloats{960u, 1024u, 1056u, 1088u, 1104u, 1120u, 1136u};
    std::vector<std::byte> repackMemory;
    const auto repacked = wordCube(repackMemory, 43u, [&unsignedFloats](std::uint32_t layer) { return unsignedFloats[layer] << 21u; });
    auto repackStorage = StorageTexture::Create(context, detiler, repacked, 0);
    recorder.Keep(repackStorage);
    Require(Texture::CanRepackCopyFrom(*repackStorage, repacked), "a cube with a slice past its last whole cube cannot take a repacked copy");
    Texture repackCopy(context, repackStorage, repacked, identity, Texture::RepackCopyTag{});
    repackCopy.RefreshDepthCopy();
    for (std::uint32_t layer = 0; layer < geometry.imageLayers; ++layer) {
        const auto expected = 16.0f * (layer + 1) / 255.0f;
        expectRed(program.Red(texture.View(), texture.Layout(), 0, static_cast<float>(layer)), expected, "a cube snapshot with a slice past its last whole cube sampled the wrong slice");
        expectRed(program.Red(storageView.View(), storageView.Layout(), 0, static_cast<float>(layer)), expected, "a cube storage view with a slice past its last whole cube sampled the wrong slice");
        expectRed(program.Red(depthCopy.View(), depthCopy.Layout(), 0, static_cast<float>(layer)), static_cast<float>(layer + 1) / 8.0f, "a cube depth copy with a slice past its last whole cube sampled the wrong slice");
        expectRed(program.Red(repackCopy.View(), repackCopy.Layout(), 0, static_cast<float>(layer)), static_cast<float>(layer + 1), "a cube repacked copy with a slice past its last whole cube sampled the wrong slice");
    }
}

struct SampledViewProbe {
    Context& context;
    PFN_vkGetDeviceProcAddr resolver;
    std::uint32_t views = 0;
    bool checking = false;
    static inline SampledViewProbe* active = nullptr;
    explicit SampledViewProbe(Context& context) : context(context), resolver(context.deviceProc) {
        Require(active == nullptr, "sampled view probe is already active");
        active = this;
        context.deviceProc = resolve;
    }
    ~SampledViewProbe() {
        context.deviceProc = resolver;
        active = nullptr;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL createView(VkDevice device, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* callbacks, VkImageView* created) {
        if (active->checking && info->format == VK_FORMAT_R8G8B8A8_SRGB) {
            const auto* next = static_cast<const VkBaseInStructure*>(info->pNext);
            while (next != nullptr && next->sType != VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO) next = next->pNext;
            if (next == nullptr || reinterpret_cast<const VkImageViewUsageCreateInfo*>(next)->usage != VK_IMAGE_USAGE_SAMPLED_BIT) return VK_ERROR_VALIDATION_FAILED_EXT;
            ++active->views;
        }
        return reinterpret_cast<PFN_vkCreateImageView>(active->resolver(device, "vkCreateImageView"))(device, info, callbacks, created);
    }
    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL resolve(VkDevice device, const char* name) {
        if (std::strcmp(name, "vkCreateImageView") == 0) return reinterpret_cast<PFN_vkVoidFunction>(createView);
        return active->resolver(device, name);
    }
};

void sampledStorageBaseLevelUsageTests(const Device& device, Recorder& recorder) {
    auto context = device.GetContext();
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.width = resource.height = 8;
    resource.depthOrLastArray = 1;
    resource.mipCount = 2;
    resource.lastLevel = 1;
    resource.minLod = context.imageViewMinLod ? 0x100 : 0;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2DArray;
    resource.format = 130;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    const auto geometry = DescribeSurface(resource);
    std::vector<std::byte> memory(static_cast<std::size_t>(geometry.guestBytes) + 256, std::byte{0x80});
    resource.baseAddress = (reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255};
    auto storage = StorageTexture::Create(context, detiler, resource, 0);
    recorder.Keep(storage);
    recorder.Sync();
    SampleProgram program(context, recorder);
    auto sampledContext = context;
    SampledViewProbe probe(sampledContext);
    Texture sampled(sampledContext, storage, resource, VkComponentMapping{});
    probe.checking = true;
    const auto base = sampled.BaseLevelView();
    probe.checking = false;
    expectRed(program.Red(base, sampled.Layout(), 0), 0.21586f, "sampled sRGB base level");
    Require(sampled.BaseLevelView() == base && probe.views == 1, "the lazy base-level view of a sampled storage image must restrict usage to SAMPLED and be reused");
}

void firstLayerViewTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    TextureDetiler detiler(context);
    auto withDetiler = context;
    withDetiler.detiler = &detiler;
    GuestTextureResource resource{};
    resource.width = 64;
    resource.height = 4;
    resource.depthOrLastArray = 2;
    resource.baseArray = 1;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2DArray;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    const auto geometry = DescribeSurface(resource);
    std::vector<std::uint8_t> memory(static_cast<std::size_t>(geometry.guestBytes) + 256);
    auto* surface = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255});
    for (std::uint32_t layer = 0; layer < 3; ++layer) std::memset(surface + geometry.GuestLayerOffset(layer), 0x20 * (layer + 1), static_cast<std::size_t>(geometry.layerBytes));
    resource.baseAddress = reinterpret_cast<std::uint64_t>(surface);
    auto image = StorageTexture::Create(withDetiler, detiler, resource, 0);
    recorder.Keep(image);
    SampleProgram program(context, recorder);
    expectRed(program.Red(image->FirstLayerView(0), VK_IMAGE_LAYOUT_GENERAL, 0.0f), 0x40 / 255.0f, "a first-layer view does not read the BASE_ARRAY layer");
    Require(image->FirstLayerView(0) == image->FirstLayerView(0), "first-layer views are not reused");
    const VkComponentMapping identity{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    Texture snapshotTexture(context, detiler, resource, identity, std::span<const std::byte>(reinterpret_cast<const std::byte*>(surface), static_cast<std::size_t>(geometry.guestBytes)));
    Require(snapshotTexture.FirstLayerView() != VK_NULL_HANDLE, "a sampled 2D array texture has no first-layer view");
    expectRed(program.Red(snapshotTexture.FirstLayerView(), snapshotTexture.Layout(), 0.0f), 0x40 / 255.0f, "a sampled first-layer view does not read the BASE_ARRAY layer");
    Texture storageView(context, image, resource, identity);
    Require(storageView.FirstLayerView() != VK_NULL_HANDLE, "a sampled view of a 2D array storage image has no first-layer view");
    expectRed(program.Red(storageView.FirstLayerView(), storageView.Layout(), 0.0f), 0x40 / 255.0f, "a sampled first-layer view of a storage image does not read the BASE_ARRAY layer");
    auto flat = resource;
    flat.dimension = TextureDimension::k2D;
    flat.depthOrLastArray = 0;
    flat.baseArray = 0;
    Texture flatTexture(context, detiler, flat, identity, std::span<const std::byte>(reinterpret_cast<const std::byte*>(surface), static_cast<std::size_t>(DescribeSurface(flat).guestBytes)));
    Require(flatTexture.FirstLayerView() == VK_NULL_HANDLE, "a 2D texture has a first-layer view");
    expectRed(program.Red(flatTexture.View(), flatTexture.Layout(), 0.0f), 0x20 / 255.0f, "a 2D texture over the surface does not read its first layer");
}

void sliceDirtyTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    constexpr std::uint32_t side = 128;
    constexpr std::size_t sliceBytes = 65536;
    constexpr std::size_t bytes = 4 * sliceBytes;
    constexpr std::uint8_t untouched = 0x11;
    for (const auto tileMode : {TextureTileMode::kLinear, TextureTileMode::kR64KBX}) {
        const std::string what = tileMode == TextureTileMode::kLinear ? "a linear two-slice image" : "a tiled two-slice image";
        void* block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: slice dirty marking not tested\n";
            return;
        }
        auto* memory = static_cast<std::uint8_t*>(block);
        std::memset(memory, untouched, bytes);
        const auto address = reinterpret_cast<std::uint64_t>(block);
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(block, bytes, true, true, true);
        }
        struct Unregister {
            const Context& context;
            void* block;
            std::uint64_t address;
            std::size_t bytes;
            ~Unregister() {
                {
                    GuestAllocations::Mutation mutation;
                    mutation.Remove(block);
                }
                HostImportFor(context, address, bytes);
                ReleaseWatched(block, bytes);
            }
        } unregister{base, block, address, bytes};
        AgcDriver::GuestMemory::CollectWritesUncached(address, bytes);
        TextureDetiler detiler(base);
        auto context = base;
        context.hostImportAlignment = 0;
        context.detiler = &detiler;
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = side;
        resource.height = side;
        resource.depthOrLastArray = 1;
        resource.mipCount = 1;
        resource.tileMode = tileMode;
        resource.dimension = TextureDimension::k2DArray;
        resource.format = 56;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        Require(geometry.imageLayers == 2 && geometry.layerBytes == sliceBytes && geometry.guestBytes == 2 * sliceBytes, what + " does not hold two 64 KiB slices");
        const auto stored = [&](std::uint32_t slice, std::array<std::uint8_t, 4> color) {
            std::vector<std::byte> read(sliceBytes);
            AgcDriver::GuestMemory::Read(address + geometry.GuestLayerOffset(slice), read);
            for (std::size_t at = 0; at < read.size(); ++at) {
                if (std::to_integer<std::uint8_t>(read[at]) != color[at % 4]) return false;
            }
            return true;
        };
        {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
            const VkClearColorValue green{{0.0f, 1.0f, 0.0f, 1.0f}};
            const VkImageSubresourceRange first{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const VkImageSubresourceRange second{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1, 1};
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1, &first);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &green, 1, &second);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkSliceDirty(0);
            for (std::uint32_t slice = 0; slice < 2; ++slice) {
                const auto display = address + geometry.GuestLayerOffset(slice);
                const auto found = StorageTexture::FindPendingSlice(display, sliceBytes);
                Require(found == image, what + ": a display buffer that is slice " + std::to_string(slice) + " does not find the pending image");
                Require(AgcDriver::ResidentPresentSlice(found->Descriptor().baseAddress, found->GuestBytes(), found->ImageLayers(), display, sliceBytes) == slice, what + ": a display buffer that is slice " + std::to_string(slice) + " is not presented from that layer");
            }
            Require(StorageTexture::FindPending(address + geometry.GuestLayerOffset(1), sliceBytes) == nullptr, what + ": the lookup by base address finds a display buffer in slice 1");
            Require(StorageTexture::FindPendingSlice(address + 2 * sliceBytes, sliceBytes) == nullptr && StorageTexture::FindPendingSlice(address + geometry.GuestLayerOffset(1), sliceBytes + 1) == nullptr, what + ": a range past the image's last slice finds the pending image");
            image->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(stored(0, {255, 0, 0, 255}), what + ": the slice a draw wrote was not stored");
            Require(stored(1, {untouched, untouched, untouched, untouched}), what + ": a slice no draw wrote was stored over guest memory");
            static_cast<void>(recorder.Commands());
            recorder.Keep(image);
            image->MarkSliceDirty(1);
            image->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            Require(stored(0, {255, 0, 0, 255}) && stored(1, {0, 255, 0, 255}), what + ": the second slice was not stored once a draw wrote it");
            auto other = StorageTexture::Create(context, detiler, resource, 0);
            static_cast<void>(recorder.Commands());
            recorder.Keep(image);
            recorder.Keep(other);
            image->MarkSliceDirty(1);
            other->MarkSliceDirty(1);
            std::string ambiguous;
            try {
                static_cast<void>(StorageTexture::FindPendingSlice(address + geometry.GuestLayerOffset(1), sliceBytes));
            } catch (const std::runtime_error& error) {
                ambiguous = error.what();
            }
            Require(ambiguous.find("is a slice of the pending array images") != std::string::npos, what + ": a slice of two pending array images did not throw: " + ambiguous);
            image->WriteBack();
            other->WriteBack();
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
        }
        recorder.Sync();
    }
    std::cout << "slice dirty marking: ok\n";
}

void atomicViewTests(const Device& device, Recorder& recorder) {
    const auto& context = device.GetContext();
    TextureDetiler detiler(context);
    auto withDetiler = context;
    withDetiler.detiler = &detiler;
    for (const std::uint32_t format : {20u, 21u, 22u, 56u}) {
        GuestTextureResource resource{};
        resource.width = 64;
        resource.height = 4;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kLinear;
        resource.dimension = TextureDimension::k2D;
        resource.format = format;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const auto geometry = DescribeSurface(resource);
        std::vector<std::uint8_t> memory(static_cast<std::size_t>(geometry.guestBytes) + 256);
        resource.baseAddress = (reinterpret_cast<std::uintptr_t>(memory.data()) + 255) & ~std::uintptr_t{255};
        auto image = StorageTexture::Create(withDetiler, detiler, resource, 0);
        recorder.Keep(image);
        if (format == 56u) {
            bool refused = false;
            try {
                static_cast<void>(image->AtomicView(0, false));
            } catch (const std::exception&) {
                refused = true;
            }
            Require(refused, "an 8_8_8_8 storage image has an atomic view");
            continue;
        }
        const auto atomic = image->AtomicView(0, false);
        Require(atomic != VK_NULL_HANDLE && atomic == image->AtomicView(0, false), "atomic storage views are not reused");
        Require((atomic == image->View(0)) == (format == 20u), "only a 32_UINT storage image serves atomics through its own view");
    }
}

bool vertexTargetRangeTests(const Device& device, Recorder& recorder, bool imported) {
    if (!device.Graphics() || (imported && device.GetContext().hostImportAlignment == 0)) return false;
    constexpr std::size_t unit = 65536;
    constexpr std::size_t bytes = unit * 3;
    void* watched = AllocateWatched(bytes, unit);
    std::vector<std::byte> fallback(watched == nullptr ? bytes + unit : 0);
    auto* block = watched != nullptr ? static_cast<std::byte*>(watched) : reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(fallback.data()) + unit - 1) & ~(unit - 1));
    struct Release {
        void* watched;
        ~Release() { if (watched != nullptr) ReleaseWatched(watched, bytes); }
    } release{watched};
    std::fill_n(block, bytes, std::byte{0});
    const auto address = reinterpret_cast<std::uint64_t>(block);
    auto context = device.GetContext();
    context.recorder = &recorder;
    if (!imported) {
        context.hostImportAlignment = 0;
        context.dmaBufImport = false;
    }
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Cleanup {
        const Context& context;
        Recorder& recorder;
        void* block;
        ~Cleanup() {
            recorder.Sync();
            WaitForOptimizedPipelines(context.device);
            SharedResourceCache().Clear();
            ClearCachedTextures(context.device);
            recorder.Sync();
            ClearCachedPipelines(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } cleanup{context, recorder, block};
    if (imported && HostImportFor(context, address, bytes) == nullptr) return false;
    const auto shader = [](auto& words, std::uint64_t variant) {
        ShaderRecompiler::RecompileResult program;
        program.spirv = std::vector<std::uint32_t>(std::begin(words), std::end(words));
        program.variantId = variant;
        return program;
    };
    auto producerVertex = shader(COVER_VERT_SPV, 101);
    auto producerPixel = shader(VERTEXRANGEPRODUCER_FRAG_SPV, 102);
    auto vertex = shader(VERTEXRANGE_VERT_SPV, 103);
    auto pixel = shader(COVER_FRAG_SPV, 104);
    const std::array producerShaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &producerVertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &producerPixel, 0}};
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &pixel, 0}};
    const auto stateFor = [](std::uint64_t base, VkFormat format, std::uint32_t elementBytes) {
        State state{};
        state.stages.path = ShaderPath::Vertex;
        state.color = {base, {64, 64}, format, unit, 0xe4u, ColorTileMode::RenderTarget, elementBytes};
        state.colors = {state.color};
        state.hasColorTarget = true;
        state.renderExtent = {64, 64};
        state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        state.viewport = {0, 64, 64, -64, 0, 1};
        state.scissor = {{0, 0}, {64, 64}};
        state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        state.blend.colorWriteMask = 15;
        state.blends = {state.blend};
        return state;
    };
    const auto producer = stateFor(address + unit, VK_FORMAT_R32G32B32A32_SFLOAT, 16);
    const auto consumer = stateFor(address, VK_FORMAT_R8G8B8A8_UNORM, 4);
    const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 1, 0, false};
    const auto produce = [&] {
        Draw(context, producer, triangle, producerShaders);
        WaitForOptimizedPipelines(context.device);
        const std::array range{std::pair{address + unit, address + 2 * unit}};
        Require(StorageTexture::AnyPendingOverlaps(range), "the color producer did not leave GPU results pending");
    };
    const auto attributeFor = [](std::uint64_t base, std::uint32_t stride, std::uint32_t fetchIndex, std::uint32_t outOfBoundsSelect = 0) {
        return ShaderRecompiler::VertexAttribute{0, 4, {{static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(base >> 32u) | (stride << 16u), static_cast<std::uint32_t>(bytes / 16), (77u << 12u) | (outOfBoundsSelect << 28u)}}, fetchIndex};
    };
    const auto check = [&](std::uint32_t edge) {
        std::vector<std::byte> texels(64 * 64 * 4);
        ReadColorTarget(consumer.color, texels);
        for (std::size_t y = 0; y < 64; ++y) {
            for (std::size_t x = 0; x < 64; ++x) {
                const auto expected = x < edge ? std::byte{0} : std::byte{255};
                for (std::size_t channel = 0; channel < 4; ++channel) Require(texels[(y * 64 + x) * 4 + channel] == expected, "the draw did not consume the GPU-produced vertex offset");
            }
        }
    };
    const auto reject = [&](auto action, const char* expected) {
        try { action(); }
        catch (const std::runtime_error& error) {
            Require(std::string(error.what()).find(expected) != std::string::npos, error.what());
            return;
        }
        throw std::runtime_error("an actual render-target alias was accepted");
    };
    for (const std::uint32_t fetchIndex : {0u, 1u}) {
        producerPixel = shader(VERTEXRANGEPRODUCER_FRAG_SPV, 102);
        vertex.vertexAttributes = {attributeFor(address, 16, fetchIndex)};
        auto draw = triangle;
        if (fetchIndex == 0) draw.firstVertex = unit / 16;
        else draw.firstInstance = unit / 16;
        produce();
        std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
        Draw(context, consumer, draw, shaders, {}, &recipe);
        WaitForOptimizedPipelines(context.device);
        const auto read = recorder.DescribePendingRead(address + unit, 16);
        Require(read.has_value() == imported && (!read || read->kind == Recorder::ReadKind::DrawInput || read->kind == Recorder::ReadKind::StorageUpload), "the vertex regression used the wrong copied/imported path");
        check(16);
        Require(recipe != nullptr, "the range regression did not create a draw recipe");
        std::vector<std::byte> cleared(unit);
        AgcDriver::GuestMemory::Write(address, cleared);
        for (std::size_t offset = 5; offset < producerPixel.spirv.size();) {
            const auto count = producerPixel.spirv[offset] >> 16u;
            Require(count != 0, "invalid producer SPIR-V");
            if ((producerPixel.spirv[offset] & 0xffffu) == 43u && count == 4 && producerPixel.spirv[offset + 3] == 0x3f000000u) producerPixel.spirv[offset + 3] = 0x3f400000u;
            offset += count;
        }
        producerPixel.variantId = 105;
        produce();
        const auto hit = DrawWithRecipe(context, consumer, draw, shaders, {}, *recipe);
        WaitForOptimizedPipelines(context.device);
        Require(hit.recorded, "the vertex range regression did not reuse its draw recipe");
        check(24);
        vertex.vertexAttributes = {attributeFor(address + unit, 16, fetchIndex)};
        Draw(context, consumer, triangle, shaders);
        WaitForOptimizedPipelines(context.device);
        check(24);
        vertex.vertexAttributes = {attributeFor(address, 16, fetchIndex)};
        auto indirect = draw;
        indirect.indirect.emplace();
        reject([&] { Draw(context, consumer, indirect, shaders); }, "vertex buffer aliases the render target");
        if (fetchIndex == 0) {
            const std::array<std::uint16_t, 3> indices{0, 1, 2};
            AgcDriver::GuestMemory::Write(address + 2 * unit, std::as_bytes(std::span(indices)));
            auto indexed = draw;
            indexed.indexed = true;
            indexed.indexSize = 2;
            indexed.indexAddress = address + 2 * unit;
            produce();
            Draw(context, consumer, indexed, shaders);
            WaitForOptimizedPipelines(context.device);
            check(24);
            indexed.indexAddress = address;
            reject([&] { Draw(context, consumer, indexed, shaders); }, "index buffer aliases the render target");
        }
        reject([&] { Draw(context, consumer, triangle, shaders); }, "vertex buffer aliases the render target");
        vertex.vertexAttributes = {attributeFor(address, 0, fetchIndex, 3)};
        reject([&] { Draw(context, consumer, draw, shaders); }, "vertex buffer aliases the render target");
    }
    vertex.vertexAttributes = {attributeFor(address, 0, 0)};
    Draw(context, consumer, triangle, shaders);
    WaitForOptimizedPipelines(context.device);
    std::cout << "Vertex range and pending color producer tests passed (" << (imported ? "imported" : "copied") << ")\n";
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_HELPER_FETCH_PLAN, PARALLEL-COMMIT-PLAN B2; ctest agc_helper_fetch_plan,
// _verify and _vram1): a ticket's FetchPlan part, made as a producer makes it (a context without a
// recorder, inside a PrepareScope) and handed to Draw as the committer's ticket (the copied draw
// input path; the counts need APS5_DEBUG_Q0_PREP_STATS):
// - a plan whose predicted maxIndex is the draw's own is accepted, with the first vertex added and
//   with a zero buffer for an out-of-range attribute (=2 also plans inline and compares);
// - an index store after the plan raises the highest index: the committer refuses the plan on the
//   mismatch and plans inline;
// - an index past the zero-fetch clamp's slack gives no part, and the committer clamps as before;
// - a vertex range aliasing the color target gives no part, and the committer raises the alias at
//   its own place, with the message a draw without a ticket raises;
// - a plan built on another thread with the kernel copy a prep helper uses after the push, with a
//   zero buffer, is accepted (=2 compares it); a ticket that thread drops (a failed
//   compare-exchange) hands its zero buffer to the ticket release list instead of destroying it;
// - the budget, asked before the part and before every 64 KiB chunk after the first, leaves the
//   part out (cut) when it holds before the part or before the second chunk;
// - an index range past HelperFetchPlanScanBytes (128 KiB in the ctest environments) is not
//   scanned (no part, the budget never asked);
// - under the VRAM draw inputs, queue 0's receipt plan walks the plan's own copies instead of the
//   descriptors' whole extents.
bool helperFetchPlanTests(const Device& device, Recorder& recorder) {
    if (!device.Graphics()) return false;
    if (HelperFetchPlan() == 0 || AgcDriver::GuestMemory::Q0PrepStats() == 0) {
        std::cout << "skipped, the fetch plan tests need APS5_EXPERIMENT_Q0_TICKET, APS5_EXPERIMENT_HELPER_FETCH_PLAN and APS5_DEBUG_Q0_PREP_STATS\n";
        return false;
    }
    constexpr std::size_t unit = 65536;
    // The color target, the vertex records, then four units of indices (the scan tests' two chunks
    // and the scan cap's range).
    constexpr std::size_t bytes = unit * 6;
    void* watched = AllocateWatched(bytes, unit);
    std::vector<std::byte> fallback(watched == nullptr ? bytes + unit : 0);
    auto* block = watched != nullptr ? static_cast<std::byte*>(watched) : reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(fallback.data()) + unit - 1) & ~(unit - 1));
    struct Release {
        void* watched;
        ~Release() { if (watched != nullptr) ReleaseWatched(watched, bytes); }
    } release{watched};
    std::fill_n(block, bytes, std::byte{0});
    const auto address = reinterpret_cast<std::uint64_t>(block);
    auto context = device.GetContext();
    context.recorder = &recorder;
    context.hostImportAlignment = 0;
    context.dmaBufImport = false;
    static_cast<void>(GetBufferPool(context));
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Cleanup {
        const Context& context;
        Recorder& recorder;
        void* block;
        ~Cleanup() {
            recorder.Sync();
            DrainTicketRelease(context);
            WaitForOptimizedPipelines(context.device);
            SharedResourceCache().Clear();
            ClearCachedTextures(context.device);
            recorder.Sync();
            ClearCachedPipelines(context.device);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, reinterpret_cast<std::uint64_t>(block), bytes);
        }
    } cleanup{context, recorder, block};
    const auto shader = [](auto& words, std::uint64_t variant) {
        ShaderRecompiler::RecompileResult program;
        program.spirv = std::vector<std::uint32_t>(std::begin(words), std::end(words));
        program.variantId = variant;
        return program;
    };
    auto vertex = shader(VERTEXRANGE_VERT_SPV, 201);
    auto pixel = shader(COVER_FRAG_SPV, 202);
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &pixel, 0}};
    State consumer{};
    consumer.stages.path = ShaderPath::Vertex;
    consumer.color = {address, {64, 64}, VK_FORMAT_R8G8B8A8_UNORM, unit, 0xe4u, ColorTileMode::RenderTarget, 4};
    consumer.colors = {consumer.color};
    consumer.hasColorTarget = true;
    consumer.renderExtent = {64, 64};
    consumer.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    consumer.viewport = {0, 64, 64, -64, 0, 1};
    consumer.scissor = {{0, 0}, {64, 64}};
    consumer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    consumer.blend.colorWriteMask = 15;
    consumer.blends = {consumer.blend};
    // Sixteen 16-byte records past the color target (the clamp's first zero-reading vertex is 16).
    const auto vertices = address + unit;
    const auto indices = address + 2 * unit;
    const auto attributeFor = [](std::uint64_t base, std::uint32_t stride, std::uint32_t records) {
        return ShaderRecompiler::VertexAttribute{0, 4, {{static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(base >> 32u) | (stride << 16u), records, 77u << 12u}}, 0};
    };
    vertex.vertexAttributes = {attributeFor(vertices, 16, 16)};
    auto producerContext = context;
    producerContext.recorder = nullptr;
    producerContext.copiedWriters = nullptr;
    const auto buildWith = [&](const AgcDriver::Pm4::DrawParameters& draw, bool (*stop)(const void*), const void* budget, bool unpinned) {
        auto ticket = std::make_unique<DrawTicket>();
        ticket->device = context.device;
        ticket->pool = context.bufferPool;
        const AgcDriver::GuestMemory::PrepareScope scope;
        FillDrawTicketFetchPlan(*ticket, producerContext, consumer, draw, shaders, stop, budget, unpinned);
        return ticket;
    };
    const auto build = [&](const AgcDriver::Pm4::DrawParameters& draw) { return buildWith(draw, nullptr, nullptr, false); };
    // A budget that counts how often it is asked and holds at its `holdAt`th ask (0: never).
    struct Asks {
        mutable int count = 0;
        int holdAt = 0;
    };
    bool (*const asked)(const void*) = [](const void* budget) {
        const auto& asks = *static_cast<const Asks*>(budget);
        return ++asks.count == asks.holdAt;
    };
    // The committer's side, as VulkanDevice::Draw runs it; the ticket's references are settled
    // whether the draw records or throws.
    const auto commit = [&](DrawTicket& ticket, const AgcDriver::Pm4::DrawParameters& draw) {
        DrainTicketRelease(context);
        try {
            Draw(context, consumer, draw, shaders, {}, nullptr, nullptr, &ticket);
        } catch (...) {
            ReleaseTicketReferences(ticket, context);
            throw;
        }
        ReleaseTicketReferences(ticket, context);
        WaitForOptimizedPipelines(context.device);
    };
    const auto events = [] { return TakeTicketCounts().fetchEvents; };
    using Events = decltype(TakeTicketCounts().fetchEvents);
    const auto count = [](const Events& counts, FetchPlanEvent event) { return counts[static_cast<std::size_t>(event)]; };
    const auto writeIndices = [&](const auto& values) { AgcDriver::GuestMemory::Write(indices, std::as_bytes(std::span(values))); };
    static_cast<void>(events());
    {
        // Accepted: the plan for the scanned highest index, with the first vertex added.
        writeIndices(std::array<std::uint16_t, 3>{0, 1, 2});
        AgcDriver::Pm4::DrawParameters indexed{indices, 3, 2, 1, 0, true};
        auto ticket = build(indexed);
        Require(ticket->Has(DrawTicket::FetchPlan) && ticket->predictedMax == 2u && ticket->zeroBuffers.size() == 1 && ticket->zeroBuffers[0] == nullptr && ticket->fetchPlan.fetches.size() == 1 && ticket->fetchPlan.fetches[0].begin == vertices && ticket->fetchPlan.fetches[0].end == vertices + 48, "an indexed draw's fetch plan is not the one for its highest index");
        commit(*ticket, indexed);
        indexed.firstVertex = 4;
        ticket = build(indexed);
        Require(ticket->Has(DrawTicket::FetchPlan) && ticket->predictedMax == 6u, "an indexed draw's predicted maxIndex lacks its first vertex");
        commit(*ticket, indexed);
        const auto counts = events();
        Require(count(counts, FetchPlanEvent::Built) == 2 && count(counts, FetchPlanEvent::Offered) == 2 && count(counts, FetchPlanEvent::Accepted) == 2 && count(counts, FetchPlanEvent::Mismatch) == 0 && count(counts, FetchPlanEvent::NoPart) == 0, "a fetch plan whose prediction holds was not accepted");
    }
    {
        // A zero buffer for an out-of-range attribute (stride 0, OOB_SELECT 0), on an auto draw.
        vertex.vertexAttributes = {attributeFor(vertices, 0, 16)};
        const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 1, 0, false};
        auto ticket = build(triangle);
        Require(ticket->Has(DrawTicket::FetchPlan) && ticket->predictedMax == 2u && ticket->fetchPlan.zero.size() == 1 && ticket->fetchPlan.zero[0] == 1 && ticket->zeroBuffers.size() == 1 && ticket->zeroBuffers[0] != nullptr, "an out-of-range attribute's fetch plan has no zero buffer");
        const auto zero = ticket->zeroBuffers[0]->Bytes();
        Require(zero.size() == ticket->fetchPlan.zeroBytes[0] && std::all_of(zero.begin(), zero.end(), [](std::byte value) { return value == std::byte{0}; }), "the fetch plan's zero buffer is not zeros of the attribute's size");
        commit(*ticket, triangle);
        const auto counts = events();
        Require(count(counts, FetchPlanEvent::Accepted) == 1, "an auto draw's fetch plan was not accepted");
        vertex.vertexAttributes = {attributeFor(vertices, 16, 16)};
    }
    {
        // Mismatch: an index store after the plan raises the highest index; the committer's own scan
        // differs from the prediction and it plans inline.
        writeIndices(std::array<std::uint16_t, 3>{0, 1, 2});
        const AgcDriver::Pm4::DrawParameters indexed{indices, 3, 2, 1, 0, true};
        auto ticket = build(indexed);
        Require(ticket->Has(DrawTicket::FetchPlan) && ticket->predictedMax == 2u, "the plan before the index store was not made");
        writeIndices(std::array<std::uint16_t, 3>{0, 1, 3});
        commit(*ticket, indexed);
        const auto counts = events();
        Require(count(counts, FetchPlanEvent::Offered) == 1 && count(counts, FetchPlanEvent::Mismatch) == 1 && count(counts, FetchPlanEvent::Accepted) == 0, "a fetch plan for a stale highest index was not refused");
    }
    {
        // The zero-fetch clamp: an index 0x20000 past the 16 records gives no part; the committer
        // clamps the indices as it does without a ticket.
        writeIndices(std::array<std::uint32_t, 3>{0, 1, 0x20000});
        const AgcDriver::Pm4::DrawParameters indexed{indices, 3, 4, 1, 0, true};
        auto ticket = build(indexed);
        auto counts = events();
        Require(!ticket->Has(DrawTicket::FetchPlan) && ticket->zeroBuffers.empty() && count(counts, FetchPlanEvent::Clamp) == 1 && count(counts, FetchPlanEvent::Built) == 0, "a draw the zero-fetch clamp rewrites got a fetch plan");
        commit(*ticket, indexed);
        counts = events();
        Require(count(counts, FetchPlanEvent::NoPart) == 1 && count(counts, FetchPlanEvent::Offered) == 0, "the committer did not plan a clamped draw itself");
    }
    {
        // The alias: a vertex range over the color target throws in the producer's plan (no part)
        // and again at the committer's own planVertexFetches, as without a ticket.
        vertex.vertexAttributes = {attributeFor(address, 16, 16)};
        const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 1, 0, false};
        const auto failure = [&](DrawTicket* ticket) {
            try {
                if (ticket != nullptr) commit(*ticket, triangle);
                else Draw(context, consumer, triangle, shaders);
            } catch (const std::runtime_error& error) {
                return std::string(error.what());
            }
            return std::string();
        };
        const auto today = failure(nullptr);
        Require(today.find("vertex buffer aliases the render target") != std::string::npos, "the alias test's draw did not fail on the alias without a ticket");
        static_cast<void>(events());
        auto ticket = build(triangle);
        auto counts = events();
        Require(!ticket->Has(DrawTicket::FetchPlan) && count(counts, FetchPlanEvent::Threw) == 1, "a vertex range over the color target got a fetch plan");
        Require(failure(ticket.get()) == today, "the committer did not raise the alias at its own place");
        counts = events();
        Require(count(counts, FetchPlanEvent::NoPart) == 1, "the committer did not find the alias draw's ticket without a part");
        vertex.vertexAttributes = {attributeFor(vertices, 16, 16)};
    }
    if (AgcDriver::GuestMemory::SpeculativeReadsSupported()) {
        // A producer on another thread, as a prep helper after the push (the scan copies through the
        // kernel), with a zero buffer: the committer on this thread accepts its plan (=2 also plans
        // inline and compares the plan and the zero buffer).
        writeIndices(std::array<std::uint16_t, 3>{0, 1, 2});
        vertex.vertexAttributes = {attributeFor(vertices, 0, 16)};
        const AgcDriver::Pm4::DrawParameters indexed{indices, 3, 2, 1, 0, true};
        std::unique_ptr<DrawTicket> ticket;
        std::thread([&] { ticket = buildWith(indexed, nullptr, nullptr, true); }).join();
        Require(ticket->Has(DrawTicket::FetchPlan) && ticket->predictedMax == 2u && ticket->zeroBuffers.size() == 1 && ticket->zeroBuffers[0] != nullptr, "a fetch plan built on another thread through the kernel copy was not made");
        commit(*ticket, indexed);
        const auto counts = events();
        Require(count(counts, FetchPlanEvent::Built) == 1 && count(counts, FetchPlanEvent::Accepted) == 1, "a fetch plan built on another thread was not accepted");
        // A ticket its commit never takes (a helper whose compare-exchange failed destroys it): its
        // zero buffer goes to the ticket release list, not destroyed on the producing thread.
        std::weak_ptr<Buffer> zero;
        bool made = false;
        std::thread([&] {
            auto lost = buildWith(indexed, nullptr, nullptr, true);
            made = lost->Has(DrawTicket::FetchPlan) && lost->zeroBuffers.size() == 1 && lost->zeroBuffers[0] != nullptr;
            if (made) zero = lost->zeroBuffers[0];
        }).join();
        Require(made && !zero.expired(), "a dropped ticket's zero vertex buffer was destroyed on its producing thread");
        DrainTicketRelease(context);
        static_cast<void>(events());
        vertex.vertexAttributes = {attributeFor(vertices, 16, 16)};
    } else {
        std::cout << "the kernel-copy fetch plan checks need speculative guest reads, skipped\n";
    }
    {
        // The budget: holding before the part leaves it out; holding before the second 64 KiB chunk
        // cuts the scan (no part, no zero buffer); not holding, it is asked before the part and
        // before the second chunk, and the scan finds the second chunk's highest index.
        writeIndices(std::array<std::uint16_t, 3>{0, 1, 2});
        const AgcDriver::Pm4::DrawParameters indexed{indices, 3, 2, 1, 0, true};
        const Asks before{0, 1};
        auto ticket = buildWith(indexed, asked, &before, false);
        auto counts = events();
        Require(!ticket->Has(DrawTicket::FetchPlan) && before.count == 1 && count(counts, FetchPlanEvent::Cut) == 1, "a budget holding before the fetch plan did not leave the part out as cut");
        constexpr std::uint32_t twoChunks = 40000;
        std::vector<std::uint16_t> many(twoChunks, 0);
        many.back() = 5;
        writeIndices(many);
        const AgcDriver::Pm4::DrawParameters scanned{indices, twoChunks, 2, 1, 0, true};
        const Asks second{0, 2};
        ticket = buildWith(scanned, asked, &second, false);
        counts = events();
        Require(!ticket->Has(DrawTicket::FetchPlan) && ticket->zeroBuffers.empty() && second.count == 2 && count(counts, FetchPlanEvent::Cut) == 1, "a budget holding before the second chunk did not cut the index scan");
        const Asks never{0, 0};
        ticket = buildWith(scanned, asked, &never, false);
        counts = events();
        Require(ticket->Has(DrawTicket::FetchPlan) && ticket->predictedMax == 5u && never.count == 2 && count(counts, FetchPlanEvent::Built) == 1, "a scan of two chunks did not find the second chunk's highest index, or did not ask the budget before each chunk");
    }
    if (const auto cap = HelperFetchPlanScanBytes(); cap != 0 && cap + 4 <= 4 * unit) {
        // Past the scan cap: no part, and neither the budget nor the guest bytes are asked.
        const AgcDriver::Pm4::DrawParameters large{indices, static_cast<std::uint32_t>(cap / 4 + 1), 4, 1, 0, true};
        const Asks never{0, 0};
        auto ticket = buildWith(large, asked, &never, false);
        const auto counts = events();
        Require(!ticket->Has(DrawTicket::FetchPlan) && never.count == 0 && count(counts, FetchPlanEvent::Large) == 1, "an index range past the scan cap got a fetch plan or was scanned");
    } else {
        std::cout << "the scan cap check needs APS5_EXPERIMENT_HELPER_FETCH_PLAN_SCAN_KIB between 1 and 255, skipped\n";
    }
    if (DrawInputsVram() != 0) {
        // Queue 0's receipt plan with the part walks the plan's own copies (exact for the committer's
        // copies while its maxIndex equals the prediction), without it each descriptor's extent.
        writeIndices(std::array<std::uint16_t, 3>{0, 1, 2});
        const AgcDriver::Pm4::DrawParameters indexed{indices, 3, 2, 1, 0, true};
        auto ticket = build(indexed);
        Require(ticket->Has(DrawTicket::FetchPlan) && !ticket->fetchPlan.copies.copies.empty(), "the receipt plan's fetch plan was not made");
        std::vector<std::pair<std::uint64_t, std::uint64_t>> planned;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> extents;
        PlanDrawTicketWalks(*ticket, consumer, indexed, shaders, planned);
        const DrawTicket bare;
        PlanDrawTicketWalks(bare, consumer, indexed, shaders, extents);
        for (const auto& [begin, end] : ticket->fetchPlan.copies.copies) {
            Require(std::find(planned.begin(), planned.end(), std::pair<std::uint64_t, std::uint64_t>{begin, end}) != planned.end(), "queue 0's receipt plan does not walk a copy of the ticket's fetch plan");
        }
        Require(planned != extents, "queue 0's receipt plan with a fetch plan still walks the descriptors' whole extents");
        static_cast<void>(events());
    }
    std::cout << "Helper fetch plan tests passed (APS5_EXPERIMENT_HELPER_FETCH_PLAN=" << HelperFetchPlan() << ", APS5_EXPERIMENT_DRAW_INPUTS_VRAM=" << DrawInputsVram() << ")\n";
    return true;
}

void pipelineCacheTests(const Device& device, bool benchmark) {
    std::thread establishThreadedProcess([] {});
    establishThreadedProcess.join();
    std::lock_guard gpu(GpuMutex());
    const auto& context = device.GetContext();
    struct Cleanup {
        VkDevice device;
        ~Cleanup() { ClearCachedPipelines(device); }
    } cleanup{context.device};
    ShaderRecompiler::RecompileResult vertex, fragment;
    vertex.variantId = 1;
    fragment.variantId = 2;
    vertex.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_vert_SPV), std::end(TRIANGLE_vert_SPV));
    fragment.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_frag_SPV), std::end(TRIANGLE_frag_SPV));
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    State state{};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.hasColorTarget = true;
    state.color.format = VK_FORMAT_R8G8B8A8_UNORM;
    state.colors.push_back(state.color);
    state.blend.colorWriteMask = 15;
    state.blends.push_back(state.blend);
    ShaderResources resources(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), 0, 0);
    const auto lookup = [&](const VertexInputLayout& input) {
        return CachedPipeline(context, state, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL);
    };
    VertexInputLayout input;
    const auto first = lookup(input);
    Require(lookup(input) == first, "pipeline cache did not retain an identical key");
    state.cullMode = VK_CULL_MODE_BACK_BIT;
    Require(lookup(input) != first, "pipeline cache ignored changed raster state");
    state.cullMode = 0;
    const auto largest = std::min({32u, context.limits.maxVertexInputBindings, context.limits.maxVertexInputAttributes});
    for (const auto attributes : {0u, 8u, largest}) {
        input.bindings.clear();
        input.attributes.clear();
        for (std::uint32_t i = 0; i < attributes; ++i) {
            input.bindings.push_back({i, 16, VK_VERTEX_INPUT_RATE_VERTEX});
            input.attributes.push_back({i, i, VK_FORMAT_R32G32B32A32_SFLOAT, 0});
        }
        const auto expected = lookup(input);
        if (attributes != 0) Require(expected != first, "pipeline cache ignored changed vertex input");
        Require(lookup(input) == expected, "pipeline cache lost the larger key");
        if (!benchmark) continue;
        std::array<double, 9> times;
        for (unsigned pass = 0; pass <= times.size(); ++pass) {
            const auto started = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < 65536; ++i) Require(lookup(input) == expected, "pipeline cache benchmark missed a warmed key");
            if (pass != 0) times[pass - 1] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count() / 65536;
        }
        std::ranges::sort(times);
        std::cout << "Pipeline cache " << attributes << " attributes: median " << times[4] << " us/lookup, p95 pass " << times.back() << " us\n";
    }
    input = {};
    Require(lookup(input) == first, "pipeline cache retained bytes from a larger key");
    vertex.variantId = 0;
    Require(lookup(input) == first, "pipeline cache keyed on the variant id instead of the module");
    vertex.spirv[2] ^= 1u;
    const auto other = lookup(input);
    Require(other != first && lookup(input) == other, "pipeline cache ignored another module");
    vertex.spirv[2] ^= 1u;
    vertex.variantId = 1;
    Require(lookup(input) == first, "another module disturbed the retained pipeline");
    ClearCachedPipelines(context.device);
    Require(lookup(input) != first, "clearing the pipeline cache retained its old entry");
    std::cout << "Pipeline cache key and retained lifetime tests passed\n";
}

int pipelineCacheFileTests(const Device& device) {
    const auto& context = device.GetContext();
    const auto directory = std::filesystem::temp_directory_path() / ("aps5-pipeline-cache-file-test-" + std::to_string(std::random_device{}()));
    struct RemoveDirectory {
        std::filesystem::path directory;
        ~RemoveDirectory() {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
        }
    } removeDirectory{directory};
#ifdef _WIN32
    _putenv_s("ANYPS5_NO_SHADER_CACHE", "0");
    _putenv_s("ANYPS5_SHADER_CACHE_DIR", directory.string().c_str());
#else
    setenv("ANYPS5_NO_SHADER_CACHE", "0", 1);
    setenv("ANYPS5_SHADER_CACHE_DIR", directory.string().c_str(), 1);
#endif
    const auto properties = device.Properties();
    struct Objects {
        const Context& context;
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::array<VkShaderModule, 2> modules{};
        ~Objects() {
            for (const auto module : modules) {
                if (module != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
            }
            if (layout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
            if (setLayout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
        }
    } objects{context};
    const std::array bindings{VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo setLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    setLayoutInfo.pBindings = bindings.data();
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &setLayoutInfo, nullptr, &objects.setLayout), "vkCreateDescriptorSetLayout");
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 2};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &objects.setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &push;
    Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &objects.layout), "vkCreatePipelineLayout");
    const std::array<std::span<const std::uint32_t>, 2> codes{std::span<const std::uint32_t>(SAMPLE_LOD_SPV), std::span<const std::uint32_t>(SAMPLE_ARRAY_SPV)};
    for (std::size_t index = 0; index < codes.size(); ++index) {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = codes[index].size_bytes();
        moduleInfo.pCode = codes[index].data();
        Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &objects.modules[index]), "vkCreateShaderModule");
    }
    const auto build = [&](VkPipelineCache cache, std::size_t index) {
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, objects.modules[index], "main", nullptr};
        info.layout = objects.layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, cache, 1, &info, nullptr, &pipeline), "vkCreateComputePipelines");
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
    };
    const auto bytes = [&](VkPipelineCache cache) {
        std::size_t size = 0;
        Check(context.Function<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData")(context.device, cache, &size, nullptr), "vkGetPipelineCacheData");
        return size;
    };
    const auto footprint = [&](std::size_t index) {
        VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        VkPipelineCache fresh = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreatePipelineCache>("vkCreatePipelineCache")(context.device, &info, nullptr, &fresh), "vkCreatePipelineCache");
        const auto before = bytes(fresh);
        build(fresh, index);
        const auto after = bytes(fresh);
        context.Function<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache")(context.device, fresh, nullptr);
        return after > before ? after - before : 0;
    };
    const std::array<std::size_t, 2> footprints{footprint(0), footprint(1)};
    const auto holds = [&](VkPipelineCache cache, std::size_t index) {
        const auto before = bytes(cache);
        build(cache, index);
        const auto after = bytes(cache);
        return after < before + footprints[index] / 2;
    };
    const auto cacheFile = [&] {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) Require(entry.is_directory(), "a pipeline cache file lives in the shader cache root");
        char version[17];
        std::snprintf(version, sizeof(version), "%016llx", static_cast<unsigned long long>(ShaderRecompiler::ShaderDiskCache::Generated::SourceVersion));
        const auto versionDirectory = directory / version;
        Require(std::filesystem::is_directory(versionDirectory), "the current shader cache version has no directory");
        std::vector<std::filesystem::path> found;
        for (const auto& entry : std::filesystem::directory_iterator(versionDirectory)) found.push_back(entry.path());
        Require(found.size() == 1 && found.front().filename().string().starts_with("vk-pipelines-"), "the current shader cache version's directory does not hold exactly one pipeline cache file");
        return found.front();
    };
    const auto contents = [](const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    };
    if (footprints[0] == 0 || footprints[1] == 0) {
        std::cout << "skipped, the device does not keep pipelines in a pipeline cache\n";
        return 77;
    }
    for (const bool firstSavesFirst : {true, false}) {
        std::filesystem::remove_all(directory);
        auto first = std::make_unique<PipelineCache>(context, properties);
        auto second = std::make_unique<PipelineCache>(context, properties);
        build(first->Handle(), 0);
        build(second->Handle(), 1);
        if (firstSavesFirst) {
            first.reset();
            second.reset();
        } else {
            second.reset();
            first.reset();
        }
        PipelineCache third(context, properties);
        Require(holds(third.Handle(), 0), firstSavesFirst ? "the cache file lost the pipeline of the object that saved first" : "the cache file lost the pipeline of the object that saved last");
        Require(holds(third.Handle(), 1), firstSavesFirst ? "the cache file lost the pipeline of the object that saved last" : "the cache file lost the pipeline of the object that saved first");
    }
    const auto path = cacheFile();
    const auto stored = contents(path);
    {
        PipelineCache idle(context, properties);
    }
    Require(contents(path) == stored, "an object that added no pipelines rewrote the cache file");
    {
        PipelineCache unseeded(context, properties, false);
        Require(!holds(unseeded.Handle(), 0), "a cache created without the file already held a pipeline of the file");
    }
    {
        PipelineCache reloaded(context, properties);
        Require(holds(reloaded.Handle(), 0) && holds(reloaded.Handle(), 1), "an unseeded cache dropped pipelines from the cache file when it saved");
    }
    std::filesystem::remove_all(directory);
    {
        PipelineCache writer(context, properties);
        build(writer.Handle(), 0);
    }
    {
        PipelineCache unseeded(context, properties, false);
        unseeded.SaveNow();
        build(unseeded.Handle(), 1);
        unseeded.SaveNow();
    }
    {
        PipelineCache reloaded(context, properties);
        Require(holds(reloaded.Handle(), 0) && holds(reloaded.Handle(), 1), "an unseeded cache that saved twice over an unchanged file lost pipelines of the file");
    }
    std::filesystem::remove_all(directory);
    {
        PipelineCache writer(context, properties);
        build(writer.Handle(), 0);
    }
    {
        PipelineCache extender(context, properties);
        Require(holds(extender.Handle(), 0), "the cache file lost a pipeline");
        build(extender.Handle(), 1);
    }
    {
        PipelineCache reloaded(context, properties);
        Require(holds(reloaded.Handle(), 0) && holds(reloaded.Handle(), 1), "a cache that saved over an unchanged file lost pipelines");
    }
    {
        std::ofstream corrupt(path, std::ios::binary | std::ios::trunc);
        corrupt << "not a pipeline cache";
    }
    {
        PipelineCache replacing(context, properties);
        build(replacing.Handle(), 1);
    }
    {
        PipelineCache reloaded(context, properties);
        Require(holds(reloaded.Handle(), 1), "an invalid cache file was not replaced by the saving object's pipelines");
    }
    {
        PipelineCache busy(context, properties);
        build(busy.Handle(), 0);
        std::promise<void> held;
        std::promise<void> release;
        std::thread creation([&] {
            auto use = busy.Synchronization().LockForPipelineUse();
            held.set_value();
            release.get_future().wait();
        });
        held.get_future().wait();
        auto saving = std::async(std::launch::async, [&] { busy.SaveUnlessBusy(); });
        const bool saveReturned = saving.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        bool otherCreationProceeded = false;
        if (saveReturned) {
            auto other = std::async(std::launch::async, [&] { auto use = busy.Synchronization().LockForPipelineUse(); });
            otherCreationProceeded = other.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
            release.set_value();
            other.wait();
        } else {
            release.set_value();
        }
        creation.join();
        saving.wait();
        Require(saveReturned, "a periodic pipeline cache save waited for a pipeline creation to finish");
        Require(otherCreationProceeded, "a pipeline creation waited behind a periodic pipeline cache save");
    }
    std::cout << "Pipeline cache file merge tests passed\n";
    return 0;
}

void descriptorPoolTests(const Device& device) {
    DescriptorCache cache(device.GetContext());
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    const std::array<std::uint32_t, 4> key{binding.binding, binding.descriptorType, binding.descriptorCount, binding.stageFlags};
    const auto layout = cache.Layout(key, std::span(&binding, 1));
    const VkDescriptorPoolSize size{binding.descriptorType, binding.descriptorCount};
    const auto held = cache.Allocate(layout, std::span(&size, 1));
    std::uint64_t initialPools = 0;
    for (unsigned round = 0; round < 3; ++round) {
        std::vector<DescriptorCache::SetAllocation> sets;
        for (unsigned i = 0; i < 1100; ++i) {
            const auto allocation = cache.Allocate(layout, std::span(&size, 1));
            Require(allocation.set != VK_NULL_HANDLE, "descriptor pool rollover refused a valid layout");
            sets.push_back(allocation);
        }
        const auto pools = cache.Counters().pools;
        if (round == 0) {
            initialPools = pools;
        } else {
            Require(pools <= initialPools + 1, "descriptor pool recycling grows behind one retained set");
        }
        for (const auto& allocation : sets) cache.Free(allocation);
    }
    cache.Free(held);
    std::cout << "Descriptor pool recycling passed across 3,301 sets (" << initialPools << " initial pools, " << cache.Counters().pools << " final pools)\n";
}

void keysFillTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: DCC key fills not tested\n";
        return;
    }
    constexpr std::uint32_t side = 256;
    constexpr std::size_t surfaceBytes = side * side * 4;
    constexpr std::size_t keyCount = surfaceBytes / 256;
    constexpr std::size_t bytes = surfaceBytes + 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the key fill block");
    auto* texels = static_cast<std::uint8_t*>(block);
    auto* keys = texels + surfaceBytes;
    std::memset(texels, 0x55, surfaceBytes);
    std::memset(keys, 0x00, keyCount);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto keysAddress = address + surfaceBytes;
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the key fill block refused: DCC key fills not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    resource.dccAddress = keysAddress;
    Require(DescribeSurface(resource).guestBytes == surfaceBytes, "the key fill surface has an unexpected size");
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto draw = [&](VkClearColorValue value) {
            const auto commands = recorder.Commands();
            recorder.Keep(image);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            image->MarkDirty();
        };
        const auto holds = [&](std::array<std::uint8_t, 4> texel) {
            Buffer readback(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                if (std::to_integer<std::uint8_t>(pixels[i]) != texel[i % 4]) return false;
            }
            return true;
        };
        const auto fill = [&](std::uint8_t key) {
            recorder.Sync();
            std::memset(keys, key, keyCount);
            return StorageTexture::NoteKeysFill(keysAddress, keyCount, key);
        };
        Require(holds({0, 0, 0, 0}), "a surface under 0000 keys was not cleared");
        draw({{1.0f, 0.0f, 0.0f, 1.0f}});
        Require(fill(0x00) == 1, "a 0000 key fill did not cover the surface");
        Require(image->FilledKeys() == DccKeys::Clear0000, "a key fill over pending results was not recorded");
        image->Refresh();
        Require(holds({0, 0, 0, 0}), "a key fill did not clear the results made before it at the next refresh");
        Require(image->FilledKeys() == DccKeys::Uncompressed, "the image cleared by a refresh still holds the fill");
        draw({{0.0f, 0.0f, 1.0f, 1.0f}});
        Require(fill(0x00) == 1, "a second 0000 key fill did not cover the surface");
#ifdef _WIN32
        _putenv_s("APS5_KEYS_FILL_CLEAR", "1");
#else
        setenv("APS5_KEYS_FILL_CLEAR", "1", 1);
#endif
        Require(StorageTexture::ClearByKeysFill(keysAddress, keyCount, 0x00) == 1, "a 0000 key fill did not clear the surface at once");
        Require(holds({0, 0, 0, 0}), "a key fill cleared at once left results made before it");
        draw({{0.0f, 1.0f, 0.0f, 1.0f}});
        Require(image->FilledKeys() == DccKeys::Uncompressed, "results drawn after a key fill cleared at once are held under the fill's code");
        Require(holds({0, 255, 0, 255}), "results drawn after a key fill cleared at once were lost");
    }
    recorder.Sync();
}

void integerClearTests(const Device& device) {
    const auto& context = device.GetContext();
    std::size_t tested = 0;
    for (const std::uint32_t format : {5u, 11u, 20u}) {
        if (!StorageFormatAvailable(context, format)) continue;
        ++tested;
        Require(StorageClearAvailable(context, format, DccKeys::Clear0000), "an integer storage format refused the 0000 clear");
        for (const auto keys : {DccKeys::Clear0001, DccKeys::Clear1111, DccKeys::ClearRegister, DccKeys::Uncompressed}) {
            Require(!StorageClearAvailable(context, format, keys), "an integer storage format took a clear code other than 0000");
        }
    }
    Require(tested == 3, "an integer storage format has no storage form");
    std::cout << "integer storage clears: " << tested << " formats ok\n";
}

bool clearHandOverTests(const Device& device, Recorder& recorder, bool watched) {
    const auto& base = device.GetContext();
    const std::string what = watched ? "clear-key hand-over (watched)" : "clear-key hand-over";
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: " << what << " not tested\n";
        return false;
    }
    constexpr std::size_t surfaceA = 256 * 256 * 4;
    constexpr std::size_t surfaceB = 128 * 128 * 8;
    constexpr std::size_t keyCount = surfaceB / 256;
    constexpr std::size_t bytes = surfaceA + 65536;
    static_assert(surfaceB <= surfaceA && keyCount <= bytes - surfaceA, "B lies over the first units of A, and B's DCC keys follow A inside the block");
    const auto decided = PrepareImportWatch(base);
#ifndef _WIN32
    if (watched && decided == ImportWatch::Unwatch) {
        std::cout << "host imports are compared on this device: " << what << " not tested\n";
        return false;
    }
#endif
    void* block = nullptr;
    if (watched) {
        block = AllocateWatched(bytes, 65536);
        if (block == nullptr) {
            std::cout << "no write watching: " << what << " not tested\n";
            return false;
        }
    } else {
#ifdef _WIN32
        block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        block = std::aligned_alloc(65536, bytes);
#endif
    }
    Require(block != nullptr && reinterpret_cast<std::uint64_t>(block) % 65536 == 0, "cannot allocate the clear-key hand-over block");
    if (watched) SetImportWatch(base, ImportWatch::Watch);
    struct RestoreWatch {
        const Context& context;
        ImportWatch decided;
        ~RestoreWatch() { SetImportWatch(context, decided); }
    } restoreWatch{base, decided};
    auto* memory = static_cast<std::uint8_t*>(block);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        bool watched;
        ~Unregister() {
            StorageTexture::SetClearHandOver(true);
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
            if (watched) ReleaseWatched(block, bytes);
        }
    } unregister{base, block, address, watched};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the clear-key hand-over block refused: " << what << " not tested\n";
        return false;
    }
    if (watched && !AgcDriver::GuestMemory::Watched(address, bytes)) {
        std::cout << "the imported block left the write watch: " << what << " not tested\n";
        return false;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    const auto surface = [&](std::uint32_t width, std::uint32_t height, std::uint32_t format, std::uint64_t dcc) {
        GuestTextureResource resource{};
        resource.baseAddress = address;
        resource.width = width;
        resource.height = height;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kR64KBX;
        resource.dimension = TextureDimension::k2D;
        resource.format = format;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        resource.dccAddress = dcc;
        return resource;
    };
    const auto resourceA = surface(256, 256, 56, 0);
    const auto resourceB = surface(128, 128, 71, address + surfaceA);
    Require(DescribeSurface(resourceA).guestBytes == surfaceA && DescribeSurface(resourceB).guestBytes == surfaceB, "the clear-key hand-over surfaces have unexpected sizes");
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const auto draw = [&](const std::shared_ptr<StorageTexture>& image, VkClearColorValue value) {
        const auto commands = recorder.Commands();
        recorder.Keep(image);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        image->MarkDirty();
    };
    const auto readImage = [&](const std::shared_ptr<StorageTexture>& image, std::uint32_t width, std::uint32_t height, std::size_t texelBytes) {
        Buffer readback(context, static_cast<std::size_t>(width) * height * texelBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
        const auto pixels = readback.Bytes();
        return std::vector<std::byte>(pixels.begin(), pixels.end());
    };
    const auto readMemory = [&](std::uint64_t from, std::size_t length) {
        std::vector<std::byte> read(length);
        AgcDriver::GuestMemory::Read(from, read);
        recorder.Sync();
        if (!watched) std::memcpy(read.data(), memory + (from - address), length);
        return read;
    };
    const auto filled = [](std::span<const std::byte> read, std::span<const std::uint8_t> texel) {
        for (std::size_t i = 0; i < read.size(); ++i) {
            if (std::to_integer<std::uint8_t>(read[i]) != texel[i % texel.size()]) return false;
        }
        return true;
    };
    constexpr std::array<std::uint8_t, 4> red{255, 0, 0, 255};
    constexpr std::array<std::uint8_t, 4> blue{0, 0, 255, 255};
    constexpr std::array<std::uint8_t, 8> green{0, 0, 0x00, 0x3c, 0, 0, 0x00, 0x3c};
    constexpr std::array<std::uint8_t, 8> cleared{};
    enum class Order { Superseded, ReadFirst, Rewritten, CpuStore };
    struct Outcome {
        std::vector<std::byte> bytes;
        std::vector<std::byte> imageB;
        std::uint64_t handOvers = 0;
        std::uint64_t superseded = 0;
    };
    const auto run = [&](bool on, Order order) {
        const std::string where = what + (order == Order::Superseded ? ", A never read again" : order == Order::ReadFirst ? ", A read before B wrote" : order == Order::Rewritten ? ", A written again after the hand-over" : ", a CPU store before B wrote") + (on ? ", on" : ", off");
        StorageTexture::SetClearHandOver(on);
        recorder.Sync();
        std::memset(memory, 0x55, surfaceA);
        std::memset(memory + surfaceA, 0x00, keyCount);
        Outcome outcome{};
        const auto handOvers = StorageTexture::ClearHandOvers();
        const auto superseded = StorageTexture::HandedOverSuperseded();
        {
            auto imageA = StorageTexture::Create(context, detiler, resourceA, 0);
            auto imageB = StorageTexture::Create(context, detiler, resourceB, 0);
            draw(imageA, {{1.0f, 0.0f, 0.0f, 1.0f}});
            imageB->Refresh();
            if (on && watched) {
                Require(StorageTexture::ClearHandOvers() == handOvers + 1, where + ": B's refresh under clear keys did not hand A's results over");
                Require(StorageTexture::FindPending(address, surfaceA) == imageA, where + ": A's results did not stay pending");
                Require(std::all_of(memory, memory + 2 * 65536, [](std::uint8_t value) { return value == 0x55; }), where + ": B's refresh stored A's results over B");
            } else {
                Require(StorageTexture::ClearHandOvers() == handOvers, where + ": the hand-over ran while off");
            }
            outcome.imageB = readImage(imageB, 128, 128, 8);
            Require(filled(outcome.imageB, cleared), where + ": B does not hold its keys' clear");
            if (order == Order::ReadFirst) {
                Require(filled(readMemory(address, 65536), red), where + ": a read of A's memory before B wrote it did not see A's results");
                imageA->Refresh();
                Require(filled(readImage(imageA, 256, 256, 4), red), where + ": A read again as A lost its results");
            }
            if (order == Order::Rewritten) {
                draw(imageA, {{0.0f, 0.0f, 1.0f, 1.0f}});
                Require(filled(readMemory(address, 65536), blue), where + ": a read of A's memory did not see A's newer results");
                imageB->Refresh();
                Require(filled(readImage(imageB, 128, 128, 8), cleared), where + ": B refreshed again does not hold its keys' clear");
            }
            if (order == Order::CpuStore) {
                std::memset(memory + 65536, 0x77, 65536);
            }
            draw(imageB, {{0.0f, 1.0f, 0.0f, 1.0f}});
            Require(StorageTexture::FlushPending(address, bytes, nullptr, "clear-key hand-over test", PublishScope::Whole), where + ": nothing was pending at the end");
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            outcome.bytes = readMemory(address, surfaceA);
            Require(filled(readImage(imageB, 128, 128, 8), green), where + ": B lost its results");
        }
        outcome.handOvers = StorageTexture::ClearHandOvers() - handOvers;
        outcome.superseded = StorageTexture::HandedOverSuperseded() - superseded;
        recorder.Sync();
        return outcome;
    };
    for (const auto order : {Order::Superseded, Order::ReadFirst, Order::Rewritten, Order::CpuStore}) {
        const std::string name = what + (order == Order::Superseded ? ", A never read again" : order == Order::ReadFirst ? ", A read before B wrote" : order == Order::Rewritten ? ", A written again after the hand-over" : ", a CPU store before B wrote");
        const auto off = run(false, order);
        const auto on = run(true, order);
        const auto& lastA = order == Order::Rewritten ? blue : red;
        if (order != Order::CpuStore) Require(filled(std::span(off.bytes).first(2 * 65536), green), name + ": B's results did not reach guest memory with the hand-over off");
        Require(filled(std::span(off.bytes).subspan(2 * 65536), lastA), name + ": A's results did not reach guest memory with the hand-over off");
        Require(on.imageB == off.imageB, name + ": B held other texels with the hand-over on than off");
        Require(on.bytes == off.bytes, name + ": the hand-over left other guest bytes than the old order");
        Require(off.handOvers == 0 && off.superseded == 0, name + ": the hand-over ran while off");
        if (!watched) {
            Require(on.handOvers == 0, name + ": results were handed over in memory the write tracker does not watch");
            continue;
        }
        if (order == Order::Superseded) Require(on.superseded == 2, name + ": B's results did not drop A's two units under B (dropped " + std::to_string(on.superseded) + ")");
        if (order == Order::ReadFirst) Require(on.superseded <= 1, name + ": a unit stored by the read was dropped again");
    }
    std::cout << what << ": ok\n";
    return true;
}

void denormalClearTests(const Device& device, Recorder& recorder) {
    const auto& base = device.GetContext();
    if (base.hostImportAlignment == 0) {
        std::cout << "host imports unavailable: denormal fill clears not tested\n";
        return;
    }
    constexpr std::uint32_t side = 64;
    constexpr std::size_t bytes = 65536;
#ifdef _WIN32
    void* block = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* block = std::aligned_alloc(65536, bytes);
#endif
    Require(block != nullptr, "cannot allocate the denormal clear block");
    std::memset(block, 0x55, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(block, bytes, true, true, true);
    }
    struct Unregister {
        const Context& context;
        void* block;
        std::uint64_t address;
        ~Unregister() {
            {
                GuestAllocations::Mutation mutation;
                mutation.Remove(block);
            }
            HostImportFor(context, address, bytes);
        }
    } unregister{base, block, address};
    if (HostImportFor(base, address, bytes) == nullptr) {
        std::cout << "host import of the denormal clear block refused: denormal fill clears not tested\n";
        return;
    }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = side;
    resource.height = side;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 29;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    Require(ResolveTextureFormat(resource.format) == VK_FORMAT_R16G16_SFLOAT, "guest format 29 is not R16G16_SFLOAT");
    const bool keeps = ClearKeepsDenormals(context, VK_FORMAT_R16G16_SFLOAT);
    std::cout << "denormal half clears are " << (keeps ? "kept" : "flushed") << " by this device\n";
    Require(ClearKeepsDenormals(context, VK_FORMAT_R16G16_SFLOAT) == keeps, "the denormal clear probe changed its answer");
    Require(!ClearKeepsDenormals(context, VK_FORMAT_R32G32B32A32_SFLOAT), "a format without small floats probed as keeping denormals");
    {
        auto image = StorageTexture::Create(context, detiler, resource, 0);
        const auto holds = [&](std::uint32_t texel) {
            Buffer readback(context, side * side * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            const auto commands = recorder.Commands();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {side, side, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            recorder.Submit();
            device.WaitQueue();
            recorder.Sync();
            const auto pixels = readback.Bytes();
            for (std::size_t at = 0; at + 4 <= pixels.size(); at += 4) {
                std::uint32_t value = 0;
                std::memcpy(&value, pixels.data() + at, 4);
                if (value != texel) return false;
            }
            return true;
        };
        const char* refusal = nullptr;
        const std::array<std::uint32_t, 4> normal{0x3c00bc00u, 0x3c00bc00u, 0x3c00bc00u, 0x3c00bc00u};
        Require(image->FillClear(normal, StorageTexture::WholeImage, refusal), "a normal half pattern was refused");
        Require(holds(0x3c00bc00u), "a normal half fill clear stored other bits");
        const std::array<std::uint32_t, 4> denormal{0x83ff0001u, 0x83ff0001u, 0x83ff0001u, 0x83ff0001u};
        refusal = nullptr;
        const bool cleared = image->FillClear(denormal, StorageTexture::WholeImage, refusal);
        Require(cleared == keeps, keeps ? "a denormal half pattern was refused on a device whose clears keep denormals" : "a denormal half pattern was cleared on a device whose clears flush them");
        if (cleared) Require(holds(0x83ff0001u), "a denormal half fill clear stored other bits");
        const std::array<std::uint32_t, 4> nan{0x7e007e00u, 0x7e007e00u, 0x7e007e00u, 0x7e007e00u};
        refusal = nullptr;
        Require(!image->FillClear(nan, StorageTexture::WholeImage, refusal), "a NaN half pattern was cleared");
    }
    recorder.Sync();
}

// Counts the vkCmd* calls the bound-state filter decides on in Pipeline (binds, viewport and
// scissor, the library state, push constants), forwarding each to the device: the context's entry
// point table is swapped for one of counting shims while the object lives.
class BoundStateCalls {
public:
    enum Entry : std::size_t { BindPipeline = 0, Viewport, Scissor, Library, Push, EntryCount };
    explicit BoundStateCalls(Context& context) : context(context), previous(context.functions) {
        Require(active == nullptr, "nested bound-state call counts");
        if (previous != nullptr) functions = *previous;
        install<&DeviceFunctions::cmdBindPipeline, BindPipeline, 0>("vkCmdBindPipeline");
        install<&DeviceFunctions::cmdSetViewport, Viewport, 1>("vkCmdSetViewport");
        install<&DeviceFunctions::cmdSetScissor, Scissor, 2>("vkCmdSetScissor");
        install<&DeviceFunctions::cmdPushConstants, Push, 3>("vkCmdPushConstants");
        install<&DeviceFunctions::cmdSetDepthBias, Library, 4>("vkCmdSetDepthBias");
        install<&DeviceFunctions::cmdSetDepthBounds, Library, 5>("vkCmdSetDepthBounds");
        install<&DeviceFunctions::cmdSetStencilCompareMask, Library, 6>("vkCmdSetStencilCompareMask");
        install<&DeviceFunctions::cmdSetStencilWriteMask, Library, 7>("vkCmdSetStencilWriteMask");
        install<&DeviceFunctions::cmdSetStencilReference, Library, 8>("vkCmdSetStencilReference");
        // The extended dynamic state is enabled with the pipeline libraries only.
        if (context.graphicsPipelineLibrary) {
            install<&DeviceFunctions::cmdSetCullMode, Library, 9>("vkCmdSetCullModeEXT");
            install<&DeviceFunctions::cmdSetFrontFace, Library, 10>("vkCmdSetFrontFaceEXT");
            install<&DeviceFunctions::cmdSetDepthTestEnable, Library, 11>("vkCmdSetDepthTestEnableEXT");
            install<&DeviceFunctions::cmdSetDepthWriteEnable, Library, 12>("vkCmdSetDepthWriteEnableEXT");
            install<&DeviceFunctions::cmdSetDepthCompareOp, Library, 13>("vkCmdSetDepthCompareOpEXT");
            install<&DeviceFunctions::cmdSetDepthBoundsTestEnable, Library, 14>("vkCmdSetDepthBoundsTestEnableEXT");
            install<&DeviceFunctions::cmdSetStencilTestEnable, Library, 15>("vkCmdSetStencilTestEnableEXT");
            install<&DeviceFunctions::cmdSetStencilOp, Library, 16>("vkCmdSetStencilOpEXT");
        }
        context.functions = &functions;
        active = this;
    }
    ~BoundStateCalls() {
        context.functions = previous;
        active = nullptr;
    }
    BoundStateCalls(const BoundStateCalls&) = delete;
    BoundStateCalls& operator=(const BoundStateCalls&) = delete;
    void Reset() { counts = {}; }
    std::size_t Total() const {
        std::size_t total = 0;
        for (const auto count : counts) total += count;
        return total;
    }
    std::array<std::size_t, EntryCount> counts{};

private:
    template<typename TFunction>
    struct Forward;
    template<typename... TArgs>
    struct Forward<void (VKAPI_PTR*)(TArgs...)> {
        template<std::size_t Index, std::size_t Slot>
        static VKAPI_ATTR void VKAPI_CALL Call(TArgs... args) {
            ++active->counts[Index];
            reinterpret_cast<void (VKAPI_PTR*)(TArgs...)>(active->real[Slot])(args...);
        }
    };
    template<auto Member, std::size_t Index, std::size_t Slot>
    void install(const char* name) {
        const auto resolved = context.Resolved(Member, name);
        using Function = std::remove_const_t<decltype(resolved)>;
        real[Slot] = reinterpret_cast<PFN_vkVoidFunction>(resolved);
        functions.*Member = &Forward<Function>::template Call<Index, Slot>;
    }
    Context& context;
    const DeviceFunctions* previous;
    DeviceFunctions functions{};
    std::array<PFN_vkVoidFunction, 17> real{};
    static inline BoundStateCalls* active = nullptr;
};

// The bound-state record's own decisions for descriptor sets, push constants, vertex and index
// buffers and the dynamic states a pipeline bind keeps or forgets. Without `countOnly` a call that
// repeats what the record holds is left out; with it every call records.
void boundStateRecordTests(bool countOnly) {
    using Dynamic = BoundState::Dynamic;
    BoundState record;
    record.countOnly = countOnly;
    const bool skips = !countOnly;
    const auto handle = [](std::uintptr_t value) { return value; };
    const auto layoutA = reinterpret_cast<VkPipelineLayout>(handle(0x1000));
    const auto layoutB = reinterpret_cast<VkPipelineLayout>(handle(0x2000));
    const auto pipelineA = reinterpret_cast<VkPipeline>(handle(0x3000));
    const auto pipelineB = reinterpret_cast<VkPipeline>(handle(0x4000));
    const auto setA = reinterpret_cast<VkDescriptorSet>(handle(0x5000));
    const auto setB = reinterpret_cast<VkDescriptorSet>(handle(0x6000));
    Require(record.ChangesSet(layoutA, 0, setA) && record.ChangesSet(layoutA, 1, setB), "a first set bind was left out");
    Require(record.ChangesSet(layoutA, 0, setA) != skips && record.ChangesSet(layoutA, 1, setB) != skips, "a repeated set bind was not decided by the record");
    Require(record.ChangesSet(layoutB, 0, setA), "a set bound under another layout was left out");
    Require(record.ChangesSet(layoutA, 1, setB), "a set that a bind under another layout may have disturbed was left out");
    std::array<std::byte, BoundState::PushBytes> block{};
    Require(record.ChangesPush(layoutA, VK_SHADER_STAGE_VERTEX_BIT, block), "a first push was left out");
    Require(record.ChangesPush(layoutA, VK_SHADER_STAGE_VERTEX_BIT, block) != skips, "a repeated push was not decided by the record");
    Require(record.ChangesPush(layoutA, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, block), "a push for other stages was left out");
    block[7] = std::byte{1};
    Require(record.ChangesPush(layoutA, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, block), "a push of other bytes was left out");
    Require(record.ChangesPipeline(pipelineA, layoutB, true, false, false), "a first pipeline bind was left out");
    Require(record.ChangesPush(layoutA, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, block), "a push the bind of another layout's pipeline may have disturbed was left out");
    Require(record.ChangesSet(layoutA, 1, setB), "a set the bind of another layout's pipeline may have disturbed was left out");
    Require(record.ChangesPipeline(pipelineA, layoutB, true, false, false) != skips, "a repeated pipeline bind was not decided by the record");
    const VkViewport viewport{0, 0, 8, 8, 0, 1};
    const VkCullModeFlags back = VK_CULL_MODE_BACK_BIT;
    const std::array<float, 3> bias{1.0f, 0.0f, 2.0f};
    Require(record.Changes(Dynamic::Viewport, viewport) && record.Changes(Dynamic::CullMode, back) && record.Changes(Dynamic::DepthBias, bias), "a first dynamic state was left out");
    Require(record.Changes(Dynamic::Viewport, viewport) != skips && record.Changes(Dynamic::CullMode, back) != skips && record.Changes(Dynamic::DepthBias, bias) != skips, "a repeated dynamic state was not decided by the record");
    Require(record.Changes(Dynamic::CullMode, VkCullModeFlags{VK_CULL_MODE_FRONT_BIT}) && record.Changes(Dynamic::CullMode, back), "another cull mode was left out");
    // A whole pipeline that keeps depth bias dynamic: viewport and depth bias stay, the cull mode
    // (static there) is forgotten.
    Require(record.ChangesPipeline(pipelineB, layoutB, false, true, false), "a whole pipeline's bind was left out");
    Require(record.Changes(Dynamic::Viewport, viewport) != skips && record.Changes(Dynamic::DepthBias, bias) != skips, "a whole pipeline's bind forgot a state it keeps dynamic");
    Require(record.Changes(Dynamic::CullMode, back), "a whole pipeline's bind kept a state it declares static");
    Require(record.ChangesPipeline(pipelineA, layoutB, false, false, false), "a whole pipeline's bind was left out");
    Require(record.Changes(Dynamic::DepthBias, bias), "a whole pipeline's bind kept a depth bias it declares static");
    const std::array<VkBuffer, 3> buffers{reinterpret_cast<VkBuffer>(handle(0x7000)), reinterpret_cast<VkBuffer>(handle(0x8000)), reinterpret_cast<VkBuffer>(handle(0x9000))};
    const std::array<VkDeviceSize, 3> offsets{0, 16, 32};
    const auto vertex = [&](std::size_t count, std::span<const VkDeviceSize> at) { return record.ChangesVertexBuffers(std::span(buffers.data(), count), at.first(count)); };
    Require(vertex(2, offsets), "a first vertex buffer bind was left out");
    Require(vertex(2, offsets) != skips, "a repeated vertex buffer bind was not decided by the record");
    Require(vertex(1, offsets) != skips, "a bind of fewer slots than the record holds was not decided by the record");
    Require(vertex(3, offsets), "a bind of a slot the record does not hold was left out");
    const std::array<VkDeviceSize, 3> moved{0, 48, 32};
    Require(vertex(2, moved), "a vertex buffer bind at another offset was left out");
    std::array<VkBuffer, BoundState::VertexSlots + 1> many{};
    many.fill(buffers[0]);
    const std::array<VkDeviceSize, BoundState::VertexSlots + 1> manyOffsets{};
    Require(record.ChangesVertexBuffers(many, manyOffsets), "a bind of more slots than the record follows was left out");
    Require(record.ChangesVertexBuffers(std::span(many.data(), 1), std::span(manyOffsets.data(), 1)), "the record kept slots a bind of more than it follows overwrote");
    Require(record.ChangesIndexBuffer(buffers[0], 0, VK_INDEX_TYPE_UINT16), "a first index buffer bind was left out");
    Require(record.ChangesIndexBuffer(buffers[0], 0, VK_INDEX_TYPE_UINT16) != skips, "a repeated index buffer bind was not decided by the record");
    Require(record.ChangesIndexBuffer(buffers[0], 0, VK_INDEX_TYPE_UINT32) && record.ChangesIndexBuffer(buffers[0], 4, VK_INDEX_TYPE_UINT32) && record.ChangesIndexBuffer(buffers[1], 4, VK_INDEX_TYPE_UINT32), "an index buffer bind of another type, offset or buffer was left out");
    record.Clear();
    Require(record.countOnly == countOnly, "clearing the record changed its mode");
    Require(record.ChangesPipeline(pipelineA, layoutB, false, false, false) && record.Changes(Dynamic::Viewport, viewport) && record.ChangesSet(layoutB, 0, setA) && record.ChangesPush(layoutB, VK_SHADER_STAGE_VERTEX_BIT, block) && vertex(1, offsets) && record.ChangesIndexBuffer(buffers[1], 4, VK_INDEX_TYPE_UINT32), "a cleared record left a call out");
    Require(record.tally[static_cast<std::size_t>(BoundState::Call::Index)][0] == 1 && record.tally[static_cast<std::size_t>(BoundState::Call::Index)][1] == 5, "the record miscounted repeated index buffer binds");
}

// LOCAL EXPERIMENT APS5_EXPERIMENT_PASS_STATE_FILTER (registered unset, =1 and =2): with the
// filter a draw continuing the open pass records only what its predecessors did not bind or set,
// and what the record does not follow clears it: the pass's end, a new batch, a pipeline with
// static state (its library states), a face split and a second channel. Unset and under =2 every
// call records (=2 still counts the repeats).
int boundStateTests(const Device& device, Recorder& recorder) {
    const int mode = Recorder::PassStateFilter();
    const bool filters = mode == 1;
    boundStateRecordTests(false);
    boundStateRecordTests(true);
    if (!device.Graphics()) {
        std::cout << "skipped, the device has no graphics queue\n";
        return 77;
    }
    auto context = device.GetContext();
    BoundStateCalls calls(context);
    struct Cleanup {
        VkDevice device;
        ~Cleanup() { ClearCachedPipelines(device); }
    } cleanup{context.device};
    ShaderRecompiler::RecompileResult vertex, fragment;
    vertex.variantId = 31;
    fragment.variantId = 32;
    vertex.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_vert_SPV), std::end(TRIANGLE_vert_SPV));
    fragment.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_frag_SPV), std::end(TRIANGLE_frag_SPV));
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    State state{};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.hasColorTarget = true;
    state.color = ColorTarget{0, {8, 8}, VK_FORMAT_R8G8B8A8_UNORM, 8 * 8 * 4, 0};
    state.colors.push_back(state.color);
    state.blend.colorWriteMask = 15;
    state.blends.push_back(state.blend);
    state.renderExtent = {8, 8};
    state.viewport = {0, 0, 8, 8, 0, 1};
    state.scissor = {{0, 0}, {8, 8}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    ShaderResources resources(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), 0, 0);
    VertexInputLayout input;
    auto culledState = state;
    culledState.cullMode = VK_CULL_MODE_BACK_BIT;
    // The alternate component order with a separate alpha blend: a whole pipeline (static state)
    // with a second channel, even where pipelines are built from libraries.
    auto splitState = state;
    splitState.colors.front().componentMapping = 0xecu;
    splitState.blends.front().blendEnable = VK_TRUE;
    splitState.blends.front().srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    splitState.blends.front().dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    splitState.blends.front().srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    splitState.blends.front().dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    auto faceState = state;
    faceState.depth = DepthTarget{0, 0, {8, 8}, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    faceState.depthBias = true;
    faceState.depthBiasPerFace = true;
    faceState.depthBiasConstant = 1.0f;
    faceState.backDepthBiasConstant = 2.0f;
    Pipeline first(context, state, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL);
    Pipeline culled(context, culledState, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL);
    Pipeline split(context, splitState, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL);
    Pipeline faces(context, faceState, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL);
    Require(split.SplitsSecondChannel() && faces.SplitsFaces(), "the split pipelines do not split");
    // The optimized pipelines are ready before the first bind, so every draw binds the same handle.
    WaitForOptimizedPipelines(context.device);
    RenderTarget target(context, state.color, false);
    const std::array<VkImageView, 1> views{target.View()};
    const std::array<std::shared_ptr<void>, 1> owners{};
    const auto framebuffer = first.AcquireFramebuffer(views, owners, state.renderExtent);
    {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = target.Image();
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(recorder.Commands(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    // Pipelines built from libraries set 17 library states; whole ones set none here (no depth).
    const std::size_t library = context.graphicsPipelineLibrary ? 17u : 0u;
    // Only pipeline layouts of library pipelines have push constant ranges for these shaders.
    const bool pushes = context.graphicsPipelineLibrary;
    const std::array<std::byte, PipelinePushConstantBytes> block{};
    constexpr std::uint64_t key = 0xb0d5;
    const auto expect = [&](std::size_t filtered, std::size_t full, const char* what) {
        if (calls.Total() == (filters ? filtered : full)) return;
        throw std::runtime_error(std::string("AGC graphics: ") + what + ": " + std::to_string(calls.Total()) + " calls recorded, expected " + std::to_string(filters ? filtered : full));
    };
    const auto begin = [&](const char* what) {
        const auto start = recorder.StartDrawPass(key, false, false, PassAccess{});
        Require(!start.continued, what);
        auto* bound = recorder.PassBoundState();
        Require((bound != nullptr) == (mode != 0), "the bound-state record is handed out without the switch, or withheld under it");
        if (bound != nullptr) Require(bound->held.pipeline == VK_NULL_HANDLE && bound->held.known == 0 && bound->held.pushLayout == VK_NULL_HANDLE && bound->held.vertexSlots == 0 && !bound->held.indexKnown, "a pass started with a filled bound-state record");
        calls.Reset();
        first.Begin(start.commands, *framebuffer, state.renderExtent, state, bound);
        if (pushes) first.PushConstants(start.commands, VK_SHADER_STAGE_VERTEX_BIT, block, bound);
        expect(3 + library + (pushes ? 2 : 0), 3 + library + (pushes ? 2 : 0), what);
        recorder.LeaveRenderPassOpen(key, Recorder::NoTiming, false, PassAccess{});
        return bound;
    };
    const auto next = [&](BoundState* bound) {
        const auto start = recorder.StartDrawPass(key, false, false, PassAccess{});
        Require(start.continued, "a draw into the open pass did not continue it");
        Require(recorder.PassBoundState() == bound, "a continued draw got another bound-state record");
        calls.Reset();
        return start.commands;
    };
    auto* bound = begin("the first draw continued a pass");
    {
        const auto before = bound != nullptr ? bound->tally : decltype(bound->tally){};
        const auto commands = next(bound);
        first.Continue(commands, state, bound);
        if (pushes) first.PushConstants(commands, VK_SHADER_STAGE_VERTEX_BIT, block, bound);
        expect(0, 3 + library + (pushes ? 2 : 0), "an identical continued draw");
        if (bound != nullptr) {
            const auto repeated = [&](BoundState::Call call) { return bound->tally[static_cast<std::size_t>(call)][0] - before[static_cast<std::size_t>(call)][0]; };
            Require(repeated(BoundState::Call::Pipeline) == 1 && repeated(BoundState::Call::Viewport) == 1 && repeated(BoundState::Call::Scissor) == 1 && repeated(BoundState::Call::Library) == library && repeated(BoundState::Call::Push) == (pushes ? 1u : 0u), "the record did not count every call of an identical draw as a repeat");
        }
        recorder.LeaveRenderPassOpen(key, Recorder::NoTiming, false, PassAccess{});
    }
    {
        // Another pipeline that differs in its cull mode alone: its bind and (from libraries) the
        // cull mode; its push constants follow another layout.
        const auto commands = next(bound);
        culled.Continue(commands, culledState, bound);
        expect(1 + (library != 0 ? 1 : 0), 3 + library, "a continued draw of another cull mode");
        if (pushes) {
            calls.Reset();
            culled.PushConstants(commands, VK_SHADER_STAGE_VERTEX_BIT, block, bound);
            expect(2, 2, "a push under another pipeline's layout");
            calls.Reset();
            culled.PushConstants(commands, VK_SHADER_STAGE_VERTEX_BIT, block, bound);
            expect(0, 2, "a repeated push");
            auto changed = block;
            changed[4] = std::byte{1};
            calls.Reset();
            culled.PushConstants(commands, VK_SHADER_STAGE_VERTEX_BIT, changed, bound);
            expect(2, 2, "a push of other bytes");
        }
        recorder.LeaveRenderPassOpen(key, Recorder::NoTiming, false, PassAccess{});
    }
    {
        // A pipeline with static state: its bind alone (viewport and scissor are dynamic in every
        // pipeline), after which a library pipeline sets its library states again.
        const auto commands = next(bound);
        split.Continue(commands, splitState, bound);
        expect(1, 3, "a continued draw of a whole pipeline");
        calls.Reset();
        first.Continue(commands, state, bound);
        expect(1 + library, 3 + library, "a library pipeline's draw after a whole pipeline's");
        recorder.LeaveRenderPassOpen(key, Recorder::NoTiming, false, PassAccess{});
    }
    {
        // A second channel clears the record: the next draw records everything.
        const auto commands = next(bound);
        split.Continue(commands, splitState, bound);
        split.ContinueSecondChannel(commands, bound);
        if (bound != nullptr) Require(bound->held.pipeline == VK_NULL_HANDLE && bound->held.known == 0, "a second channel did not clear the bound-state record");
        calls.Reset();
        first.Continue(commands, state, bound);
        expect(3 + library, 3 + library, "a draw after a second channel");
        recorder.LeaveRenderPassOpen(key, Recorder::NoTiming, false, PassAccess{});
    }
    // The pass's end clears the record, and so does a new batch.
    static_cast<void>(recorder.Commands());
    if (bound != nullptr) Require(recorder.PassBoundState() == bound && bound->held.pipeline == VK_NULL_HANDLE && bound->held.known == 0, "the end of a pass did not clear its bound-state record");
    bound = begin("a draw after the end of the pass continued it");
    {
        const auto commands = next(bound);
        first.Continue(commands, state, bound);
        expect(0, 3 + library, "an identical draw in the second pass");
        recorder.LeaveRenderPassOpen(key, Recorder::NoTiming, false, PassAccess{});
    }
    recorder.Submit();
    static_cast<void>(begin("the first draw of a new batch continued a pass"));
    // A face split clears the record (recorded outside a pass: binds and dynamic state need none).
    {
        const auto commands = recorder.Commands();
        BoundState local;
        local.countOnly = !filters;
        calls.Reset();
        faces.Continue(commands, faceState, &local);
        expect(4, 4, "a whole pipeline with depth bias");
        calls.Reset();
        faces.Continue(commands, faceState, &local);
        expect(0, 4, "a repeated whole pipeline with depth bias");
        faces.ContinueBackFaces(commands, faceState, &local);
        Require(local.held.pipeline == VK_NULL_HANDLE && local.held.known == 0, "a face split did not clear the bound-state record");
        calls.Reset();
        faces.Continue(commands, faceState, &local);
        expect(4, 4, "a draw after a face split");
    }
    recorder.Sync();
    std::cout << "Bound-state filter tests passed (" << (mode == 0 ? "switch unset" : mode == 1 ? "filter" : "count only") << ", " << (context.graphicsPipelineLibrary ? "pipeline libraries" : "whole pipelines") << ")\n";
    return 0;
}

void pipelineLibraryTests(const Device& device) {
    std::lock_guard gpu(GpuMutex());
    const auto& context = device.GetContext();
    struct Cleanup {
        VkDevice device;
        ~Cleanup() { ClearCachedPipelines(device); }
    } cleanup{context.device};
    ClearCachedPipelines(context.device);
    ShaderRecompiler::RecompileResult vertex, fragment;
    vertex.variantId = 11;
    fragment.variantId = 12;
    vertex.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_vert_SPV), std::end(TRIANGLE_vert_SPV));
    fragment.spirv = std::vector<std::uint32_t>(std::begin(TRIANGLE_frag_SPV), std::end(TRIANGLE_frag_SPV));
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    State state{};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.hasColorTarget = true;
    state.color.format = VK_FORMAT_R8G8B8A8_UNORM;
    state.colors.push_back(state.color);
    state.blend.colorWriteMask = 15;
    state.blends.push_back(state.blend);
    ShaderResources resources(context, shaders, state.color, static_cast<std::uint32_t>(state.colors.size()), 0, 0);
    VertexInputLayout input;
    const auto lookup = [&] { return CachedPipeline(context, state, input, resources, shaders, VK_IMAGE_LAYOUT_GENERAL); };
    const auto counters = [&] { return PipelineLibraryCountersOf(context.device); };
    const auto first = lookup();
    if (!context.graphicsPipelineLibrary) {
        Require(counters().linked == 0, "a pipeline was linked from libraries without VK_EXT_graphics_pipeline_library");
        std::cout << "graphics pipeline libraries unavailable or disabled: pipelines are created whole\n";
        return;
    }
    const auto built = counters();
    Require(built.linked == 1 && built.built == std::array<std::uint64_t, 4>{1, 1, 1, 1}, "the first pipeline was not linked from four new libraries");
    state.blends.front().blendEnable = VK_TRUE;
    state.blends.front().srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    state.blends.front().dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    const auto blended = lookup();
    Require(blended != first && counters().built == std::array<std::uint64_t, 4>{1, 1, 1, 2}, "a blend change rebuilt more than the fragment output library");
    state.cullMode = VK_CULL_MODE_BACK_BIT;
    state.frontFace = VK_FRONT_FACE_CLOCKWISE;
    const auto culled = lookup();
    Require(culled != blended && counters().built == std::array<std::uint64_t, 4>{1, 1, 1, 2}, "a cull mode change rebuilt a library instead of setting dynamic state");
    input.bindings.push_back({0, 16, VK_VERTEX_INPUT_RATE_VERTEX});
    input.attributes.push_back({0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0});
    static_cast<void>(lookup());
    Require(counters().built == std::array<std::uint64_t, 4>{2, 1, 1, 2} && counters().linked == 4, "a vertex input change rebuilt more than the vertex input library");
    Require(StagePushOffset(88, ShaderRecompiler::ShaderStage::Fragment, true) == PipelinePushSlotBytes && StagePushOffset(88, ShaderRecompiler::ShaderStage::Fragment, false) == 88 && StagePushOffset(40, ShaderRecompiler::ShaderStage::TessellationEvaluation, true) == 40, "a stage got the wrong push offset");
    const auto beforeSlots = counters().built;
    ShaderRecompiler::RecompileResult otherVertex = vertex;
    otherVertex.spirv[2] ^= 1u;
    std::array slotted{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, PipelinePushSlotBytes}};
    static_cast<void>(CachedPipeline(context, state, input, resources, slotted, VK_IMAGE_LAYOUT_GENERAL));
    slotted.front().program = &otherVertex;
    static_cast<void>(CachedPipeline(context, state, input, resources, slotted, VK_IMAGE_LAYOUT_GENERAL));
    Require(counters().built[2] == beforeSlots[2] + 1u && counters().built[1] == beforeSlots[1] + 1u, "a pixel shader in its fixed slot built its library again for another vertex shader");
    fragment.spirv[2] ^= 1u;
    static_cast<void>(lookup());
    Require(counters().built == std::array<std::uint64_t, 4>{2, 2, 3, 2}, "a new pixel shader rebuilt more than the fragment shader library");
    state.colors.front().format = VK_FORMAT_R16G16B16A16_SFLOAT;
    static_cast<void>(lookup());
    Require(counters().built == std::array<std::uint64_t, 4>{2, 2, 3, 3}, "another target format rebuilt more than the fragment output library");
    state.samples = 4;
    state.colors.front().samples = 4;
    const auto multisampled = lookup();
    Require(counters().built == std::array<std::uint64_t, 4>{2, 2, 4, 4}, "a 4x pipeline shared the 1x fragment shader or fragment output library");
    state.samples = 1;
    state.colors.front().samples = 1;
    Require(lookup() != multisampled && counters().built == std::array<std::uint64_t, 4>{2, 2, 4, 4}, "the 1x pipeline after a 4x one built its libraries again");
    state.depth = DepthTarget{0, 0, {64, 64}, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    state.depthTest = true;
    state.depthCompare = VK_COMPARE_OP_LESS;
    static_cast<void>(lookup());
    Require(counters().built == std::array<std::uint64_t, 4>{2, 3, 5, 5}, "a depth target shared the shader libraries built with no depth format");
    state.depth.reset();
    state.depthTest = false;
    if ((context.sampleLocationSampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0) {
        state.samples = 4;
        state.colors.front().samples = 4;
        state.customSampleLocations = true;
        state.sampleLocations = {0x10, 0x21, 0x32, 0x43};
        const auto pattern = lookup();
        const auto beforePatternChange = counters().built;
        state.sampleLocations[0] = 0x20;
        const auto otherPattern = lookup();
        Require(otherPattern != pattern && counters().built[2] == beforePatternChange[2] + 1u && counters().built[3] == beforePatternChange[3] + 1u && counters().built[0] == beforePatternChange[0] && counters().built[1] == beforePatternChange[1], "a custom sample pattern did not rebuild exactly the fragment shader and fragment output libraries");
        state.sampleLocations[0] = 0x10;
        Require(lookup() == pattern && counters().built[2] == beforePatternChange[2] + 1u && counters().built[3] == beforePatternChange[3] + 1u, "a cached custom sample pattern was not reused");
        if ((context.sampleLocationGridSampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0) {
            state.sampleLocationGrid = true;
            state.sampleLocations[3u * 8u] = 0x20;
            const auto gridPattern = lookup();
            Require(gridPattern != pattern && counters().built[2] == beforePatternChange[2] + 2u && counters().built[3] == beforePatternChange[3] + 2u, "a pattern varying between pixels shared a uniform pattern's fragment libraries");
            state.sampleLocationGrid = false;
            state.sampleLocations[3u * 8u] = 0;
            Require(lookup() == pattern, "the uniform pattern was not reused after a grid pattern");
        }

        std::optional<VkFormat> compatibleDepth, incompatibleDepth;
        state.customSampleLocations = false;
        for (const auto format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
            state.depth = DepthTarget{};
            state.depth->format = format;
            state.depth->samples = state.samples;
            (ValidateSampleLocations(context, state) ? compatibleDepth : incompatibleDepth) = format;
        }
        if (compatibleDepth && incompatibleDepth) {
            state.depth->format = *compatibleDepth;
            const auto compatible = lookup();
            const auto beforeDepthChange = counters().built;
            state.depth->format = *incompatibleDepth;
            const auto incompatible = lookup();
            Require(incompatible != compatible && counters().built[2] == beforeDepthChange[2] + 1u && counters().built[3] == beforeDepthChange[3] + 1u, "depth-format sample-location enablement did not rebuild both fragment libraries");
        }
        state.customSampleLocations = true;
    }
    WaitForOptimizedPipelines(context.device);
    const auto lto = std::getenv("APS5_NO_PIPELINE_LTO") == nullptr && context.pipelineLinkTimeOptimization;
    Require(counters().optimized == (lto ? counters().linked : 0u), "a linked pipeline was not optimized in the background");
    Require(first->Optimized() == lto && culled->Optimized() == lto, "a pipeline did not switch to its optimized link");
    ClearCachedPipelines(context.device);
    Require(counters().linked == 0, "clearing the pipelines kept the device's libraries");
    std::cout << "Pipeline library reuse tests passed\n";
}

#ifdef _WIN32
namespace StorageCpuSnapshotTest {
constexpr std::uint32_t Side = 256;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileR64KBX = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t SurfaceBytes = std::size_t{Side} * Side * 4u;
constexpr std::size_t BlockBytes = 12 * SurfaceBytes;

class Block {
public:
    explicit Block(const Context& context, bool watched = false) : context(context), watched(watched) {
        if (watched) {
            memory = GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536);
            const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, BlockBytes, nullptr);
            Require(section != nullptr, "cannot allocate the shared texture section");
            try { GuestArena::GuestArenaMap_nid_postfix(memory, BlockBytes, section, 0, PAGE_READWRITE); }
            catch (...) { CloseHandle(section); throw; }
            CloseHandle(section);
        } else memory = VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        Require(memory != nullptr, "cannot allocate the texture block");
        std::memset(memory, 0x5a, BlockBytes);
        GuestAllocations::Mutation().Add(memory, BlockBytes, true, true, true);
    }

    ~Block() {
        ClearCachedTextures(context.device);
        GuestAllocations::Mutation().Remove(memory);
        HostImportFor(context, Surface(0), BlockBytes);
        if (watched) {
            GuestArena::GuestArenaReset_nid_postfix(memory, BlockBytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, BlockBytes);
        } else VirtualFree(memory, 0, MEM_RELEASE);
    }

    std::uint64_t Surface(std::size_t surface) const { return reinterpret_cast<std::uint64_t>(memory) + surface * SurfaceBytes; }

private:
    const Context& context;
    bool watched;
    void* memory = nullptr;
};

std::array<std::uint32_t, 8> Descriptor(std::uint64_t address, std::uint32_t side = Side) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

GuestTextureResource Resource(std::uint64_t address, std::uint32_t side = Side) {
    const auto words = Descriptor(address, side);
    return DecodeTextureResource(words);
}

void Run(Context context) {
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    for (const bool watched : {true, false}) {
        Block block(context, watched);
        const auto address = block.Surface(0);
        Require(AgcDriver::GuestMemory::Watched(address, SurfaceBytes) == watched, "snapshot fixture has the wrong write tracking");
        Require(HostImportFor(context, address, SurfaceBytes) != nullptr, "snapshot fixture requires a host import");
        Require(AgcDriver::GuestMemory::Watched(address, SurfaceBytes) == watched, "host import changed the fixture's write tracking");
        auto image = StorageTexture::Create(context, *context.detiler, Resource(address), 0);
        const auto expectedBytes = watched ? 0u : SurfaceBytes;
        Require(image->OriginalSnapshotBytes() == expectedBytes, "watched GPU-direct upload retained an unnecessary CPU surface snapshot");
        const auto readPixel = [&] {
            Buffer result(context, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            CommandBatch batch(context);
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {1, 1, 1};
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(batch.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, result.Handle(), 1, &copy);
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            batch.SubmitAndWait();
            result.Invalidate();
            std::uint32_t pixel = 0;
            std::memcpy(&pixel, result.Bytes().data(), sizeof(pixel));
            return pixel;
        };
        Require(readPixel() == 0x5a5a5a5au, "snapshot upload returned incorrect GPU texels");
        std::memset(reinterpret_cast<void*>(address), 0x6b, SurfaceBytes);
        image->Refresh();
        Require(readPixel() == 0x6b6b6b6bu, "snapshot refresh missed a CPU write");
        Require(image->OriginalSnapshotBytes() == expectedBytes, "watched refresh allocated a CPU surface snapshot");
        {
            CommandBatch batch(context);
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkClearColorValue clear{};
            clear.uint32[0] = 0x7c7c7c7cu;
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(batch.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            batch.SubmitAndWait();
        }
        image->MarkDirty();
        image->WriteBack();
        std::uint32_t pixel = 0;
        AgcDriver::GuestMemory::ReadCommitted(address, std::as_writable_bytes(std::span(&pixel, 1)));
        Require(pixel == 0x7c7c7c7cu, "snapshot write-back lost GPU texels");
        Require(image->OriginalSnapshotBytes() == expectedBytes, "GPU-direct write-back allocated a CPU surface snapshot");
        if (watched) {
            AgcDriver::GuestMemory::Unwatch(address, SurfaceBytes);
            Require(!AgcDriver::GuestMemory::Watched(address, SurfaceBytes), "snapshot fixture is still write-watched");
            std::memset(reinterpret_cast<void*>(address), 0x8d, SurfaceBytes);
            image->Refresh();
            Require(image->OriginalSnapshotBytes() == SurfaceBytes, "unwatched refresh did not allocate its comparison snapshot");
            Require(readPixel() == 0x8d8d8d8du, "unwatched refresh returned incorrect GPU texels");
        }
        image.reset();
        Context cpu = context;
        cpu.hostImportAlignment = 0;
        image = StorageTexture::Create(cpu, *context.detiler, Resource(address), 0);
        Require(image->OriginalSnapshotBytes() == SurfaceBytes, "CPU upload has no guest-byte snapshot");
        const auto expectedPixel = watched ? 0x8d8d8d8du : 0x7c7c7c7cu;
        Require(readPixel() == expectedPixel, "CPU upload returned incorrect GPU texels");
        {
            CommandBatch batch(context);
            VkClearColorValue clear{};
            clear.uint32[0] = 0x9e9e9e9eu;
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(batch.Handle(), image->Image(), VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            batch.SubmitAndWait();
        }
        image->MarkDirty();
        image->WriteBack();
        AgcDriver::GuestMemory::ReadCommitted(address, std::as_writable_bytes(std::span(&pixel, 1)));
        Require(pixel == 0x9e9e9e9eu, "CPU write-back lost guest texels");
        std::printf("%s snapshot fixture: direct %zu bytes, CPU fallback %zu bytes\n", watched ? "watched" : "unwatched", expectedBytes, image->OriginalSnapshotBytes());
    }
}
}
#endif

namespace {

// LOCAL EXPERIMENT (APS5_EXPERIMENT_SUBMIT_THREAD, PARALLEL-COMMIT-PLAN A4): the recorder's batches
// sealed and issued by a gate's submitter (a gate of each test's own, with the mode `mode` and own
// pools or not, whatever the switches say). A held submitter (PauseSubmitter) keeps a sealed batch
// unissued until a helper thread releases it.
struct SubmitThreadCase {
    explicit SubmitThreadCase(const Device& device, int mode, bool pools) : gate(Modes(mode, pools)), context(device.GetContext()) {
        gate.Attach(context.device, context.queue, context.Function<PFN_vkQueueSubmit>("vkQueueSubmit"), context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle"));
        context.queueGate = &gate;
    }
    static QueueGate::Modes Modes(int mode, bool pools) {
        QueueGate::Modes modes;
        modes.submitThread = mode;
        modes.submitPools = pools;
        return modes;
    }
    QueueGate gate;
    Context context;
};

// Releases a held submitter after `delay` on its own thread (joined by the destructor).
class DelayedRelease {
public:
    DelayedRelease(QueueGate& gate, std::chrono::milliseconds delay) : thread([&gate, delay, this] {
        std::this_thread::sleep_for(delay);
        released.store(true, std::memory_order_release);
        gate.PauseSubmitter(false);
    }) {}
    ~DelayedRelease() { thread.join(); }
    bool Released() const { return released.load(std::memory_order_acquire); }

private:
    std::atomic<bool> released{false};
    std::thread thread;
};

void recordFill(const Context& context, VkCommandBuffer commands, VkBuffer buffer, std::uint32_t value) {
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, buffer, 0, 16, value);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
}

// A poll before the submit reads unsignaled (Reap keeps the batch, its completion has not run);
// finish (Sync) awaits the submission before its fence wait.
void sealedPollAndFinishTest(const Device& device, int mode, bool pools) {
    SubmitThreadCase test(device, mode, pools);
    Recorder recorder(test.context, true);
    recorder.Activate();
    int ran = 0;
    Buffer target(test.context, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (mode == 1) test.gate.PauseSubmitter(true);
    recordFill(test.context, recorder.Commands(), target.Handle(), 0x11111111u);
    recorder.OnComplete([&] { ++ran; });
    recorder.Submit();
    if (mode == 1) {
        Require(!recorder.Reap() && ran == 0, "a sealed batch the submitter has not issued was reaped");
        Require(recorder.UnsignaledBatches() == 1, "a sealed batch the submitter has not issued polled signaled");
        DelayedRelease release(test.gate, std::chrono::milliseconds(50));
        recorder.Sync();
        Require(release.Released(), "finish returned before the batch was issued");
    } else {
        // =2: Seal returned once the batch was issued.
        Require(test.gate.SubmittedSerial() == recorder.Submissions(), "a synchronous seal returned before its batch was issued");
        recorder.Sync();
    }
    std::uint32_t value = 0;
    std::memcpy(&value, target.Bytes().data(), sizeof(value));
    Require(ran == 1 && recorder.Idle() && value == 0x11111111u, "the issued batch's work and completion did not run");
}

// WaitSerial (the unlocked drain's wait) awaits the submission before the timeline wait.
void waitSerialTest(const Device& device, int mode, bool pools) {
    SubmitThreadCase test(device, mode, pools);
    Recorder recorder(test.context, true);
    recorder.Activate();
    Require(recorder.HasTimeline(), "the submit thread recorder has no timeline");
    if (mode == 1) test.gate.PauseSubmitter(true);
    recorder.OnComplete([] {});
    const auto serial = recorder.SubmitAndEpoch();
    Require(serial != 0, "SubmitAndEpoch sealed nothing");
    if (mode == 1) {
        DelayedRelease release(test.gate, std::chrono::milliseconds(50));
        recorder.WaitSerial(serial);
        Require(release.Released() && test.gate.SubmittedSerial() >= serial, "WaitSerial returned before the batch was issued");
    } else {
        recorder.WaitSerial(serial);
    }
    recorder.FinishUpTo(serial);
    Require(recorder.Idle(), "FinishUpTo left the waited batch in flight");
}

// A device lost by a submit (injected: no Vulkan call) is stored and rethrown at the next seal (=1),
// or thrown by its own seal (=2); its batch never reads signaled.
void submitFailureTest(const Device& device, int mode, bool pools) {
    SubmitThreadCase test(device, mode, pools);
    Recorder recorder(test.context, true);
    recorder.Activate();
    test.gate.InjectSubmitFailure(VK_ERROR_DEVICE_LOST);
    recorder.OnComplete([] {});
    const auto thrownBy = [&] {
        try {
            recorder.Submit();
        } catch (const std::exception& error) {
            return std::string(error.what());
        }
        return std::string();
    };
    if (mode == 1) {
        Require(thrownBy().empty(), "a seal waited for its own submit");
        test.gate.Drain();
        Require(!recorder.Reap(), "the batch whose submit failed was reaped");
        recorder.OnComplete([] {});
    }
    const auto error = thrownBy();
    Require(error.find("vkQueueSubmit recorder") != std::string::npos && error.find("-4") != std::string::npos, "the injected device loss was not rethrown at the next seal: " + error);
    Require(QueueGate::CountsSoFar().submitFailures != 0, "the failed submit was not counted");
    // The recorder's teardown: its Sync throws the stored failure (reported, not escaping), and the
    // submitter is joined.
}

// A recorder torn down with seals in flight: its Sync awaits and completes them, then the submitter
// is joined before the batches' objects go.
void teardownInFlightTest(const Device& device, int mode, bool pools) {
    SubmitThreadCase test(device, mode, pools);
    int ran = 0;
    const auto running = QueueGate::RunningSubmitters();
    {
        // Declared first, so the recorder is destroyed while the release may still be pending.
        std::optional<DelayedRelease> release;
        Recorder recorder(test.context, true);
        recorder.Activate();
        if (mode == 1) test.gate.PauseSubmitter(true);
        for (int batch = 0; batch < 3; ++batch) {
            recorder.OnComplete([&] { ++ran; });
            recorder.Submit();
        }
        if (mode == 1) release.emplace(test.gate, std::chrono::milliseconds(30));
    }
    Require(ran == 3, "a recorder torn down with seals in flight dropped their completions");
    Require(QueueGate::RunningSubmitters() == running, "the recorder's teardown left its submitter running");
    Require(test.gate.SubmittedSerial() == 3, "the torn down recorder's seals were not all issued");
}

// A foreign submit (a command batch) lands after the seals before it: it drains the FIFO first. The
// sealed batch fills 0x11, the foreign one 0x22 after a full barrier: 0x22 must remain.
void foreignAfterSealsTest(const Device& device, int mode, bool pools) {
    SubmitThreadCase test(device, mode, pools);
    Recorder recorder(test.context, true);
    recorder.Activate();
    Buffer target(test.context, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (mode == 1) test.gate.PauseSubmitter(true);
    recordFill(test.context, recorder.Commands(), target.Handle(), 0x11111111u);
    recorder.Submit();
    const auto sealed = recorder.Submissions();
    {
        std::optional<DelayedRelease> release;
        if (mode == 1) release.emplace(test.gate, std::chrono::milliseconds(50));
        CommandBatch batch(test.context);
        recordFill(test.context, batch.Handle(), target.Handle(), 0x22222222u);
        batch.SubmitAndWait();
        Require(test.gate.SubmittedSerial() >= sealed, "a foreign submit was issued before the seals ahead of it");
    }
    recorder.Sync();
    std::uint32_t value = 0;
    std::memcpy(&value, target.Bytes().data(), sizeof(value));
    Require(value == 0x22222222u, "the foreign submit ran before the batch sealed ahead of it");
}

int submitThreadTests(const Device& device, int mode) {
    if (!device.Timeline()) {
        Require(std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr, "the submit thread tests need timeline semaphores");
        std::cout << "skipped, the device has no timeline semaphores\n";
        return 77;
    }
    for (const bool pools : {false, true}) {
        sealedPollAndFinishTest(device, mode, pools);
        waitSerialTest(device, mode, pools);
        submitFailureTest(device, mode, pools);
        teardownInFlightTest(device, mode, pools);
        foreignAfterSealsTest(device, mode, pools);
    }
    const auto counts = QueueGate::CountsSoFar();
    Require(counts.unheld == 0 && counts.overlapped == 0 && counts.misordered == 0 && counts.ordering == 0, "the submit thread tests counted a gate violation");
    std::cout << "submit thread tests passed (APS5_EXPERIMENT_SUBMIT_THREAD=" << mode << " semantics, shared and own pools)\n";
    return 0;
}

// LOCAL EXPERIMENT (GPU-OWNED-PLAN 9.4.1 D4-0, 9.4.2 D4-A): the lease and memo keys' merge tripwire
// (AggregateFieldCount, FieldCount.hpp), checked at compile time: a member added in a padding hole,
// where sizeof does not move, changes the count; a member that is itself an aggregate counts once;
// a type that is not an aggregate never passes.
namespace FieldCountChecks {
struct Holed {
    std::uint64_t address;
    std::uint8_t flag;
};
struct HoleFilled {
    std::uint64_t address;
    std::uint8_t flag;
    bool added = false;
};
struct Nested {
    std::uint64_t address;
    VkExtent2D extent;
    std::array<std::uint32_t, 2> words{};
    TextureTileMode tileMode = TextureTileMode::kLinear;
};
struct Constructed {
    explicit Constructed(std::uint64_t value) : address(value) {}
    std::uint64_t address;
};
static_assert(sizeof(Holed) == sizeof(HoleFilled), "the probe's added member must fill a padding hole");
static_assert(AggregateFieldCount<Holed, 2> && !AggregateFieldCount<HoleFilled, 2> && AggregateFieldCount<HoleFilled, 3>, "the field count missed a member added in a padding hole");
static_assert(AggregateFieldCount<Nested, 4> && !AggregateFieldCount<Nested, 3> && !AggregateFieldCount<Nested, 5>, "the field count did not count each aggregate member once");
static_assert(!AggregateFieldCount<Constructed, 1>, "the field count passed a type that is not an aggregate");
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_MEMOS, GPU-OWNED-PLAN 9.4.2 D4-A): the storage cache's
// surface size and key memos answer what DescribeSurface and SurfaceKey answer; every field of the
// resource is in their keys (each changed alone misses), the key memo keys the context's address,
// device and physical device too, and a call that throws throws again (it is never memoized). At 0
// nothing is counted, at 2 every hit is verified. The mode comes from the environment; no device:
// the context's format support is a stub that stores every format.
VKAPI_ATTR void VKAPI_CALL targetMemoFormatProperties(VkPhysicalDevice, VkFormat, VkFormatProperties* properties) {
    *properties = {};
    properties->optimalTilingFeatures = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
}

void targetMemoTests() {
    const int mode = TargetMemos();
    const auto start = TargetMemoTotals();
    Context context{};
    context.device = reinterpret_cast<VkDevice>(std::uintptr_t{0x7d4a0010});
    context.physical = reinterpret_cast<VkPhysicalDevice>(std::uintptr_t{0x7d4a0020});
    context.formatProperties = targetMemoFormatProperties;
    GuestTextureResource base{};
    base.baseAddress = 0x7d4a00000000ull;
    base.width = 256;
    base.height = 64;
    base.mipCount = 1;
    base.tileMode = TextureTileMode::kLinear;
    base.dimension = TextureDimension::k2D;
    base.format = 56;
    base.dstSelX = 4;
    base.dstSelY = 5;
    base.dstSelZ = 6;
    base.dstSelW = 7;
    const auto expectCounts = [&](const TargetMemoCounts& from, std::uint64_t bytesHits, std::uint64_t bytesMisses, std::uint64_t keyHits, std::uint64_t keyMisses, const std::string& what) {
        const auto now = TargetMemoTotals();
        const std::uint64_t counted = mode != 0 ? 1 : 0;
        const bool expected = now.bytesHits - from.bytesHits == counted * bytesHits && now.bytesMisses - from.bytesMisses == counted * bytesMisses && now.keyHits - from.keyHits == counted * keyHits && now.keyMisses - from.keyMisses == counted * keyMisses && now.verified - from.verified == (mode == 2 ? bytesHits + keyHits : 0);
        Require(expected, "target memos (APS5_EXPERIMENT_TARGET_MEMOS=" + std::to_string(mode) + "): " + what + ": bytes hits " + std::to_string(now.bytesHits - from.bytesHits) + ", misses " + std::to_string(now.bytesMisses - from.bytesMisses) + "; key hits " + std::to_string(now.keyHits - from.keyHits) + ", misses " + std::to_string(now.keyMisses - from.keyMisses) + "; verified " + std::to_string(now.verified - from.verified));
    };
    // Eight surfaces never asked before miss and push every earlier entry out of both memos (eight
    // entries each, replaced in turn), so the next ask of a surface calls DescribeSurface and SurfaceKey.
    std::uint64_t fresh = 0;
    const auto churn = [&] {
        for (int other = 0; other < 8; ++other) {
            auto surface = base;
            surface.baseAddress = base.baseAddress + (++fresh << 24u);
            static_cast<void>(TargetMemoBytes(surface));
            static_cast<void>(TargetMemoKey(context, surface));
        }
    };
    const auto cold = [&](const GuestTextureResource& resource) {
        churn();
        return std::pair{TargetMemoBytes(resource), TargetMemoKey(context, resource)};
    };

    const auto baseAnswer = cold(base);
    Require(baseAnswer.first == DescribeSurface(base).guestBytes, "target memos: the surface bytes of a computed ask differ from DescribeSurface's");
    auto mark = TargetMemoTotals();
    Require(std::pair{TargetMemoBytes(base), TargetMemoKey(context, base)} == baseAnswer, "target memos: a second ask of a surface answered differently");
    expectCounts(mark, 1, 0, 1, 0, "a second ask of a surface");

    using Change = void (*)(GuestTextureResource&);
    const std::array<std::pair<const char*, Change>, 24> fields{{
        {"baseAddress", [](GuestTextureResource& resource) { resource.baseAddress += 0x10000; }},
        {"width", [](GuestTextureResource& resource) { resource.width += 64; }},
        {"height", [](GuestTextureResource& resource) { resource.height += 64; }},
        {"depthOrLastArray", [](GuestTextureResource& resource) { resource.depthOrLastArray += 1; }},
        {"baseArray", [](GuestTextureResource& resource) { resource.baseArray += 1; }},
        {"mipCount", [](GuestTextureResource& resource) { resource.mipCount = 2; }},
        {"baseLevel", [](GuestTextureResource& resource) { resource.baseLevel = 1; }},
        {"tileMode", [](GuestTextureResource& resource) { resource.tileMode = TextureTileMode::kR64KBX; }},
        {"dimension", [](GuestTextureResource& resource) { resource.dimension = TextureDimension::k2DArray; }},
        {"format", [](GuestTextureResource& resource) { resource.format = 22; }},
        {"dstSelX", [](GuestTextureResource& resource) { resource.dstSelX = 0; }},
        {"dstSelY", [](GuestTextureResource& resource) { resource.dstSelY = 0; }},
        {"dstSelZ", [](GuestTextureResource& resource) { resource.dstSelZ = 0; }},
        {"dstSelW", [](GuestTextureResource& resource) { resource.dstSelW = 0; }},
        {"pipeBankXor", [](GuestTextureResource& resource) { resource.pipeBankXor = 0x100; }},
        {"bcSwizzle", [](GuestTextureResource& resource) { resource.bcSwizzle = 1; }},
        {"lastLevel", [](GuestTextureResource& resource) { resource.lastLevel = 1; }},
        {"dccAddress", [](GuestTextureResource& resource) { resource.dccAddress = resource.baseAddress + 0x100000; }},
        {"dccAlphaOnMsb", [](GuestTextureResource& resource) { resource.dccAlphaOnMsb = true; }},
        {"dccPipeAligned", [](GuestTextureResource& resource) { resource.dccPipeAligned = true; }},
        {"iterate256", [](GuestTextureResource& resource) { resource.iterate256 = true; }},
        {"minLod", [](GuestTextureResource& resource) { resource.minLod = 1; }},
        {"allocatedMipCount", [](GuestTextureResource& resource) { resource.allocatedMipCount = 1; }},
        {"samples", [](GuestTextureResource& resource) { resource.samples = 2; }},
    }};
    static_assert(AggregateFieldCount<GuestTextureResource, std::tuple_size_v<decltype(fields)>>, "a GuestTextureResource field was added: change it alone here too");
    for (const auto& [name, change] : fields) {
        auto changed = base;
        change(changed);
        const auto want = cold(changed);
        Require(want.first == DescribeSurface(changed).guestBytes, std::string("target memos: the surface bytes of a computed ask with ") + name + " changed differ from DescribeSurface's");
        // Only the unchanged surface (and the other surfaces) in the memos: the changed one must miss.
        churn();
        static_cast<void>(TargetMemoBytes(base));
        static_cast<void>(TargetMemoKey(context, base));
        mark = TargetMemoTotals();
        Require(std::pair{TargetMemoBytes(changed), TargetMemoKey(context, changed)} == want, std::string("target memos: a surface with ") + name + " changed answered as the memoized surface it differs from");
        expectCounts(mark, 0, 1, 0, 1, std::string("a surface with ") + name + " changed after the unchanged one");
    }

    // The key memo keys the context: another context object, then that object with another device,
    // then with another physical device, each miss; restored, it hits its first entry.
    churn();
    static_cast<void>(TargetMemoKey(context, base));
    auto other = context;
    mark = TargetMemoTotals();
    Require(TargetMemoKey(other, base) == baseAnswer.second, "target memos: another context object answered another key");
    expectCounts(mark, 0, 0, 0, 1, "the same surface through another context object");
    other.device = reinterpret_cast<VkDevice>(std::uintptr_t{0x7d4a0030});
    mark = TargetMemoTotals();
    Require(TargetMemoKey(other, base) == baseAnswer.second, "target memos: another device answered another key");
    expectCounts(mark, 0, 0, 0, 1, "the same surface and context object with another device");
    other.device = context.device;
    other.physical = reinterpret_cast<VkPhysicalDevice>(std::uintptr_t{0x7d4a0040});
    mark = TargetMemoTotals();
    Require(TargetMemoKey(other, base) == baseAnswer.second, "target memos: another physical device with the same format support answered another key");
    expectCounts(mark, 0, 0, 0, 1, "the same surface and context object with another physical device");
    other.physical = context.physical;
    mark = TargetMemoTotals();
    Require(TargetMemoKey(other, base) == baseAnswer.second, "target memos: the restored context object answered another key");
    expectCounts(mark, 0, 0, 1, 0, "the same surface and context object restored");

    // No guest format 3: DescribeSurface and SurfaceKey throw on every ask, and nothing is kept.
    auto broken = base;
    broken.format = 3;
    for (int ask = 0; ask < 2; ++ask) {
        mark = TargetMemoTotals();
        bool bytesThrew = false;
        bool keyThrew = false;
        try {
            static_cast<void>(TargetMemoBytes(broken));
        } catch (const std::exception&) {
            bytesThrew = true;
        }
        try {
            static_cast<void>(TargetMemoKey(context, broken));
        } catch (const std::exception&) {
            keyThrew = true;
        }
        Require(bytesThrew && keyThrew, "target memos: a surface of an unknown format did not throw on ask " + std::to_string(ask + 1));
        expectCounts(mark, 0, 1, 0, 1, "a surface of an unknown format, ask " + std::to_string(ask + 1));
    }
    Require(mode != 0 || TargetMemoTotals() == start, "target memos: the switch is off but the memos counted");
    std::cout << "target memo tests passed (APS5_EXPERIMENT_TARGET_MEMOS=" << mode << ")\n";
}

// LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0): the DepthSurfaceView lease counter's
// classification and counts, without a device. Each condition moved alone misses by its own reason;
// with every condition from one on moved, the call misses by that one (the order of DepthLease); a
// writeGeneration move is no reason but is counted apart; a covering cleared HTILE left at an end
// misses as a cleared HTILE set, in that place of the order; a range with no proof is named before a
// stamp; the work a call still did is counted for would-hits only; and the [proofs] part prints the
// counts since its last call, with the counter's time by the [draws] phase that carries it.
void depthLeaseClassifyTests() {
    int surfaceObject = 0;
    int otherSurface = 0;
    DepthLeaseMark previous;
    previous.surface = &surfaceObject;
    previous.memo = {5, 6, 7};
    previous.surfaces = 10;
    previous.unretired = 11;
    previous.planeBinds = 12;
    previous.clearedHtiles = 13;
    previous.mirrorEpoch = 14;
    previous.finished = 15;
    previous.writeNotes = 16;
    previous.writeGeneration = 17;
    const auto expect = [](const DepthLeaseVerdict& verdict, DepthLease outcome, bool moved, const std::string& what) {
        Require(verdict.outcome == outcome && verdict.writeGenerationMoved == moved, "D5-0 classification: " + what + ": " + DepthLeaseName(verdict.outcome) + (verdict.writeGenerationMoved ? " with" : " without") + " a writeGeneration move, where " + DepthLeaseName(outcome) + (moved ? " with" : " without") + " one was expected");
    };
    expect(ClassifyDepthLease(previous, previous, DepthLease::WouldHit, false), DepthLease::WouldHit, false, "nothing moved");
    auto moved = previous;
    moved.writeNotes = 99;
    expect(ClassifyDepthLease(previous, moved, DepthLease::WouldHit, false), DepthLease::WouldHit, false, "a newer write note (only the HTILE test reads it)");
    moved = previous;
    moved.writeGeneration = 18;
    expect(ClassifyDepthLease(previous, moved, DepthLease::WouldHit, false), DepthLease::WouldHit, true, "writeGeneration moved alone");
    const auto single = [&](auto move, DepthLease outcome, const char* what) {
        auto now = previous;
        move(now);
        expect(ClassifyDepthLease(previous, now, DepthLease::WouldHit, false), outcome, false, what);
    };
    single([&](DepthLeaseMark& now) { now.surface = &otherSurface; }, DepthLease::Identity, "another surface");
    single([](DepthLeaseMark& now) { now.surface = nullptr; }, DepthLease::Identity, "a surface the call created");
    single([](DepthLeaseMark& now) { ++now.memo.epoch; }, DepthLease::Epoch, "an epoch bump");
    single([](DepthLeaseMark& now) { ++now.memo.mapping; }, DepthLease::Allocations, "an allocation change");
    single([](DepthLeaseMark& now) { ++now.memo.unwatched; }, DepthLease::Unwatch, "an unwatch");
    single([](DepthLeaseMark& now) { ++now.surfaces; }, DepthLease::Surfaces, "a surface created or cleared");
    single([](DepthLeaseMark& now) { ++now.unretired; }, DepthLease::Unretired, "a surface un-retired");
    single([](DepthLeaseMark& now) { ++now.planeBinds; }, DepthLease::PlaneBinds, "a plane bound");
    single([](DepthLeaseMark& now) { ++now.clearedHtiles; }, DepthLease::ClearedHtile, "a cleared HTILE set");
    single([](DepthLeaseMark& now) { ++now.mirrorEpoch; }, DepthLease::MirrorEpoch, "a mirror created");
    single([](DepthLeaseMark& now) { ++now.finished; }, DepthLease::Reaped, "a batch finished");
    for (const auto stamp : {DepthLease::Unproven, DepthLease::StampedDepth, DepthLease::StampedStencil, DepthLease::StampedHtile, DepthLease::HostWrites}) expect(ClassifyDepthLease(previous, previous, stamp, false), stamp, false, std::string("a stamp test failed (") + DepthLeaseName(stamp) + ")");
    expect(ClassifyDepthLease(previous, previous, DepthLease::WouldHit, true), DepthLease::HtileNote, false, "a write over the HTILE noted");
    // A covering cleared HTILE left at the end (the previous mark's flag; a start's is never read).
    auto left = previous;
    left.clearedHtileLeft = true;
    expect(ClassifyDepthLease(left, previous, DepthLease::WouldHit, false), DepthLease::ClearedHtile, false, "a covering cleared HTILE left at the end");
    moved = previous;
    ++moved.mirrorEpoch;
    ++moved.finished;
    expect(ClassifyDepthLease(left, moved, DepthLease::StampedDepth, true), DepthLease::ClearedHtile, false, "a covering cleared HTILE left, with every later condition moved");
    moved = previous;
    ++moved.planeBinds;
    expect(ClassifyDepthLease(left, moved, DepthLease::WouldHit, false), DepthLease::PlaneBinds, false, "a plane bound, with a covering cleared HTILE left");
    moved = previous;
    moved.clearedHtileLeft = true;
    expect(ClassifyDepthLease(previous, moved, DepthLease::WouldHit, false), DepthLease::WouldHit, false, "a start's cleared HTILE flag");
    auto all = previous;
    all.surface = nullptr;
    ++all.memo.epoch;
    ++all.memo.mapping;
    ++all.memo.unwatched;
    ++all.surfaces;
    ++all.unretired;
    ++all.planeBinds;
    ++all.clearedHtiles;
    ++all.mirrorEpoch;
    ++all.finished;
    ++all.writeGeneration;
    // The caller names a range with no proof before a stamp, and a stamp before pending host writes, so
    // `stamped` carries Unproven first, then a stamp, then HostWrites.
    auto stamped = DepthLease::Unproven;
    bool written = true;
    constexpr std::array order{DepthLease::Identity, DepthLease::Epoch, DepthLease::Allocations, DepthLease::Unwatch, DepthLease::Surfaces, DepthLease::Unretired, DepthLease::PlaneBinds, DepthLease::ClearedHtile, DepthLease::MirrorEpoch, DepthLease::Reaped, DepthLease::Unproven, DepthLease::StampedDepth, DepthLease::HostWrites, DepthLease::HtileNote, DepthLease::WouldHit};
    for (std::size_t step = 0; step < order.size(); ++step) {
        expect(ClassifyDepthLease(previous, all, stamped, written), order[step], true, std::string("every condition from ") + DepthLeaseName(order[step]) + " on moved");
        switch (step) {
        case 0: all.surface = previous.surface; break;
        case 1: all.memo.epoch = previous.memo.epoch; break;
        case 2: all.memo.mapping = previous.memo.mapping; break;
        case 3: all.memo.unwatched = previous.memo.unwatched; break;
        case 4: all.surfaces = previous.surfaces; break;
        case 5: all.unretired = previous.unretired; break;
        case 6: all.planeBinds = previous.planeBinds; break;
        case 7: all.clearedHtiles = previous.clearedHtiles; break;
        case 8: all.mirrorEpoch = previous.mirrorEpoch; break;
        case 9: all.finished = previous.finished; break;
        case 10: stamped = DepthLease::StampedDepth; break;
        case 11: stamped = DepthLease::HostWrites; break;
        case 12: stamped = DepthLease::WouldHit; break;
        case 13: written = false; break;
        default: break;
        }
    }
    static_cast<void>(DepthLeaseReport());
    const auto before = DepthLeaseTotals();
    CountDepthLease({DepthLease::WouldHit, true}, DepthLeaseHtileStoreBack | DepthLeaseMirrorSync);
    CountDepthLease({DepthLease::WouldHit, false}, 0);
    CountDepthLease({DepthLease::PlaneBinds, true}, DepthLeaseClearedHtileTaken);
    CountDepthLease({DepthLease::NoEntry, false}, 0);
    const auto after = DepthLeaseTotals();
    const auto delta = [&](DepthLease lease) { return after.outcomes[static_cast<std::size_t>(lease)] - before.outcomes[static_cast<std::size_t>(lease)]; };
    std::uint64_t counted = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(DepthLease::Count); ++i) counted += after.outcomes[i] - before.outcomes[i];
    Require(delta(DepthLease::WouldHit) == 2 && delta(DepthLease::PlaneBinds) == 1 && delta(DepthLease::NoEntry) == 1 && counted == 4, "D5-0 counts: the outcomes were not counted once each");
    Require(after.hitsWriteGeneration - before.hitsWriteGeneration == 1 && after.missesWriteGeneration - before.missesWriteGeneration == 1, "D5-0 counts: the writeGeneration moves were not counted apart for the would-hits and the misses");
    Require(after.hitWork[0] - before.hitWork[0] == 1 && after.hitWork[1] == before.hitWork[1] && after.hitWork[2] - before.hitWork[2] == 1, "D5-0 counts: the work of the would-hits' calls was not counted as such (or a miss's was)");
    Require(after.calls == before.calls && after.ineligible[DepthLeaseEligible] == before.ineligible[DepthLeaseEligible], "D5-0 counts: counting an outcome counted a call");
    const auto report = DepthLeaseReport();
    for (const char* part : {"would-hit 2 (writeGeneration moved in 1; today's call still did work: HTILE stored back 1, cleared HTILE taken 0, mirror synced 1)", "misses by reason: no entry 1,", ", plane binds 1,", ", reaped 0, no proof (unwatched or generation 0) 0, stamped: depth 0,", ", HTILE 0, host writes pending 0, HTILE note 0", "writeGeneration moved in 1 of the classified misses", "ms in the [draws] prepare phase (Draw) and ", " ms in readTarget (DrawWithRecipe)"}) {
        Require(report.find(part) != std::string::npos, std::string("D5-0 report: '") + part + "' is not in: " + report);
    }
    Require(DepthLeaseReport().find("would-hit 0 (") != std::string::npos, "D5-0 report: a second report did not start from the first one's counts");
    std::cout << "depth lease classification tests passed\n";
}

// LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0): the lease counter over real DepthSurfaceView
// calls on a watched block, and the recorder reads it uses (WrittenAfterNote against NewestWriteNote,
// FinishedSerial against the batches finished). Without APS5_PROFILE_DRAW nothing is counted; with it
// but no collect memo key (no APS5_DEBUG_COVERAGE=1) every call is ineligible for that reason; with
// both, a second call of a target would hit and each condition moved between two calls misses by its
// reason, including a second covering cleared HTILE set before a target's first call (its second call
// takes it) and a depth plane unwatched (unproven, not stamped, after the unwatch's own miss); the
// counter's time in a DepthLeaseRecipeScope goes to DrawWithRecipe's sum only. A step across which
// the memo key moved by itself (an unwatch or an allocation change made elsewhere) is tried again.
// False when the device watches no memory.
bool depthLeaseCounterTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t block = 65536;
    constexpr std::size_t bytes = 10 * block;
    void* memory = AllocateWatched(bytes, block);
    if (memory == nullptr) {
        std::cout << "no write watching: the depth lease counter not tested\n";
        return false;
    }
    struct Release {
        void* memory;
        ~Release() { ReleaseWatched(memory, bytes); }
    } release{memory};
    const auto& context = device.GetContext();
    // ClearDepthSurfaces destroys the surfaces' images at once (the device teardown's): the batch
    // that recorded work over them must finish first, on a failed expectation too.
    struct Clear {
        Recorder& recorder;
        VkDevice device;
        ~Clear() {
            try {
                recorder.Sync();
            } catch (...) {
            }
            ClearDepthSurfaces(device);
        }
    } clear{recorder, context.device};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    constexpr VkExtent2D extent{64, 64};
    const auto htileBytes = static_cast<std::size_t>(HtileMetadataBytes(extent));
    auto* words = static_cast<std::uint8_t*>(memory);
    std::memset(words, 0, bytes);
    // HTILE words with no clear code (HtileFillClears reads the low nibble).
    std::memset(words + 3 * block, 0xff, htileBytes);
    std::memset(words + 4 * block, 0xff, htileBytes);
    std::memset(words + 9 * block + 4096, 0xff, htileBytes);
    DepthTarget a{base, 0, extent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    a.htileAddress = base + 3 * block;
    const DepthTarget b{base + block, 0, extent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    DepthTarget c{base + 2 * block, 0, extent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    c.htileAddress = base + 4 * block;
    const auto unrelated = base + 5 * block;
    // A fourth target whose depth plane is unwatched, and a fifth whose HTILE two cleared HTILEs cover.
    const DepthTarget d{base + 7 * block, 0, extent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    DepthTarget e{base + 8 * block, 0, extent, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    e.htileAddress = base + 9 * block + 4096;
    const auto settle = [&] {
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    };

    settle();
    const auto finished = recorder.FinishedSerial();
    const auto notes = recorder.WriteNotes();
    Require(!recorder.WrittenAfterNote(unrelated, block, notes), "D5-0 recorder: a write newer than the newest note");
    recorder.NotePendingWrite(unrelated, 256);
    Require(recorder.WriteNotes() == notes + 1 && recorder.NewestWriteNote(unrelated, block) == notes + 1, "D5-0 recorder: a noted write did not take the next note");
    Require(recorder.WrittenAfterNote(unrelated, block, notes) && !recorder.WrittenAfterNote(unrelated, block, notes + 1) && !recorder.WrittenAfterNote(unrelated + 256, 256, notes), "D5-0 recorder: WrittenAfterNote does not answer NewestWriteNote(range) > note for a noted write");
    recorder.NotePendingWrite(base + 6 * block, 256);
    Require(recorder.WrittenAfterNote(unrelated, block, notes) && !recorder.WrittenAfterNote(unrelated, block, notes + 1) && recorder.WrittenAfterNote(base + 6 * block, 256, notes + 1), "D5-0 recorder: a newer write elsewhere changed WrittenAfterNote's answer over the first one");
    Require(recorder.FinishedSerial() == finished, "D5-0 recorder: noting writes moved the finished serial");
    recorder.Submit();
    Require(recorder.FinishedSerial() == finished && recorder.WrittenAfterNote(unrelated, block, notes), "D5-0 recorder: a submitted batch read as finished, or its write as gone");
    device.WaitQueue();
    recorder.Sync();
    Require(recorder.FinishedSerial() > finished && !recorder.WrittenAfterNote(unrelated, block, notes) && recorder.NewestWriteNote(unrelated, block) == 0, "D5-0 recorder: a finished batch did not move the finished serial, or its write was still found");

    const bool profiling = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    BumpCollectEpoch();
    const bool keyed = CurrentCollectMemoKey().Valid();
    const auto start = DepthLeaseTotals();
    const auto view = [&](const DepthTarget& target) { Require(DepthSurfaceView(context, target) != VK_NULL_HANDLE, "D5-0: the depth surface has no view"); };
    const auto outcomes = [](const DepthLeaseCounts& counts) {
        std::uint64_t total = 0;
        for (const auto count : counts.outcomes) total += count;
        return total;
    };
    if (!profiling || !keyed) {
        view(a);
        view(a);
        view(b);
        view(a);
        const auto now = DepthLeaseTotals();
        if (!profiling) Require(now.calls == start.calls && outcomes(now) == outcomes(start) && now.counterNs == start.counterNs && now.recipeCounterNs == start.recipeCounterNs, "D5-0: calls were counted without APS5_PROFILE_DRAW");
        else Require(now.calls == start.calls + 4 && now.ineligible[DepthLeaseNoMemoKey] == start.ineligible[DepthLeaseNoMemoKey] + 4 && outcomes(now) == outcomes(start), "D5-0: calls without a collect memo key were not ineligible for that reason");
        settle();
        std::cout << "depth lease counter tests passed (" << (profiling ? "no collect memo key: every call ineligible" : "not profiling: nothing counted") << ")\n";
        return true;
    }
    // One call of `target` after `action`: the outcome it counted. A step across which the memo key
    // moved by itself is tried again (up to three times), after a call that renews the entry.
    const auto outcomeOf = [&](const DepthTarget& target, auto&& action) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            const auto key = CurrentCollectMemoKey();
            const auto before = DepthLeaseTotals();
            action();
            view(target);
            const auto after = DepthLeaseTotals();
            Require(after.calls == before.calls + 1 && after.ineligible[DepthLeaseEligible] == before.ineligible[DepthLeaseEligible] + 1, "D5-0: a call of a single-sample target under the GPU lock with a memo key was not counted eligible");
            auto outcome = DepthLease::Count;
            for (std::size_t i = 0; i < static_cast<std::size_t>(DepthLease::Count); ++i) {
                if (after.outcomes[i] == before.outcomes[i]) continue;
                Require(outcome == DepthLease::Count && after.outcomes[i] == before.outcomes[i] + 1, "D5-0: a call counted more than one outcome");
                outcome = static_cast<DepthLease>(i);
            }
            Require(outcome != DepthLease::Count, "D5-0: an eligible call counted no outcome");
            if (CurrentCollectMemoKey() == key) return outcome;
            view(target);
        }
        throw std::runtime_error("D5-0: the collect memo key kept moving by itself");
    };
    const auto none = [] {};
    const auto expect = [&](const DepthTarget& target, auto&& action, DepthLease expected, const char* what) {
        const auto outcome = outcomeOf(target, action);
        Require(outcome == expected, std::string("D5-0: ") + what + ": " + DepthLeaseName(outcome) + " where " + DepthLeaseName(expected) + " was expected");
    };
    // The first call misses (no entry, or an entry an earlier test's surface at this address left).
    Require(outcomeOf(a, none) != DepthLease::WouldHit, "D5-0: the first call of a target would hit");
    expect(a, none, DepthLease::WouldHit, "a second call");
    expect(a, [&] { MarkWritten(a.address + 4096, 64); }, DepthLease::StampedDepth, "a driver stamp over the depth plane");
    expect(a, none, DepthLease::WouldHit, "the call after the depth plane stamp");
    expect(a, [&] { MarkWritten(a.htileAddress, 4); }, DepthLease::StampedHtile, "a driver stamp over the HTILE");
    expect(a, none, DepthLease::WouldHit, "the call after the HTILE stamp");
    expect(a, [&] { recorder.NotePendingWordFill(a.htileAddress, htileBytes, 0xffffffffu); }, DepthLease::HtileNote, "a GPU fill over the HTILE noted");
    expect(a, none, DepthLease::WouldHit, "the call after the HTILE fill (still pending)");
    expect(a, settle, DepthLease::Reaped, "the batch holding the HTILE fill finished");
    expect(a, none, DepthLease::WouldHit, "the call after the reap");
    {
        // Host writes that ended unstamped (queued for the next tracker lock holder, as under
        // APS5_EXPERIMENT_HOST_WRITE_STAMPS=1; the hook installed by hand): the stamp test takes its
        // lock without the drain, so with every range unstamped a call misses as pending host writes
        // when one lies over a range's pages, the depth plane's too (not stamped first to name the
        // plane), and would hit when none does: elsewhere, or in the HTILE's block past its pages (the
        // drain stamps a write's own pages only). The call's own collects then stamp the queue, and
        // the next call would hit.
        struct HostWriteHook {
            HostWriteHook() { GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(&note); }
            ~HostWriteHook() { GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(nullptr); }
            HostWriteHook(const HostWriteHook&) = delete;
            HostWriteHook& operator=(const HostWriteHook&) = delete;
            static void note(std::uintptr_t address, std::size_t bytes) { AgcDriver::GuestMemory::NoteHostWrite(address, bytes); }
        } hook;
        const auto hostWrite = [](std::uint64_t address) {
            auto* bytes = reinterpret_cast<std::uint8_t*>(address);
            const GuestArena::HostWrite write(bytes, 16);
            Require(write.Open(), "D5-0: the host write did not open");
            std::memset(bytes, 0x66, 16);
        };
        expect(a, [&] { hostWrite(unrelated + 4096); }, DepthLease::WouldHit, "a host write elsewhere ended unstamped");
        expect(a, none, DepthLease::WouldHit, "the call after the host write elsewhere was stamped");
        Require(htileBytes + 2 * 4096 <= block, "D5-0 fixture: the HTILE leaves no page of its block free");
        const auto pastHtile = (a.htileAddress + htileBytes + 4095) / 4096 * 4096;
        expect(a, [&] { hostWrite(pastHtile); }, DepthLease::WouldHit, "a host write in the HTILE's block past its pages ended unstamped");
        expect(a, none, DepthLease::WouldHit, "the call after the host write past the HTILE was stamped");
        expect(a, [&] { hostWrite(a.address + 4096); }, DepthLease::HostWrites, "a host write over the depth plane ended unstamped");
        expect(a, none, DepthLease::WouldHit, "the call after the host write over the depth plane was stamped");
    }
    {
        const auto before = DepthLeaseTotals();
        expect(a, [&] { recorder.NotePendingWrite(unrelated, 256); }, DepthLease::WouldHit, "a write noted outside the target");
        Require(DepthLeaseTotals().hitsWriteGeneration > before.hitsWriteGeneration, "D5-0: a would-hit across a writeGeneration move was not counted apart");
    }
    Require(outcomeOf(b, none) != DepthLease::WouldHit, "D5-0: the first call of a second target would hit");
    expect(a, none, DepthLease::Surfaces, "another target's surface created");
    expect(b, none, DepthLease::PlaneBinds, "the first target bound its planes");
    expect(a, none, DepthLease::PlaneBinds, "the second target bound its planes");
    expect(a, none, DepthLease::WouldHit, "a call after the other target's");
    expect(a, [&] { RetireDepthSurfaces(context.device, a.address, block); }, DepthLease::Unretired, "the surface retired, then un-retired by the call");
    expect(a, none, DepthLease::WouldHit, "the call after the un-retire");
    expect(a, [&] { NoteHtileDepthClear(c.htileAddress, htileBytes); }, DepthLease::ClearedHtile, "a cleared HTILE set for no resident surface");
    // The third target takes that cleared HTILE (it lies over its HTILE), so nothing stays behind.
    Require(outcomeOf(c, none) != DepthLease::WouldHit, "D5-0: the first call of a third target would hit");
    expect(a, none, DepthLease::Surfaces, "the third target's surface created");
    expect(a, none, DepthLease::WouldHit, "a call after the third target's");
    {
        const auto before = DepthLeaseTotals();
        BumpCollectEpoch();
        view(a);
        const auto after = DepthLeaseTotals();
        Require(after.outcomes[static_cast<std::size_t>(DepthLease::Epoch)] == before.outcomes[static_cast<std::size_t>(DepthLease::Epoch)] + 1 && outcomes(after) == outcomes(before) + 1, "D5-0: an epoch bump did not miss as one");
    }
    expect(a, none, DepthLease::WouldHit, "the call after the epoch bump");
    {
        // DrawWithRecipe's call runs in a DepthLeaseRecipeScope: the counter's time there is not Draw's,
        // and after the scope it is Draw's again.
        auto before = DepthLeaseTotals();
        {
            const DepthLeaseRecipeScope recipeScope(true);
            view(a);
        }
        auto after = DepthLeaseTotals();
        Require(after.counterNs == before.counterNs, "D5-0: the counter's time in a DrawWithRecipe call was summed as Draw's");
        before = after;
        view(a);
        after = DepthLeaseTotals();
        Require(after.recipeCounterNs == before.recipeCounterNs, "D5-0: the counter's time after the recipe scope ended was summed as DrawWithRecipe's");
    }
    // A depth plane the tracker stops watching: the unwatch moves the memo key (its own miss), and
    // from then on the plane has no proof generation, which is no stamp.
    Require(outcomeOf(d, none) != DepthLease::WouldHit, "D5-0: the first call of a fourth target would hit");
    expect(d, none, DepthLease::WouldHit, "a second call of the fourth target");
    {
        const auto before = DepthLeaseTotals();
        Unwatch(d.address, block);
        Require(!Watched(d.address, block), "D5-0: the fourth target's depth plane is still watched");
        view(d);
        const auto after = DepthLeaseTotals();
        Require(after.outcomes[static_cast<std::size_t>(DepthLease::Unwatch)] == before.outcomes[static_cast<std::size_t>(DepthLease::Unwatch)] + 1 && outcomes(after) == outcomes(before) + 1, "D5-0: an unwatch did not miss as one");
    }
    expect(d, none, DepthLease::Unproven, "the call after the depth plane was unwatched");
    expect(d, none, DepthLease::Unproven, "a later call of the unwatched depth plane");
    // Two cleared HTILEs over the fifth target's HTILE, set before its surface exists (no resident
    // surface's HTILE lies in them): its first call takes one, its second the other with no insert in
    // between, so that call misses; the third finds none left.
    NoteHtileDepthClear(e.htileAddress - 4096, htileBytes + 4096);
    NoteHtileDepthClear(e.htileAddress, htileBytes);
    Require(outcomeOf(e, none) != DepthLease::WouldHit, "D5-0: the first call of a fifth target would hit");
    expect(e, none, DepthLease::ClearedHtile, "the second covering cleared HTILE left by the first call");
    expect(e, none, DepthLease::WouldHit, "the call after both cleared HTILEs were taken");
    expect(a, [&] {
        settle();
        ClearDepthSurfaces(context.device);
    }, DepthLease::Identity, "the surface cleared and created again");
    const auto end = DepthLeaseTotals();
    Require(end.hitWork[0] == start.hitWork[0] && end.hitWork[1] == start.hitWork[1] && end.hitWork[2] == start.hitWork[2], "D5-0: a would-hit's call still did work a lease would skip");
    settle();
    std::cout << "depth lease counter tests passed\n";
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.12 (b)): =2's classification of a
// verified DepthSurfaceView call (CheckDepthViewPrediction), without a device. Every part failing with
// no evidence aborts; the same with evidence read at its own point is raced only when that evidence can
// change the part (the store-back's per predicted route and today's verdict, never by a pending verdict
// on a not-pending prediction, never by the clock alone, the HTILE stamped excusing a failed read proof;
// the identity's by the surfaces serials, retired and a cleared HTILE entry; a sync's clock move only
// with the plane stamped); evidence read only at another part's point never excuses it; refresh values
// equal verify, above race and below abort, with the read proof checked per route (pending: the
// memo-hit generation; not pending: the prediction's; none for no HTILE or not accessible) and the
// stencil memory exactly 0 without a stencil plane. The names and the [proofs] part (counts since its
// last call) follow.
void depthViewLeaseClassifyTests() {
    DepthViewFacts agreed;
    agreed.route = DepthViewRoute::Pending;
    agreed.stencil = true;
    agreed.h = 0;
    agreed.current0 = 50;
    agreed.memoCurrent0 = 50;
    agreed.verdict = HtileVerdictKind::PendingCovered;
    agreed.htileReadFound = true;
    agreed.htileRead = 50;
    agreed.unitsLow = 50;
    agreed.unitsHigh = 50;
    agreed.depthMemory = 50;
    agreed.stencilMemory = 50;
    const auto expect = [](const DepthViewFacts& facts, DepthViewPart part, bool raced, const std::string& what) {
        const auto check = CheckDepthViewPrediction(facts);
        Require(check.part == part && check.raced == raced, "D5 classification: " + what + ": " + DepthViewPartName(check.part) + (check.raced ? " raced" : " not raced") + " (" + check.reason + "), where " + DepthViewPartName(part) + (raced ? " raced" : " not raced") + " was expected");
    };
    expect(agreed, DepthViewPart::None, false, "everything agreed (pending)");
    // Each route agreed with today's matching verdict.
    {
        auto notPending = agreed;
        notPending.route = DepthViewRoute::NotPending;
        notPending.verdict = HtileVerdictKind::NotPendingProved;
        notPending.h = 40;
        notPending.htileRead = 40;
        expect(notPending, DepthViewPart::None, false, "everything agreed (not pending: the read proof is the prediction's)");
        notPending.htileRead = 50;
        expect(notPending, DepthViewPart::Refresh, true, "a not-pending read proof above the prediction's");
        notPending.htileRead = 39;
        expect(notPending, DepthViewPart::Refresh, false, "a not-pending read proof below the prediction's");
        notPending.htileReadFound = false;
        expect(notPending, DepthViewPart::Refresh, false, "a not-pending read proof missing");
        for (const auto route : {DepthViewRoute::NoHtile, DepthViewRoute::NotAccessible}) {
            auto facts = agreed;
            facts.route = route;
            facts.verdict = route == DepthViewRoute::NoHtile ? HtileVerdictKind::NoHtile : HtileVerdictKind::NotAccessible;
            facts.htileReadFound = false;
            facts.htileRead = 0;
            expect(facts, DepthViewPart::None, false, std::string("everything agreed (") + DepthViewRouteName(route) + ", no read proof checked)");
        }
        auto noStencil = agreed;
        noStencil.stencil = false;
        noStencil.stencilMemory = 0;
        expect(noStencil, DepthViewPart::None, false, "everything agreed without a stencil plane");
        noStencil.stencilMemory = 50;
        expect(noStencil, DepthViewPart::Refresh, false, "a stencil memory without a stencil plane");
    }
    // The store-back part.
    const auto storeBackEvidence = [](DepthViewFacts facts, int which) {
        switch (which) {
            case 0: facts.storeBack.memoMoved = true; break;
            case 1: facts.storeBack.accessibleDiffers = true; break;
            case 2: facts.storeBack.foundDiffers = true; break;
            case 3: facts.storeBack.collectedAbove = true; break;
            default: facts.storeBack.htileStamped = true; break;
        }
        return facts;
    };
    const auto pointEvidence = [](DepthViewEvidence evidence, int which) {
        switch (which) {
            case 0: evidence.memoMoved = true; break;
            case 1: evidence.surfacesMoved = true; break;
            case 2: evidence.retired = true; break;
            case 3: evidence.oldestMoved = true; break;
            case 4: evidence.clearedHtile = true; break;
            default: evidence.clockMoved = true; break;
        }
        return evidence;
    };
    {
        auto storedBack = agreed;
        storedBack.verdict = HtileVerdictKind::StoredBack;
        storedBack.htileCleared = 1;
        expect(storedBack, DepthViewPart::StoreBack, false, "an HTILE clear stored back, no evidence");
        // Only the evidence that can change the predicted route's verdict excuses it (the r73 review;
        // memo key 0, Accessible 1, read proof 2, collect above 3, HTILE stamped 4): not pending, after
        // the read proof's lookup, the memo key, the read proof and the HTILE stamped; pending and not
        // accessible, Accessible only; no HTILE, none. The collect above (a memo hit's answer, the
        // tracker generation, which a stamp anywhere moves) never.
        for (const auto route : {DepthViewRoute::Pending, DepthViewRoute::NotPending, DepthViewRoute::NotAccessible, DepthViewRoute::NoHtile}) {
            auto facts = storedBack;
            facts.route = route;
            if (route == DepthViewRoute::NotPending) {
                facts.h = 40;
                facts.htileRead = 40;
            } else if (route != DepthViewRoute::Pending) {
                facts.htileReadFound = false;
                facts.htileRead = 0;
            }
            expect(facts, DepthViewPart::StoreBack, false, std::string("an HTILE clear stored back (") + DepthViewRouteName(route) + "), no evidence");
            for (int which = 0; which < 5; ++which) {
                const bool excuses = route == DepthViewRoute::NotPending ? which == 0 || which == 2 || which == 4 : which == 1 && route != DepthViewRoute::NoHtile;
                expect(storeBackEvidence(facts, which), DepthViewPart::StoreBack, excuses, std::string("an HTILE clear stored back (") + DepthViewRouteName(route) + "), with store-back evidence " + std::to_string(which));
            }
            auto all = facts;
            all.storeBack.memoMoved = true;
            all.storeBack.foundDiffers = true;
            all.storeBack.collectedAbove = true;
            all.storeBack.htileStamped = true;
            expect(all, DepthViewPart::StoreBack, route == DepthViewRoute::NotPending, std::string("an HTILE clear stored back (") + DepthViewRouteName(route) + "), with every store-back evidence but Accessible");
        }
        // Stored back after a pending write was found, on a not-pending prediction: the recorder's
        // answer, which no other thread moves while this one holds the GPU lock, so nothing excuses it.
        auto afterPending = storedBack;
        afterPending.route = DepthViewRoute::NotPending;
        afterPending.h = 40;
        afterPending.htileRead = 40;
        afterPending.storeBackPending = true;
        expect(afterPending, DepthViewPart::StoreBack, false, "an HTILE clear stored back after a pending write found (not pending), no evidence");
        for (int which = 0; which < 5; ++which) expect(storeBackEvidence(afterPending, which), DepthViewPart::StoreBack, false, "an HTILE clear stored back after a pending write found (not pending), with store-back evidence " + std::to_string(which));
        auto later = storedBack;
        later.identity.clockMoved = true;
        later.sync.memoMoved = true;
        later.depthMemory = 60;
        expect(later, DepthViewPart::StoreBack, false, "an HTILE clear stored back, with evidence at later points only");
        // Every other verdict on a not-pending prediction (the r73 review): a pending one is the
        // recorder's answer, so nothing excuses it; not accessible, Accessible only; a read proof that
        // failed (not pending read), the memo key, the read proof or the HTILE stamped. The clock alone
        // (the collect above, the later points' clocks, refresh values above) never.
        for (const auto kind : {HtileVerdictKind::NotAccessible, HtileVerdictKind::NotPendingRead, HtileVerdictKind::PendingCovered, HtileVerdictKind::PendingNotCovering, HtileVerdictKind::PendingFillsOff}) {
            auto facts = agreed;
            facts.route = DepthViewRoute::NotPending;
            facts.h = 40;
            facts.htileRead = 40;
            facts.verdict = kind;
            facts.storeBackPending = kind != HtileVerdictKind::NotAccessible && kind != HtileVerdictKind::NotPendingRead;
            const auto what = std::string("today's ") + HtileVerdictName(kind) + " on a not-pending prediction";
            expect(facts, DepthViewPart::StoreBack, false, what + ", no evidence");
            for (int which = 0; which < 5; ++which) {
                const bool excuses = kind == HtileVerdictKind::NotAccessible ? which == 1 : kind == HtileVerdictKind::NotPendingRead && (which == 0 || which == 2 || which == 4);
                expect(storeBackEvidence(facts, which), DepthViewPart::StoreBack, excuses, what + ", with store-back evidence " + std::to_string(which));
            }
            auto clockOnly = storeBackEvidence(facts, 3);
            clockOnly.identity.clockMoved = true;
            clockOnly.sync.clockMoved = true;
            clockOnly.htileRead = 60;
            clockOnly.unitsLow = 60;
            clockOnly.unitsHigh = 60;
            clockOnly.depthMemory = 60;
            clockOnly.stencilMemory = 60;
            expect(clockOnly, DepthViewPart::StoreBack, false, what + ", with only the clock moved");
            clockOnly.storeBack.htileStamped = true;
            expect(clockOnly, DepthViewPart::StoreBack, kind == HtileVerdictKind::NotPendingRead, what + ", with the clock moved and the HTILE stamped");
            auto every = clockOnly;
            every.storeBack.memoMoved = true;
            every.storeBack.accessibleDiffers = true;
            every.storeBack.foundDiffers = true;
            expect(every, DepthViewPart::StoreBack, kind == HtileVerdictKind::NotAccessible || kind == HtileVerdictKind::NotPendingRead, what + ", with every store-back evidence");
        }
        auto pendingNot = agreed;
        pendingNot.verdict = HtileVerdictKind::PendingNotCovering;
        expect(pendingNot, DepthViewPart::StoreBack, false, "a pending fill no longer covering, no evidence");
        // The recorder's answer, which no other thread moves: a tracker or memo key move excuses nothing.
        for (const int which : {0, 2, 3, 4}) expect(storeBackEvidence(pendingNot, which), DepthViewPart::StoreBack, false, "a pending fill no longer covering, with store-back evidence " + std::to_string(which));
        expect(storeBackEvidence(pendingNot, 1), DepthViewPart::StoreBack, true, "a pending fill no longer covering, after the HTILE's Accessible answer moved");
    }
    // The identity part, by each way it fails.
    for (int way = 0; way < 4; ++way) {
        auto facts = agreed;
        if (way == 0) facts.cleared = true;
        else if (way == 1) facts.created = true;
        else if (way == 2) facts.sameSurface = false;
        else facts.retiredBefore = true;
        const auto what = "identity failure " + std::to_string(way);
        expect(facts, DepthViewPart::Identity, false, what + ", no evidence");
        // Only the surfaces serials (1), retired (2) and a cleared HTILE entry (4) can change the
        // identity's facts (the r73 review); the memo key (0), the units (3) and the clock (5) cannot.
        for (int which = 0; which < 6; ++which) {
            auto raced = facts;
            raced.identity = pointEvidence(raced.identity, which);
            expect(raced, DepthViewPart::Identity, which == 1 || which == 2 || which == 4, what + ", with identity evidence " + std::to_string(which));
        }
        auto unrelated = facts;
        unrelated.identity.memoMoved = true;
        unrelated.identity.oldestMoved = true;
        unrelated.identity.clockMoved = true;
        expect(unrelated, DepthViewPart::Identity, false, what + ", with the memo key, units and clock moved only");
        auto earlier = storeBackEvidence(facts, 0);
        expect(earlier, DepthViewPart::Identity, false, what + ", with store-back evidence only");
        auto later = facts;
        later.sync.clockMoved = true;
        later.depthMemory = 60;
        expect(later, DepthViewPart::Identity, false, what + ", with evidence at later points only");
    }
    // The sync part.
    {
        auto synced = agreed;
        synced.synced = true;
        expect(synced, DepthViewPart::Sync, false, "a sync, no evidence");
        // The memo key (0), the surfaces serials (1) and the units (3) excuse it; retired (2) and a
        // cleared HTILE entry (4) change no unit; the clock (5) only with the plane itself stamped since
        // the prediction (the r73 review).
        for (int which = 0; which < 6; ++which) {
            auto raced = synced;
            raced.sync = pointEvidence(raced.sync, which);
            expect(raced, DepthViewPart::Sync, which == 0 || which == 1 || which == 3, "a sync, with sync evidence " + std::to_string(which));
            raced.planeStamped = true;
            expect(raced, DepthViewPart::Sync, which != 2 && which != 4, "a sync with the plane stamped, with sync evidence " + std::to_string(which));
        }
        auto stamped = synced;
        stamped.planeStamped = true;
        expect(stamped, DepthViewPart::Sync, false, "a sync with the plane stamped but no clock move by the sync point or WrittenUnits' decision");
        auto written = synced;
        written.unitsLow = 51;
        written.unitsHigh = 51;
        expect(written, DepthViewPart::Sync, false, "a sync whose WrittenUnits read a generation above the prediction's, the plane not stamped");
        written.planeStamped = true;
        expect(written, DepthViewPart::Sync, true, "a sync whose WrittenUnits read a generation above the prediction's, the plane stamped");
        auto earlier = synced;
        earlier.identity.clockMoved = true;
        earlier.storeBack.collectedAbove = true;
        earlier.storeBack.htileStamped = true;
        expect(earlier, DepthViewPart::Sync, false, "a sync, with evidence at earlier points only");
        auto later = synced;
        later.depthMemory = 60;
        later.stencilMemory = 60;
        expect(later, DepthViewPart::Sync, false, "a sync, with refresh values above only");
    }
    // The refresh part.
    {
        const auto value = [&](int which, std::uint64_t to) {
            auto facts = agreed;
            switch (which) {
                case 0: facts.htileRead = to; break;
                case 1: facts.unitsLow = to; facts.unitsHigh = to; break;
                case 2: facts.depthMemory = to; break;
                default: facts.stencilMemory = to; break;
            }
            return facts;
        };
        for (int which = 0; which < 4; ++which) {
            expect(value(which, 51), DepthViewPart::Refresh, true, "refresh value " + std::to_string(which) + " above its prediction");
            expect(value(which, 49), DepthViewPart::Refresh, false, "refresh value " + std::to_string(which) + " below its prediction");
        }
        auto mixed = value(0, 51);
        mixed.depthMemory = 49;
        expect(mixed, DepthViewPart::Refresh, false, "one refresh value above and one below");
        auto unequal = agreed;
        unequal.unitsHigh = 51;
        expect(unequal, DepthViewPart::Refresh, false, "units unequal");
        auto missing = agreed;
        missing.htileReadFound = false;
        expect(missing, DepthViewPart::Refresh, false, "the pending read proof missing");
        auto earlier = value(2, 49);
        earlier.sync.clockMoved = true;
        earlier.identity.memoMoved = true;
        expect(earlier, DepthViewPart::Refresh, false, "a refresh value below its prediction, with evidence at earlier points");
    }
    Require(std::string(DepthViewMissName(DepthViewMiss::HostWritesPending)) == "host writes pending" && std::string(DepthViewRefusalName(DepthViewRefusal::NotCovered)) == "not covered" && std::string(HtileVerdictName(HtileVerdictKind::PendingCovered)) == "pending covered" && std::string(DepthViewPartName(DepthViewPart::Refresh)) == "refresh", "D5 names: a reason is misnamed");
    static_cast<void>(DepthViewLeaseReport());
    const auto report = DepthViewLeaseReport();
    for (const char* part : {"; D5 (APS5_EXPERIMENT_DEPTH_LEASES=", "eligible 0 of 0", "hits 0 (pending 0, not pending 0, no HTILE 0, not accessible 0)", "full-path verdicts: no HTILE 0,", "stored back 0; created 0, renewed 0, refused: created 0,", "not covered 0; misses: no lease 0,", "plane: unproven 0, stamped 0, HTILE: unproven 0, stamped 0, host writes pending 0", "; table evictions 0; =2: predicted hits 0, verified 0, raced 0 (store-back 0, identity 0, sync 0, refresh 0)"}) {
        Require(report.find(part) != std::string::npos, std::string("D5 report: '") + part + "' is not in: " + report);
    }
    std::cout << "depth view lease classification tests passed\n";
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.12 (c)): DepthSurfaceView leases over
// real calls on a watched block, the mode from the environment. Every target is mirrorable (D32, swizzle
// mode Z64KBX): A (an HTILE, not pending), A2 (A's surface with another HTILE), P (an HTILE a pending
// fill covers), S (D32S8, no HTILE) and S2 (S's depth plane with another stencil plane), N (an HTILE
// that is not accessible), U (a plane not 64 KiB aligned: one unit), M (several units), and W, G and H
// for the cases that sync. Each case runs at 0, 1 and 2, and after each call the state is today's:
// every unit equal and, with no sync, equal to TrackerGeneration(), as the depth memory is (the stencil
// memory too with a stencil plane, else 0), the pending route's read proof too, the not-pending one
// unchanged since its generation, every unit stale and the plane in the stale list, no pending clear.
// (A call's collects are made memo hits first, so they move no generation: the state is then exact in
// one thread.) At 1 and 2: a target's first call creates its surface (refused: created), the next makes
// the lease, and the next hits (=1) or is verified (=2), on each of the four routes; each event of the
// plan's table then gives its stated outcome, and the call after it hits again (a cleared HTILE taken
// and ClearDepthSurfaces leave no lease for one call: their refusal). A step across which the memo key
// moved by itself is tried again, as D5-0's are. The cases that sync come last, each on its own target:
// on a device that compares host imports below 1 MiB instead of watching them, the sync's mirror upload
// unwatches its plane, whose later calls then cannot hit (said once). At 0 nothing is counted (and
// D5-0 counts while profiling); at 1 and 2, D5-0 counts nothing. Without a collect memo key (no
// APS5_DEBUG_COVERAGE=1) every call is ineligible. False only when the device watches no memory.
bool depthViewLeaseTests(const Device& device, Recorder& recorder) {
    using namespace AgcDriver::GuestMemory;
    constexpr std::size_t block = 65536;
    constexpr std::size_t bytes = 24 * block;
    void* memory = AllocateWatched(bytes, block);
    if (memory == nullptr) {
        std::cout << "no write watching: the depth view leases not tested\n";
        return false;
    }
    struct Release {
        void* memory;
        ~Release() { ReleaseWatched(memory, bytes); }
    } release{memory};
    // A context with a detiler: a sync makes the depth plane's mirror (StorageTexture::Create). Declared
    // before the surfaces' clear below, so it outlives them.
    const auto& base = device.GetContext();
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    // ClearDepthSurfaces destroys the surfaces' images at once: the batch that recorded work over them
    // must finish first, on a failed expectation too.
    struct Clear {
        Recorder& recorder;
        VkDevice device;
        ~Clear() {
            try {
                recorder.Sync();
            } catch (...) {
            }
            ClearDepthSurfaces(device);
        }
    } clear{recorder, context.device};
    const int mode = DepthViewLeases();
    const bool profiling = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto address = reinterpret_cast<std::uint64_t>(memory);
    auto* data = static_cast<std::uint8_t*>(memory);
    std::memset(data, 0, bytes);
    constexpr VkExtent2D extent{64, 64};
    constexpr VkExtent2D large{256, 256};
    const auto htileBytes = static_cast<std::size_t>(HtileMetadataBytes(extent));
    const auto at = [&](std::size_t index) { return address + index * block; };
    // HTILE words with no clear code (HtileFillClears reads the low nibble).
    for (const std::size_t index : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{16}}) std::memset(data + index * block, 0xff, block / 2);
    const auto make = [](std::uint64_t address, std::uint64_t stencil, VkExtent2D size, VkFormat format, std::uint64_t htile) {
        DepthTarget target{address, stencil, size, format, 1.0f, 0};
        target.depthSwizzle = XorSwizzleMode(TextureTileMode::kZ64KBX);
        target.htileAddress = htile;
        return target;
    };
    const auto a = make(at(0), 0, extent, VK_FORMAT_D32_SFLOAT, at(1));
    const auto a2 = make(at(0), 0, extent, VK_FORMAT_D32_SFLOAT, at(2));
    const auto p = make(at(3), 0, extent, VK_FORMAT_D32_SFLOAT, at(4));
    const auto s = make(at(5), at(6), extent, VK_FORMAT_D32_SFLOAT_S8_UINT, 0);
    const auto s2 = make(at(5), at(7), extent, VK_FORMAT_D32_SFLOAT_S8_UINT, 0);
    // An HTILE in the first 64 KiB of the address space, which is never mapped.
    const auto n = make(at(8), 0, extent, VK_FORMAT_D32_SFLOAT, 0x1000);
    const auto u = make(at(9) + 4096, 0, extent, VK_FORMAT_D32_SFLOAT, 0);
    const auto m = make(at(12), 0, large, VK_FORMAT_D32_SFLOAT, at(16));
    const auto w = make(at(17), 0, extent, VK_FORMAT_D32_SFLOAT, 0);
    const auto g = make(at(18), 0, extent, VK_FORMAT_D32_SFLOAT, 0);
    const auto h = make(at(19), 0, extent, VK_FORMAT_D32_SFLOAT, 0);
    const auto b = make(at(20), 0, extent, VK_FORMAT_D32_SFLOAT, 0);
    const auto elsewhere = at(21);
    const auto planeBytes = [](const DepthTarget& target) {
        GuestTextureResource resource{};
        resource.baseAddress = target.address;
        resource.width = target.extent.width;
        resource.height = target.extent.height;
        resource.mipCount = 1;
        resource.allocatedMipCount = 1;
        resource.tileMode = TextureTileMode::kZ64KBX;
        resource.dimension = TextureDimension::k2D;
        resource.format = 22;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        return static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
    };
    Require(planeBytes(a) <= block && planeBytes(u) <= block && planeBytes(m) > block && planeBytes(m) <= 4 * block && HtileMetadataBytes(large) <= block / 2, "D5 fixture: a plane or HTILE outgrows its blocks");
    const auto settle = [&] {
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    };
    // Every range a call collects, collected first: the call's collects are then memo hits.
    const auto collect = [&](const DepthTarget& target) {
        CollectWrites(target.address, planeBytes(target));
        CollectWrites(target.address, static_cast<std::size_t>(DepthSliceBytes(target.extent, 4u)));
        if (target.stencilAddress != 0) CollectWrites(target.stencilAddress, static_cast<std::size_t>(DepthSliceBytes(target.extent, 1u)));
        if (target.htileAddress >= address) CollectWrites(target.htileAddress, static_cast<std::size_t>(HtileMetadataBytes(target.extent)));
    };
    enum class Seen { Off, Ineligible, Answered, Missed };
    struct Outcome {
        Seen seen = Seen::Off;
        DepthViewMiss miss = DepthViewMiss::Count;
        DepthViewLeaseCounts before;
        DepthViewLeaseCounts after;
    };
    const auto describe = [&](const Outcome& outcome) -> std::string {
        switch (outcome.seen) {
            case Seen::Off: return "nothing counted";
            case Seen::Ineligible: return "ineligible";
            case Seen::Answered: return mode == 1 ? "a hit" : "a verified prediction";
            default: return std::string("a miss (") + DepthViewMissName(outcome.miss) + ")";
        }
    };
    const auto eligible = DepthViewIndex(DepthViewIneligible::Eligible);
    // One call, classified by the counts it moved.
    const auto call = [&](const DepthTarget& target, bool collectFirst) {
        if (collectFirst) collect(target);
        Outcome outcome;
        outcome.before = DepthViewLeaseTotals();
        Require(DepthSurfaceView(context, target) != VK_NULL_HANDLE, "D5: the depth surface has no view");
        outcome.after = DepthViewLeaseTotals();
        const auto& before = outcome.before;
        const auto& after = outcome.after;
        if (mode == 0) {
            Require(after.calls == before.calls && after.hits == before.hits && after.created == before.created, "D5: a call was counted with the switch unset");
            return outcome;
        }
        Require(after.calls == before.calls + 1, "D5: a call was not counted once");
        if (after.ineligible[eligible] == before.ineligible[eligible]) {
            outcome.seen = Seen::Ineligible;
            return outcome;
        }
        std::size_t misses = 0;
        for (std::size_t i = 0; i < DepthViewIndex(DepthViewMiss::Count); ++i) {
            if (after.misses[i] == before.misses[i]) continue;
            Require(after.misses[i] == before.misses[i] + 1 && misses++ == 0, "D5: a call missed more than once");
            outcome.miss = static_cast<DepthViewMiss>(i);
        }
        for (std::size_t i = 0; i < 4; ++i) Require(after.raced[i] == before.raced[i], "D5 =2: a prediction raced in a single-threaded test");
        Require(after.predicted - before.predicted == after.verified - before.verified, "D5 =2: a predicted hit was not verified");
        if (mode == 1) Require(after.predicted == before.predicted && after.verified == before.verified, "D5 =1: a call predicted");
        if (mode == 2) Require(after.hits == before.hits && after.renewed == before.renewed, "D5 =2: a lease answered a call");
        const bool answered = mode == 1 ? after.hits == before.hits + 1 && after.renewed == before.renewed + 1 : after.verified == before.verified + 1;
        Require(answered != (misses == 1), "D5: an eligible call was neither answered nor missed once");
        outcome.seen = answered ? Seen::Answered : Seen::Missed;
        return outcome;
    };
    // Today's state after a call (all modes). With `synced`, only what a sync leaves the same.
    const auto checkState = [&](const DepthTarget& target, DepthViewRoute route, bool synced, const std::string& what) {
        const auto state = DepthViewLeaseProbe(context, target);
        Require(state.found && !state.units.empty() && std::all_of(state.units.begin(), state.units.end(), [&](std::uint64_t unit) { return unit == state.units[0]; }), "D5 state: " + what + ": no surface, or its units differ");
        Require(state.staleCount == state.units.size() && state.staleUnits == state.units.size() && state.stalePlaneListed && state.pendingClear == 0 && !state.retired, "D5 state: " + what + ": a unit not stale, the plane not listed, a pending clear left or the surface retired");
        if (synced) return;
        const auto generation = TrackerGeneration();
        Require(state.units[0] == generation && state.depthMemory == generation && state.stencilMemory == (target.stencilAddress != 0 ? generation : 0), "D5 state: " + what + ": the units (" + std::to_string(state.units[0]) + "), depth memory (" + std::to_string(state.depthMemory) + ") or stencil memory (" + std::to_string(state.stencilMemory) + ") are not the tracker generation " + std::to_string(generation));
        if (route == DepthViewRoute::Pending) Require(state.htileRead && state.htileReadValue == generation, "D5 state: " + what + ": the pending route's read proof is not the tracker generation");
        if (route == DepthViewRoute::NotPending) Require(state.htileRead && state.htileReadValue != 0 && UnchangedSince(target.htileAddress, htileBytes, state.htileReadValue), "D5 state: " + what + ": the not-pending route's read proof is missing or stamped");
    };
    struct Expected {
        bool answered;
        DepthViewMiss miss;
    };
    const Expected hit{true, DepthViewMiss::Count};
    const auto missing = [](DepthViewMiss miss) { return Expected{false, miss}; };
    const auto keyMiss = [](DepthViewMiss miss) { return miss == DepthViewMiss::Epoch || miss == DepthViewMiss::Allocations || miss == DepthViewMiss::Unwatch; };
    const auto matches = [](const Outcome& outcome, const Expected& expected) { return expected.answered ? outcome.seen == Seen::Answered : outcome.seen == Seen::Missed && outcome.miss == expected.miss; };
    const auto none = [] {};
    // One call of `target` after `action`, expected to give `expected` at 1 and 2. A call that missed by
    // a memo key move the step did not ask for is renewed and tried again, action included.
    const auto step = [&](const DepthTarget& target, DepthViewRoute route, auto&& action, Expected expected, const std::string& what, bool synced = false) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            action();
            const auto outcome = call(target, true);
            checkState(target, route, synced, what);
            if (mode == 0) return;
            if (matches(outcome, expected)) return;
            Require(outcome.seen == Seen::Missed && keyMiss(outcome.miss) && (expected.answered || !keyMiss(expected.miss)), "D5: " + what + ": " + describe(outcome) + " where " + (expected.answered ? std::string("an answer") : std::string("a miss (") + DepthViewMissName(expected.miss) + ")") + " was expected");
            static_cast<void>(call(target, true));
            checkState(target, route, synced, what + " (renewed)");
        }
        throw std::runtime_error("D5: " + what + ": the collect memo key kept moving by itself");
    };
    // Calls until one is answered (at 0: one call).
    const auto establish = [&](const DepthTarget& target, DepthViewRoute route, const std::string& what) {
        for (int attempt = 0; attempt < 4; ++attempt) {
            const auto outcome = call(target, true);
            checkState(target, route, false, what);
            if (mode == 0 || outcome.seen == Seen::Answered) return;
        }
        Require(false, "D5: " + what + ": no call was answered");
    };

    settle();
    BumpCollectEpoch();
    const bool keyed = CurrentCollectMemoKey().Valid();
    const auto start = DepthViewLeaseTotals();
    const auto startCounter = DepthLeaseTotals();
    if (!keyed) {
        for (const auto* target : {&a, &p, &n, &u}) {
            const auto outcome = call(*target, false);
            if (mode != 0) Require(outcome.seen == Seen::Ineligible, "D5: a call without a collect memo key was eligible");
        }
        const auto now = DepthViewLeaseTotals();
        if (mode != 0) Require(now.ineligible[DepthViewIndex(DepthViewIneligible::NoMemoKey)] == start.ineligible[DepthViewIndex(DepthViewIneligible::NoMemoKey)] + 4 && now.hits == start.hits && now.created == start.created, "D5: calls without a collect memo key were not ineligible for that reason");
        settle();
        std::cout << "depth view lease tests passed (" << (mode == 0 ? "switch unset, " : "") << "no collect memo key: every call ineligible)\n";
        return true;
    }

    // Each target's first call creates its surface, the next makes its lease (with the HTILE routes:
    // pending, not pending, not accessible, none), and the next is answered.
    recorder.NotePendingWordFill(p.htileAddress, htileBytes, 0xffffffffu);
    bool stencil = true;
    struct Target {
        const DepthTarget* target;
        DepthViewRoute route;
        const char* name;
    };
    std::vector<Target> targets{{&a, DepthViewRoute::NotPending, "A"}, {&p, DepthViewRoute::Pending, "P"}, {&s, DepthViewRoute::NoHtile, "S"}, {&s2, DepthViewRoute::NoHtile, "S2"}, {&n, DepthViewRoute::NotAccessible, "N"}, {&u, DepthViewRoute::NoHtile, "U"}, {&m, DepthViewRoute::NotPending, "M"}, {&w, DepthViewRoute::NoHtile, "W"}, {&g, DepthViewRoute::NoHtile, "G"}, {&h, DepthViewRoute::NoHtile, "H"}};
    for (auto it = targets.begin(); it != targets.end();) {
        Outcome outcome;
        try {
            outcome = call(*it->target, true);
        } catch (const std::exception& error) {
            // A D32S8 surface the device cannot make (the stencil cases are then not tested).
            Require(it->target == &s || it->target == &s2, std::string("D5: the first call of ") + it->name + " threw: " + error.what());
            std::cout << "D5: the D32S8 surface could not be made here (" << error.what() << "): the stencil cases not tested\n";
            stencil = false;
            it = targets.erase(it);
            continue;
        }
        checkState(*it->target, it->route, false, std::string("the first call of ") + it->name);
        if (mode != 0) Require(outcome.seen == Seen::Missed && outcome.miss == DepthViewMiss::NoLease && outcome.after.refusals[DepthViewIndex(DepthViewRefusal::Created)] == outcome.before.refusals[DepthViewIndex(DepthViewRefusal::Created)] + 1, std::string("D5: the first call of ") + it->name + " (" + describe(outcome) + ") was not refused as a creation");
        ++it;
    }
    if (!stencil) targets.erase(std::remove_if(targets.begin(), targets.end(), [&](const Target& target) { return target.target == &s || target.target == &s2; }), targets.end());
    for (const auto& target : targets) {
        const auto before = DepthViewLeaseTotals();
        establish(*target.target, target.route, std::string("the lease of ") + target.name);
        if (mode != 0) Require(DepthViewLeaseTotals().created > before.created, std::string("D5: no lease was made for ") + target.name);
    }
    if (mode != 0) {
        const auto now = DepthViewLeaseTotals();
        for (const auto route : {DepthViewRoute::NoHtile, DepthViewRoute::NotAccessible, DepthViewRoute::NotPending, DepthViewRoute::Pending}) {
            const auto index = DepthViewIndex(route);
            Require(mode == 1 ? now.hitRoutes[index] > start.hitRoutes[index] : now.verified > start.verified, std::string("D5: no call on the ") + DepthViewRouteName(route) + " route was answered");
        }
        Require(now.verdicts[DepthViewIndex(HtileVerdictKind::PendingCovered)] > start.verdicts[DepthViewIndex(HtileVerdictKind::PendingCovered)] && now.verdicts[DepthViewIndex(HtileVerdictKind::NotAccessible)] > start.verdicts[DepthViewIndex(HtileVerdictKind::NotAccessible)], "D5: the full paths' verdicts were not counted");
    }
    // A2 resolves to A's surface (sameSurface does not compare the HTILE): no surface is created.
    {
        const auto first = call(a2, true);
        checkState(a2, DepthViewRoute::NotPending, false, "the first call of A2");
        if (mode != 0) Require(first.seen == Seen::Missed && first.miss == DepthViewMiss::NoLease && first.after.created == first.before.created + 1, "D5: the first call of A2 (on A's surface) did not make its lease");
        step(a2, DepthViewRoute::NotPending, none, hit, "the second call of A2");
    }

    // A store over the stencil plane: nothing a hit skips reads it.
    if (stencil) {
        establish(s, DepthViewRoute::NoHtile, "S before a stencil store");
        step(s, DepthViewRoute::NoHtile, [&] { MarkWritten(s.stencilAddress + 4096, 64); }, hit, "a store over the stencil plane");
    }
    // An HTILE clear over P's pending HTILE: the live fast clear records it.
    {
        establish(p, DepthViewRoute::Pending, "P before an HTILE clear");
        NoteHtileDepthClear(p.htileAddress, htileBytes);
        Require((DepthViewLeaseProbe(context, p).pendingClear & VK_IMAGE_ASPECT_DEPTH_BIT) != 0, "D5: an HTILE clear over P left no pending clear");
        step(p, DepthViewRoute::Pending, none, hit, "an HTILE clear over the pending route's HTILE");
    }
    // A's HTILE: a store, then a GPU fill noted (A turns pending), then the reap (P's fill too).
    establish(a, DepthViewRoute::NotPending, "A before an HTILE store");
    step(a, DepthViewRoute::NotPending, [&] { MarkWritten(a.htileAddress, 4); }, missing(DepthViewMiss::HtileStamped), "a store over the not-pending HTILE");
    step(a, DepthViewRoute::NotPending, none, hit, "the call after the HTILE store");
    step(a, DepthViewRoute::Pending, [&] { recorder.NotePendingWordFill(a.htileAddress, htileBytes, 0xffffffffu); }, missing(DepthViewMiss::HtileNote), "a GPU fill noted over the not-pending HTILE");
    step(a, DepthViewRoute::Pending, none, hit, "the call after the noted fill (pending)");
    step(a, DepthViewRoute::NotPending, settle, missing(DepthViewMiss::Reaped), "the batch of the pending fill finished");
    step(a, DepthViewRoute::NotPending, none, hit, "the call after the reap");
    step(p, DepthViewRoute::NotPending, none, missing(DepthViewMiss::Reaped), "P's pending fill finished");
    step(p, DepthViewRoute::NotPending, none, hit, "P's call after the reap");
    // A reap on the not-pending route revokes nothing (a reap that stamps one of A's ranges is said).
    {
        establish(a, DepthViewRoute::NotPending, "A before a reap");
        settle();
        const auto outcome = call(a, true);
        checkState(a, DepthViewRoute::NotPending, false, "the call after a reap (not pending)");
        if (mode != 0 && outcome.seen != Seen::Answered) {
            Require(outcome.seen == Seen::Missed && (outcome.miss == DepthViewMiss::PlaneStamped || outcome.miss == DepthViewMiss::HtileStamped || keyMiss(outcome.miss)), "D5: a reap on the not-pending route: " + describe(outcome));
            std::cout << "D5: the reap stamped or moved the memo key over A (" << describe(outcome) << ")\n";
        }
    }
    // HTILE clears and fills over A and A2.
    establish(a, DepthViewRoute::NotPending, "A before an HTILE clear");
    step(a, DepthViewRoute::NotPending, [&] { NoteHtileDepthClear(a.htileAddress, htileBytes); }, missing(DepthViewMiss::HtileReadProofMoved), "an HTILE clear over the not-pending HTILE (its surface's)");
    step(a, DepthViewRoute::NotPending, none, hit, "the call after the HTILE clear");
    establish(a2, DepthViewRoute::NotPending, "A2 before an HTILE clear");
    step(a2, DepthViewRoute::NotPending, [&] { NoteHtileDepthClear(a2.htileAddress, htileBytes); }, missing(DepthViewMiss::ClearedHtile), "an HTILE clear no resident surface covers");
    step(a2, DepthViewRoute::NotPending, none, missing(DepthViewMiss::NoLease), "the call after the cleared HTILE was taken (refused)");
    step(a2, DepthViewRoute::NotPending, none, hit, "the call after the lease was made again");
    establish(a, DepthViewRoute::NotPending, "A before an HTILE fill");
    step(a, DepthViewRoute::NotPending, [&] { NoteDepthMetadataFill(a.htileAddress, htileBytes, 0xffffffffu); }, missing(DepthViewMiss::HtileReadProofMoved), "a metadata fill over the not-pending HTILE");
    step(a, DepthViewRoute::NotPending, none, hit, "the call after the HTILE fill");
    step(a, DepthViewRoute::NotPending, [&] {
        NoteDepthMetadataFill(a.address, static_cast<std::size_t>(DepthSliceBytes(extent, 4u)), 0x3f800000u);
        Require((DepthViewLeaseProbe(context, a).pendingClear & VK_IMAGE_ASPECT_DEPTH_BIT) != 0, "D5: a plane fill over A left no pending clear");
    }, hit, "a metadata fill over the depth plane (the live fast clear applies it)");
    // Another target binding S's depth plane: the live takePlanes copies it back.
    if (stencil) {
        establish(s, DepthViewRoute::NoHtile, "S before S2 binds its plane");
        step(s, DepthViewRoute::NoHtile, [&] {
            static_cast<void>(call(s2, true));
            checkState(s2, DepthViewRoute::NoHtile, false, "S2 binding S's plane");
        }, hit, "another target binding the same depth plane");
    }
    // M's units: half of them stored at an older generation with no store over the plane since (a hit,
    // as today does not sync), then with a store in another unit since (a miss: the plane row is
    // stricter than today, which syncs nothing either).
    {
        establish(m, DepthViewRoute::NotPending, "M before a NoteStored");
        const auto older = TrackerGeneration();
        MarkWritten(elsewhere, 16);
        MarkWritten(elsewhere + 4096, 16);
        establish(m, DepthViewRoute::NotPending, "M after stores elsewhere");
        const auto units = DepthViewLeaseProbe(context, m).units.size();
        Require(units >= 2, "D5 fixture: M has one unit");
        step(m, DepthViewRoute::NotPending, [&] { Require(DepthSurfaceNoteStoredForTest(context, m, older, units / 2), "D5: M has no surface"); }, hit, "half of M's units stored at an older generation with no store since");
        const auto olderAgain = TrackerGeneration();
        step(m, DepthViewRoute::NotPending, [&] {
            MarkWritten(m.address + (units - 1) * block + 100, 16);
            Require(DepthSurfaceNoteStoredForTest(context, m, TrackerGeneration()), "D5: M has no surface");
        }, hit, "a store in M's last unit, then every unit stored after it");
        step(m, DepthViewRoute::NotPending, [&] { Require(DepthSurfaceNoteStoredForTest(context, m, olderAgain, units / 2), "D5: M has no surface"); }, missing(DepthViewMiss::PlaneStamped), "half of M's units stored before a store in its last unit");
        Require(!DepthViewLeaseProbe(context, m).mirrored, "D5: today synced M, whose changed units are none");
        step(m, DepthViewRoute::NotPending, none, hit, "M's call after the units were assigned again");
    }
    // A retire, an epoch bump and a new target's surface.
    establish(a, DepthViewRoute::NotPending, "A before a retire");
    step(a, DepthViewRoute::NotPending, [&] { RetireDepthSurfaces(context.device, a.address, block); }, missing(DepthViewMiss::Retired), "the surface retired");
    step(a, DepthViewRoute::NotPending, none, hit, "the call after the un-retire");
    step(a, DepthViewRoute::NotPending, [] { BumpCollectEpoch(); }, missing(DepthViewMiss::Epoch), "an epoch bump");
    step(a, DepthViewRoute::NotPending, none, hit, "the call after the epoch bump");
    step(a, DepthViewRoute::NotPending, [&] {
        static_cast<void>(call(b, true));
        checkState(b, DepthViewRoute::NoHtile, false, "a new target's first call");
    }, missing(DepthViewMiss::Surfaces), "a new target's surface created");
    step(a, DepthViewRoute::NotPending, none, hit, "the call after the new surface");
    establish(n, DepthViewRoute::NotAccessible, "N, the not-accessible route");

    // The cases that sync, each on its own target. After its sync a target's next calls are answered
    // again, unless the sync's upload left its plane compared rather than watched (no proof).
    const auto afterSync = [&](const DepthTarget& target, const std::string& what) {
        if (Watched(target.address, planeBytes(target))) {
            establish(target, DepthViewRoute::NoHtile, what + ": the calls after the sync");
            return;
        }
        std::cout << "D5: " << what << ": the sync's upload left the plane compared, not watched; its later calls are not answered\n";
        for (int attempt = 0; attempt < 2; ++attempt) {
            const auto outcome = call(target, true);
            checkState(target, DepthViewRoute::NoHtile, true, what + ": a call after the sync");
            if (mode != 0) Require(outcome.seen != Seen::Answered, "D5: " + what + ": a plane without a proof was answered");
        }
    };
    {
        establish(w, DepthViewRoute::NoHtile, "W before a store over its plane");
        Require(!DepthViewLeaseProbe(context, w).mirrored, "D5 fixture: W synced before its store");
        step(w, DepthViewRoute::NoHtile, [&] { MarkWritten(w.address + 4096, 64); }, missing(DepthViewMiss::PlaneStamped), "a store over the depth plane", true);
        Require(DepthViewLeaseProbe(context, w).mirrored, "D5: today did not sync W's stamped plane");
        afterSync(w, "W");
    }
    {
        establish(u, DepthViewRoute::NoHtile, "U before a store in front of its plane");
        try {
            step(u, DepthViewRoute::NoHtile, [&] { MarkWritten(at(9) + 100, 16); }, missing(DepthViewMiss::PlaneStamped), "a store in front of the unaligned plane in its first block", true);
            Require(DepthViewLeaseProbe(context, u).mirrored, "D5: today did not sync U, whose first block was stamped");
            afterSync(u, "U");
        } catch (const std::exception& error) {
            if (std::string_view(error.what()).starts_with("D5")) throw;
            std::cout << "D5: the unaligned plane's mirror could not be made here (" << error.what() << "): its sync not tested\n";
        }
    }
    {
        // The refutation's gap (GPU-OWNED-PLAN 11.2 row 1): a store over G's plane after `older`, the
        // units raised past it (as a plane store's NoteStored raises them), G answered, then the units
        // set back to `older`: the lease reads the live oldest generation, so G misses and today syncs.
        establish(g, DepthViewRoute::NoHtile, "G before the gap");
        const auto older = TrackerGeneration();
        step(g, DepthViewRoute::NoHtile, [&] {
            MarkWritten(g.address + 4096, 64);
            Require(DepthSurfaceNoteStoredForTest(context, g, TrackerGeneration()), "D5: G has no surface");
        }, hit, "G's units raised past a store over its plane");
        step(g, DepthViewRoute::NoHtile, none, hit, "G's call after its units were raised");
        Require(!DepthViewLeaseProbe(context, g).mirrored, "D5: today synced G before its units went back");
        step(g, DepthViewRoute::NoHtile, [&] { Require(DepthSurfaceNoteStoredForTest(context, g, older), "D5: G has no surface"); }, missing(DepthViewMiss::PlaneStamped), "G's units set back below the store (the refutation's gap)", true);
        Require(DepthViewLeaseProbe(context, g).mirrored, "D5: today's re-sync did not run after G's units went back");
        afterSync(g, "G");
    }
    {
        // A host write over H's plane that ended unstamped (queued for the next tracker lock holder, as
        // under APS5_EXPERIMENT_HOST_WRITE_STAMPS=1; the hook installed here): =1's draining lock stamps
        // it, a stamped plane; =2's prediction takes no drain, a miss of its own; today syncs.
        establish(h, DepthViewRoute::NoHtile, "H before a host write");
        struct HostWriteHook {
            HostWriteHook() { GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(&note); }
            ~HostWriteHook() { GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(nullptr); }
            HostWriteHook(const HostWriteHook&) = delete;
            HostWriteHook& operator=(const HostWriteHook&) = delete;
            static void note(std::uintptr_t address, std::size_t bytes) { AgcDriver::GuestMemory::NoteHostWrite(address, bytes); }
        } hook;
        collect(h);
        {
            auto* bytesOver = reinterpret_cast<std::uint8_t*>(h.address + 4096);
            const GuestArena::HostWrite write(bytesOver, 16);
            Require(write.Open(), "D5: the host write did not open");
            std::memset(bytesOver, 0x66, 16);
        }
        const auto outcome = call(h, false);
        checkState(h, DepthViewRoute::NoHtile, true, "a host write over the depth plane");
        if (mode != 0) Require(matches(outcome, missing(mode == 1 ? DepthViewMiss::PlaneStamped : DepthViewMiss::HostWritesPending)), "D5: a host write over the depth plane: " + describe(outcome));
        Require(DepthViewLeaseProbe(context, h).mirrored, "D5: today did not sync H after the host write");
        afterSync(h, "H");
    }
    // ClearDepthSurfaces (the device teardown's, after the batch finished): every lease misses; the
    // call that creates the surface again is refused, the next makes the lease.
    step(a, DepthViewRoute::NotPending, [&] {
        settle();
        ClearDepthSurfaces(context.device);
    }, missing(DepthViewMiss::Surfaces), "the surfaces cleared");
    step(a, DepthViewRoute::NotPending, none, missing(DepthViewMiss::NoLease), "the call after the surface was created again (refused)");
    step(a, DepthViewRoute::NotPending, none, hit, "the call after the lease was made again");

    const auto end = DepthViewLeaseTotals();
    const auto counter = DepthLeaseTotals();
    if (mode == 0) Require(end.calls == start.calls && end.hits == start.hits && end.created == start.created && end.predicted == start.predicted, "D5: calls were counted with the switch unset");
    if (mode == 0 && profiling) Require(counter.calls > startCounter.calls, "D5: D5-0 counted nothing while profiling with the switch unset");
    if (mode != 0 && profiling) Require(counter.calls == startCounter.calls, "D5: D5-0 counted while the switch is set");
    if (mode == 1) Require(end.hits > start.hits && end.renewed > start.renewed && end.predicted == start.predicted && end.verified == start.verified, "D5 =1: no hit, or a prediction");
    if (mode == 2) Require(end.hits == start.hits && end.predicted > start.predicted && end.verified - start.verified == end.predicted - start.predicted, "D5 =2: a hit, no prediction, or a prediction not verified");
    for (std::size_t i = 0; i < 4; ++i) Require(end.raced[i] == start.raced[i], "D5: a call raced in a single-threaded test");
    settle();
    std::cout << "depth view lease tests passed (APS5_EXPERIMENT_DEPTH_LEASES=" << mode << ")\n";
    return true;
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--collect-mapping-history-only") {
            if (!collectMappingHistoryTests()) return 77;
            std::cout << "collect mapping history tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--hint-mapping-history-only") {
            if (!hintMappingHistoryTests()) return 77;
            std::cout << "hint mapping history tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--partial-watch-restoration-only") {
            if (!partialWatchRestorationTests()) return 77;
            std::cout << "partial private write-watch restoration tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--device-selection-only") {
            deviceSelectionTests();
            std::cout << "recorder device selection tests passed\n";
            return 0;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_MEMOS, GPU-OWNED-PLAN 9.4.2 D4-A): the storage cache's
        // surface size and key memos, without a device; the mode comes from the environment.
        if (argc == 2 && std::string_view(argv[1]) == "--target-memos-only") {
            targetMemoTests();
            return 0;
        }
        // LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0): the DepthSurfaceView lease counter's
        // classification and counts, without a device.
        if (argc == 2 && std::string_view(argv[1]) == "--depth-lease-classify-only") {
            depthLeaseClassifyTests();
            return 0;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.12 (b)): =2's classification of
        // a verified DepthSurfaceView call, without a device.
        if (argc == 2 && std::string_view(argv[1]) == "--depth-view-lease-classify-only") {
            depthViewLeaseClassifyTests();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--queued-write-groups-only") {
            queuedWriteGroupTests();
            std::cout << "queued write group tests passed\n";
            return 0;
        }
        std::unique_ptr<Device> created;
        try {
            const bool submitThread = argc == 2 && std::string_view(argv[1]).starts_with("--submit-thread");
            created = std::make_unique<Device>(argc == 2 && (std::string_view(argv[1]) == "--import-watch-other-family-only" || std::string_view(argv[1]) == "--deferred-settle-other-family-only" || std::string_view(argv[1]) == "--host-import-lifetime-other-family-only"), submitThread);
        } catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::cout << "skipped, no usable Vulkan device: " << error.what() << '\n';
            return 77;
        }
        Device& device = *created;
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-lifetime-only") return RunHostImportLifetimeTests(device.GetContext());
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-mutation-race-only") return RunHostImportMutationRaceTests(device.GetContext());
        if (argc == 2 && (std::string_view(argv[1]) == "--gpu-timing-complete-only" || std::string_view(argv[1]) == "--gpu-timing-incomplete-only")) {
            return gpuTimingCompletionTests(device, std::string_view(argv[1]) == "--gpu-timing-incomplete-only");
        }
        if (argc == 2 && std::string_view(argv[1]) == "--import-watch-only") {
            importWatchTests(device);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--import-watch-other-family-only") {
            if (device.GetContext().settleQueue == VK_NULL_HANDLE) {
                std::cout << "skipped, no other queue family to settle imports on\n";
                return 77;
            }
            importWatchTests(device);
            return 0;
        }
#ifdef _WIN32
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-lifetime-other-family-only") {
            if (device.GetContext().settleFamily == device.GetContext().queueFamily) {
                std::cout << "skipped, no other queue family to settle imports on\n";
                return 77;
            }
            return RunHostImportLifetimeTests(device.GetContext());
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--deferred-settle-only" || std::string_view(argv[1]) == "--deferred-settle-other-family-only")) {
            if (std::string_view(argv[1]) == "--deferred-settle-other-family-only" && device.GetContext().settleFamily == device.GetContext().queueFamily) {
                std::cout << "skipped, no other queue family to settle imports on\n";
                return 77;
            }
            return deferredSettleTests(device) ? 0 : 77;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--settle-first-use-only") return settleFirstUseTests(device) ? 0 : 77;
        if (argc == 2 && std::string_view(argv[1]) == "--settle-earlier-import-only") return settleEarlierImportTests(device) ? 0 : 77;
        if (argc == 2 && std::string_view(argv[1]) == "--settle-retire-only") return settleRetireTests(device) ? 0 : 77;
        if (argc == 2 && std::string_view(argv[1]) == "--settle-storage-image-only") return settleStorageImageTests(device) ? 0 : 77;
        if (argc == 2 && std::string_view(argv[1]) == "--reimport-bench") return reimportBench(device);
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-cut-only") return hostImportCutTests(device);
#endif
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-retirement-only") {
            const auto alignment = device.GetContext().hostImportAlignment;
            if (alignment == 0 || alignment > 65536) {
                std::cout << "skipped, host import retirement requires host imports aligned within 64 KiB; alignment=" << alignment << '\n';
                return 77;
            }
            hostImportRetirementTests(device);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--sparse-run-import-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, sparse run imports require host imports\n";
                return 77;
            }
            return sparseRunImportTests(device, false) ? 0 : 77;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--sparse-run-copied-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, sparse run imports require host imports\n";
                return 77;
            }
            return sparseRunImportTests(device, true) ? 0 : 77;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--pipeline-library-only") {
            pipelineLibraryTests(device);
            return 0;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--benchmark-pipeline-cache" || std::string_view(argv[1]) == "--pipeline-cache-only")) {
            pipelineCacheTests(device, std::string_view(argv[1]) == "--benchmark-pipeline-cache");
            return 0;
        }

        if (argc == 2 && std::string_view(argv[1]) == "--pipeline-cache-file-only") {
            return pipelineCacheFileTests(device);
        }

        if (argc == 2 && std::string_view(argv[1]) == "--descriptor-pool-only") {
            descriptorPoolTests(device);
            return 0;
        }
        std::unique_lock gpu(GpuMutex());
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_SUBMIT_THREAD, PARALLEL-COMMIT-PLAN A4).
        if (argc == 2 && std::string_view(argv[1]) == "--submit-thread-only") return submitThreadTests(device, 1);
        if (argc == 2 && std::string_view(argv[1]) == "--submit-thread-verify-only") return submitThreadTests(device, 2);
#ifdef _WIN32
        if (argc == 2 && std::string_view(argv[1]) == "--storage-cpu-snapshot-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                Require(std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr, "storage snapshot test requires host imports");
                std::cout << "skipped, host imports unavailable\n";
                return 77;
            }
            StorageCpuSnapshotTest::Run(device.GetContext());
            std::cout << "Storage CPU snapshot tests passed\n";
            return 0;
        }
#endif
        std::cout << "host imports " << (PrepareImportWatch(device.GetContext()) == ImportWatch::Unwatch ? "are compared" : "stay watched") << '\n';
        Recorder recorder(device.GetContext());
        recorder.Activate();
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_PROOF_UNITS, GPU-OWNED-PLAN L2-1): both refresh proof suites,
        // for the switch's rows; skipped (77) when the unit-checked suite could not test. The upstream
        // suite needs a watched host import, which a device whose host imports are compared lacks.
        if (argc == 2 && std::string_view(argv[1]) == "--refresh-proof-only") {
            refreshProofTests(device, recorder);
            if (!refreshProofUnitTests(device, recorder)) {
                std::cout << "skipped, the device imports no watched host memory for the unit-checked refresh proofs\n";
                return 77;
            }
            return 0;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_KEY_PIN, GPU-OWNED-PLAN 9.3, K1): both pinned key proof
        // suites, for the switch's rows; skipped (77) when either could not test (no write watching,
        // or no watched host import for the refresh suite).
        if (argc == 2 && std::string_view(argv[1]) == "--key-pin-only") {
            const bool keys = keyPinProofTests(device, recorder);
            if (!refreshKeyPinTests(device, recorder) || !keys) {
                std::cout << "skipped, the device watches no memory or imports no watched host memory for the pinned key proofs\n";
                return 77;
            }
            return 0;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): the lease refresh
        // and lease draw suites, for the switch's rows; skipped (77) when either could not test (no
        // watched host import, or no collect memo key without APS5_DEBUG_COVERAGE=1).
        if (argc == 2 && std::string_view(argv[1]) == "--target-lease-only") {
            const bool refreshes = targetLeaseRefreshTests(device, recorder);
            if (!targetLeaseDrawTests(device, recorder) || !refreshes) {
                std::cout << "skipped, the device imports no watched host memory or the lookups have no collect memo key for the target leases\n";
                return 77;
            }
            return 0;
        }
        // LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0): the lease counter over real
        // DepthSurfaceView calls, as the environment sets it up; skipped (77) without write watching.
        if (argc == 2 && std::string_view(argv[1]) == "--depth-lease-only") {
            if (!depthLeaseCounterTests(device, recorder)) {
                std::cout << "skipped, the device watches no memory for the depth lease counter\n";
                return 77;
            }
            return 0;
        }
        // LOCAL DEBUG (APS5_PROFILE_DRAW, REVALIDATE-PLAN 7 and 9, I0): the draw-input lease counters
        // over real CopyDrawInput calls, as the environment sets them up; skipped (77) without write
        // watching.
        if (argc == 2 && std::string_view(argv[1]) == "--input-lease-counter-only") {
            if (!drawInputLeaseCounterTests(device, recorder)) {
                std::cout << "skipped, the device watches no memory for the draw input lease counters\n";
                return 77;
            }
            std::cout << "draw input lease counter tests passed\n";
            return 0;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_INPUT_LEASES, REVALIDATE-PLAN 5 and 9, L1): the draw-input
        // leases over real CopyDrawInput calls in the switch's mode, and =2's death rows; skipped (77)
        // without write watching.
        if (argc == 2 && std::string_view(argv[1]).starts_with("--input-lease-") && (std::string_view(argv[1]) == "--input-lease-only" || std::string_view(argv[1]).starts_with("--input-lease-death-"))) {
            const std::string_view which = argv[1];
            const auto death = which == "--input-lease-death-c" ? InputLeaseDeath::C : which == "--input-lease-death-m" ? InputLeaseDeath::M : which == "--input-lease-death-p1" ? InputLeaseDeath::P1 : which == "--input-lease-death-p2" ? InputLeaseDeath::P2 : InputLeaseDeath::None;
            Require(which == "--input-lease-only" || death != InputLeaseDeath::None, "an unknown draw input lease death row");
            if (!drawInputLeaseTests(device, recorder, gpu, death)) {
                std::cout << "skipped, the device watches no memory for the draw input leases\n";
                return 77;
            }
            std::cout << "draw input lease tests passed\n";
            return 0;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.12 (c)): DepthSurfaceView leases
        // over real calls, the mode and the memo key as the environment sets them up; skipped (77) only
        // without write watching.
        if (argc == 2 && std::string_view(argv[1]) == "--depth-view-lease-only") {
            if (!depthViewLeaseTests(device, recorder)) {
                std::cout << "skipped, the device watches no memory for the depth view leases\n";
                return 77;
            }
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--copied-direct-only") {
            copiedDirectOrderingTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--pending-write-snapshot-only") {
            pendingWriteSnapshotTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--benchmark-pending-write-snapshot") {
            pendingWriteSnapshotBenchmark(recorder);
            return 0;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--storage-upload-lifetime-only" || std::string_view(argv[1]) == "--storage-upload-failure-only")) {
            storageUploadLifetimeTests(device, recorder, std::string_view(argv[1]) == "--storage-upload-failure-only");
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--tiled-resolve-only") {
            if ((device.GetContext().limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0 || !device.Graphics()) {
                std::cout << "skipped, the device lacks 4x color targets or a graphics queue\n";
                return 77;
            }
            tiledResolveTests(device, recorder);
            std::cout << "CB resolves into tiled targets tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--cube-only") {
            singleCubeTests(device, recorder);
            cubeExtraSliceTests(device, recorder);
            std::cout << "Single cube snapshot and storage sampling tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--sampled-storage-base-level-usage-only") {
            sampledStorageBaseLevelUsageTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--unchanged-cpu-stamps-only") {
            unchangedCpuStampTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--kept-unit-settle-only") {
            keptUnitSettleTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--clear-hand-over-only") {
            clearHandOverTests(device, recorder, false);
            if (!clearHandOverTests(device, recorder, true)) {
                std::cout << "skipped, the device imports no watched host memory for the clear-key hand-over\n";
                return 77;
            }
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--integer-clear-only") {
            integerClearTests(device);
            return 0;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_SNAPSHOT_HASH, GPU-OWNED-PLAN L1-1).
        if (argc == 2 && std::string_view(argv[1]) == "--snapshot-refresh-only") {
            if (!drawSnapshotRefreshTests(device)) return 77;
            std::cout << "draw snapshot refresh tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--settled-baseline-only") {
            settledBaselineTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--completion-labels-only") {
            completionCountTests(device, recorder);
            afterRecordedWorkTests(device, recorder);
            directCompletionLabelTests(recorder);
            labelTests(recorder);
            lateLabelTests(recorder);
            largeLabelTests(recorder);
            std::cout << "Completion label storage and ordering tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--read-written-staging-only") {
            if (!readWrittenStagingTests(device, recorder)) return 77;
            std::cout << "read and written staging tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--written-buffer-order-only") {
            if (!writtenBufferOrderTests(device, recorder)) return 77;
            std::cout << "written buffer order tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--read-list-only") {
            readListTests(device, recorder);
            std::cout << "Read list tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--deferred-failure-report-only") {
            deferredFailureReportTests(recorder);
            std::cout << "Deferred failure report tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-unmap-only") {
            gpu.unlock();
            hostImportUnmapTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--slice-dirty-only") {
            sliceDirtyTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--resident-present-only") {
            RunResidentPresentTests(device.GetContext());
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--parked-import-unmap-only") {
            gpu.unlock();
            parkedImportUnmapTests(device);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--idle-publish-wait-only") {
            if (!AgcDriver::Graphics::UnitShadowEnabled() || device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, unit shadows or host imports are unavailable\n";
                return 77;
            }
            idlePublishWaitTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--unit-shadow-only") {
            if (!AgcDriver::Graphics::UnitShadowEnabled() || device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, unit shadows or host imports are unavailable\n";
                return 77;
            }
            if (!unitShadowTests(device, recorder)) {
                std::cout << "skipped, the device does not import watched host memory for unit shadows\n";
                return 77;
            }
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--split-writeback-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            splitWriteBackTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--storage-write-back-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            storageRefreshTests(device, recorder, false);
            storageRefreshTests(device, recorder, true);
            writeBackPaddingTests(device, recorder, false);
            writeBackPaddingTests(device, recorder, true);
            singlePassTests(device, recorder, false);
            singlePassTests(device, recorder, true);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--cmask-only") {
            cmaskPassTests(device, recorder);
            std::cout << "CMASK copied clear, write-back and failure tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--writeback-blocks-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            writeBackBlockTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--metadata-pass-only") {
            metadataPassTests(device, recorder);
            std::cout << "Color metadata pass tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--pending-key-stores-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            if (!pendingKeyStoreTests(device, recorder)) {
                std::cout << "skipped, the device does not import the pending key store block\n";
                return 77;
            }
            std::cout << "Pending DCC key store tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--sample-dumps-only") {
            if (device.GetContext().hostImportAlignment == 0) {
                std::cout << "skipped, host imports are unavailable\n";
                return 77;
            }
            if (!device.GetContext().bufferDeviceAddress) {
                std::cout << "skipped, the device lacks buffer device addresses\n";
                return 77;
            }
            sampleDumpTests(device, recorder);
            std::cout << "Occlusion counter dump tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--shadowed-key-stores-only") {
            if (!shadowedKeyStoreTests(device, recorder)) {
                std::cout << "skipped, the device imports no watched host memory under a unit shadow\n";
                return 77;
            }
            std::cout << "DCC keys under unit shadows tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--draw-inputs-in-place-only") {
            drawInputInPlaceTests(device, recorder);
            std::cout << "In-place draw input tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--draw-inputs-copied-only") {
            drawInputInPlaceTests(device, recorder, true);
            std::cout << "Copied draw input tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--multisample-only") {
            multisampleGuestAccessTests(device, recorder);
            fmaskDecompressMsaaTests(device, recorder);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--bound-state-only") return boundStateTests(device, recorder);
        if (argc == 2 && std::string_view(argv[1]) == "--draw-descriptors-only") {
            drawDescriptorWriteTests(device, recorder);
            drawSnapshotTableSetTests(device, recorder);
            imageTableFirstLayerRevalidateTests(device, recorder);
            std::cout << "Draw descriptor writes and snapshot overrides passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--space-pending-writes-only") {
            overlappedRangeTests(recorder);
            spaceWriteSyncTests(device, recorder);
            std::cout << "Address space pending write tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--draw-snapshots-only") {
            misalignedSnapshotTests(device, recorder);
            misalignedRegionTests(device, recorder);
            drawSnapshotReuseTests(device, recorder);
            drawDescriptorWriteTests(device, recorder);
            if (!drawSnapshotArenaTests(device, recorder)) return 77;
            std::cout << "Draw snapshot tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--draw-snapshot-cap-only") {
            drawSnapshotCapTests(device, recorder);
            unwrittenSnapshotTests(device);
            drawSnapshotUploadTests(device, recorder);
            std::cout << "Draw snapshot cap tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--draw-snapshot-epoch-only") {
            drawSnapshotEpochTests(device);
            return 0;
        }

        if (argc == 2 && (std::string_view(argv[1]) == "--vertex-target-range-copied" || std::string_view(argv[1]) == "--vertex-target-range-imported")) {
            return vertexTargetRangeTests(device, recorder, std::string_view(argv[1]) == "--vertex-target-range-imported") ? 0 : 77;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_HELPER_FETCH_PLAN, PARALLEL-COMMIT-PLAN B2).
        if (argc == 2 && std::string_view(argv[1]) == "--helper-fetch-plan-only") return helperFetchPlanTests(device, recorder) ? 0 : 77;

#ifdef _WIN32
        if (argc == 2 && std::string_view(argv[1]) == "--host-import-second-mapping-only") {
            secondMappingImportTests(device);
            return 0;
        }
#endif
        readTrackingTests(device, recorder);
        readListTests(device, recorder);
        overlappedRangeTests(recorder);
        spaceWriteSyncTests(device, recorder);
        writeSettledTests(device, recorder);
        completionCountTests(device, recorder);
        afterRecordedWorkTests(device, recorder);
        directCompletionLabelTests(recorder);
        batchStampTests(recorder);
        labelTests(recorder);
        lateLabelTests(recorder);
        largeLabelTests(recorder);
        unchangedSinceTests();
        closeRaceTests(device, recorder);
        keyProofTests(device, recorder);
        keyPinProofTests(device, recorder);
        targetKeyProofTests(device, recorder);
        refreshProofTests(device, recorder);
        refreshProofUnitTests(device, recorder);
        refreshKeyPinTests(device, recorder);
        resourceReadTests(device, recorder);
        readWrittenStagingTests(device, recorder);
        depthSurfaceProofTests(device, recorder);
        depthLeaseClassifyTests();
        static_cast<void>(depthLeaseCounterTests(device, recorder));
        depthViewLeaseClassifyTests();
        static_cast<void>(depthViewLeaseTests(device, recorder));
        multisampleGuestAccessTests(device, recorder);
        tiledResolveTests(device, recorder);
        fmaskDecompressMsaaTests(device, recorder);
        misalignedSnapshotTests(device, recorder);
        misalignedRegionTests(device, recorder);
        drawSnapshotReuseTests(device, recorder);
        drawDescriptorWriteTests(device, recorder);
        drawSnapshotTableSetTests(device, recorder);
        imageTableFirstLayerRevalidateTests(device, recorder);
        drawSnapshotCapTests(device, recorder);
        drawSnapshotEvictionTests(device);
        unwrittenSnapshotTests(device);
        drawSnapshotUploadTests(device, recorder);
        drawInputReuseTests(device, recorder);
        drawInputInPlaceTests(device, recorder);
        RunResidentPresentTests(device.GetContext());
        sliceDirtyTests(device, recorder);
        storeRunTests(device, recorder);
        queuedWriteGroupTests();
        remappedImportTests(device);
#ifdef _WIN32
        secondMappingImportTests(device);
#endif
        movedMetadataTests(device, recorder);
        viewPastLastMipTests(device, recorder);
        keysFillTests(device, recorder);
        clearHandOverTests(device, recorder, false);
        clearHandOverTests(device, recorder, true);
        integerClearTests(device);
        denormalClearTests(device, recorder);
        unitShadowTests(device, recorder);
        storageRefreshTests(device, recorder, false);
        storageRefreshTests(device, recorder, true);
        static_cast<void>(writtenBufferOrderTests(device, recorder));
        writeBackPaddingTests(device, recorder, false);
        writeBackPaddingTests(device, recorder, true);
        splitWriteBackTests(device, recorder);
        writeBackBlockTests(device, recorder);
        singlePassTests(device, recorder, false);
        singlePassTests(device, recorder, true);
        unchangedCpuStampTests(device, recorder);
        keptUnitSettleTests(device, recorder);
        settledBaselineTests(device, recorder);
        importWatchTests(device);
        staleGenerationTests(device, recorder);
        importWindowTests(device, recorder);
        dataWordPositionsTests();
        dataRefreshTests(device, recorder);
        imageMemoTests(device, recorder);
        minLodTests(device, recorder);
        firstLayerViewTests(device, recorder);
        sampledStorageBaseLevelUsageTests(device, recorder);
        singleCubeTests(device, recorder);
        cubeExtraSliceTests(device, recorder);
        atomicViewTests(device, recorder);
        metadataPassTests(device, recorder);
        cmaskPassTests(device, recorder);
        pendingKeyStoreTests(device, recorder);
        shadowedKeyStoreTests(device, recorder);
        sampleDumpTests(device, recorder);
        copiedDirectOrderingTests(device, recorder);
        gpu.unlock();
        hostImportUnmapTests(device, recorder);
        drawSnapshotPatchTests(device);
        drawSnapshotEpochTests(device);
        gpu.lock();
        static_cast<void>(drawSnapshotArenaTests(device, recorder));
        recorder.Activate();
        std::cout << "Recorder read tracking and label tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
