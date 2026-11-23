#include "crash_diagnostics.h"

#include "rebrewu/logger.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#if defined(HAVE_SDL2)
#include <SDL.h>
#endif
#endif

namespace rbrew::crash_diag {

namespace {

// Known VkResult negative error codes (avoiding a Vulkan header dependency here).
constexpr int kVkErrorOutOfHostMemory = -1;
constexpr int kVkErrorOutOfDeviceMemory = -2;
constexpr int kVkErrorInitializationFailed = -3;
constexpr int kVkErrorDeviceLost = -4;
constexpr int kVkErrorExtensionNotPresent = -7;
constexpr int kVkErrorFeatureNotPresent = -8;
constexpr int kVkErrorIncompatibleDriver = -9;

// Known Windows SEH exception codes.
constexpr uint32_t kWinExceptionAccessViolation = 0xC0000005u;
constexpr uint32_t kWinExceptionIllegalInstruction = 0xC000001Du;
constexpr uint32_t kWinExceptionStackOverflow = 0xC00000FDu;

// Fixed-size storage for recorded GPU metadata so crash handlers do not depend on heap state.
struct StaticGpuSnapshot {
    std::atomic<bool> valid{false};
    char device_name[256]{};
    char driver_info[256]{};
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t api_version = 0;
    uint32_t driver_version = 0;
};

StaticGpuSnapshot s_gpu{};
std::mutex s_gpu_mutex;

std::mutex s_config_mutex;
std::string s_diagnostics_dir_override;
// -1 = auto-detect, 0 = forced non-interactive, 1 = forced interactive
std::atomic<int> s_interactive_override{-1};

std::string to_lower_ascii(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char c : input) out.push_back(static_cast<char>(std::tolower(c)));
    return out;
}

std::string basename_only(std::string_view path) {
    const size_t slash = path.find_last_of("/\\");
    return std::string(slash == std::string_view::npos ? path : path.substr(slash + 1));
}

bool env_truthy(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return std::strcmp(value, "0") != 0 && to_lower_ascii(value) != "false" && to_lower_ascii(value) != "off";
}

std::string current_executable_basename() {
#if defined(_WIN32)
    char path[MAX_PATH]{};
    const DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (len > 0 && len < MAX_PATH) return basename_only(path);
#elif defined(__linux__)
    char path[4096]{};
    const ssize_t len = ::readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len > 0) {
        path[len] = '\0';
        return basename_only(path);
    }
#endif
    return {};
}

bool looks_like_test_or_tool_binary(std::string_view exe_name) {
    const std::string lower = to_lower_ascii(exe_name);
    if (lower.empty()) return false;
    if (lower.rfind("gambit_", 0) == 0) return true;
    if (lower.rfind("test_", 0) == 0) return true;
    if (lower.rfind("gx2_replay", 0) == 0) return true;
    if (lower.rfind("splport_analyze", 0) == 0) return true;
    if (lower.rfind("splport_check", 0) == 0) return true;
    if (lower.find("_test") != std::string::npos) return true;
    if (lower.find("_bench") != std::string::npos) return true;
    if (lower.find("tests.") != std::string::npos || lower.ends_with("tests")) return true;
    return false;
}

const char* vendor_name(uint32_t vendor_id) {
    switch (vendor_id) {
    case 0x1002: return "AMD";
    case 0x10DE: return "NVIDIA";
    case 0x8086: return "Intel";
    case 0x1414: return "Microsoft";
    case 0x10005: return "Mesa";
    default: return "Unknown";
    }
}

std::string format_vk_api_version(uint32_t api_version) {
    if (!api_version) return "unknown";
    const uint32_t major = (api_version >> 22) & 0x7Fu;
    const uint32_t minor = (api_version >> 12) & 0x3FFu;
    const uint32_t patch = api_version & 0xFFFu;
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u", major, minor, patch);
    return buf;
}

std::string resolve_diagnostics_base_dir(const std::string& explicit_override) {
    if (!explicit_override.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(explicit_override, ec);
        return explicit_override;
    }
    {
        std::lock_guard<std::mutex> lock(s_config_mutex);
        if (!s_diagnostics_dir_override.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(s_diagnostics_dir_override, ec);
            return s_diagnostics_dir_override;
        }
    }

    std::error_code ec;
    std::string local = "Diagnostics";
    std::filesystem::create_directories(local, ec);
    if (!ec) return local;

#if defined(_WIN32)
    char appdata[MAX_PATH]{};
    if (GetEnvironmentVariableA("LOCALAPPDATA", appdata, sizeof(appdata)) > 0) {
        std::string fallback = std::string(appdata) + "\\Splatoon\\Diagnostics";
        std::filesystem::create_directories(fallback, ec);
        if (!ec) return fallback;
    }
#else
    if (const char* home = std::getenv("HOME"); home && *home) {
        std::string fallback = std::string(home) + "/.local/share/Splatoon/Diagnostics";
        std::filesystem::create_directories(fallback, ec);
        if (!ec) return fallback;
    }
#endif
    return ".";
}

bool contains_ci(std::string_view haystack, std::string_view needle) {
    if (needle.empty() || haystack.size() < needle.size()) return false;
    const std::string h = to_lower_ascii(haystack);
    const std::string n = to_lower_ascii(needle);
    return h.find(n) != std::string::npos;
}

FailureCategory infer_vulkan_category(const CrashContext& ctx) {
    switch (ctx.vk_result) {
    case kVkErrorIncompatibleDriver:
        return FailureCategory::VulkanDriverIncompatible;
    case kVkErrorOutOfDeviceMemory:
    case kVkErrorOutOfHostMemory:
        return FailureCategory::VulkanOutOfMemory;
    case kVkErrorDeviceLost:
        return FailureCategory::VulkanDeviceLost;
    case kVkErrorExtensionNotPresent:
    case kVkErrorFeatureNotPresent:
        return FailureCategory::VulkanExtensionOrFeatureMissing;
    case kVkErrorInitializationFailed:
        return FailureCategory::VulkanInitFailure;
    default:
        break;
    }

    if (ctx.detail.empty()) return FailureCategory::Unknown;

    if (contains_ci(ctx.detail, "vulkan-1.dll") || contains_ci(ctx.detail, "vulkan loader") ||
        contains_ci(ctx.detail, "volkinitialize")) {
        return FailureCategory::VulkanLoaderMissing;
    }
    if (contains_ci(ctx.detail, "vulkan 1.3") || contains_ci(ctx.detail, "api version")) {
        return FailureCategory::VulkanVersionTooOld;
    }
    if (contains_ci(ctx.detail, "VK_ERROR_INCOMPATIBLE_DRIVER") || contains_ci(ctx.detail, "incompatible driver")) {
        return FailureCategory::VulkanDriverIncompatible;
    }
    if (contains_ci(ctx.detail, "VK_ERROR_DEVICE_LOST") || contains_ci(ctx.detail, "device lost")) {
        return FailureCategory::VulkanDeviceLost;
    }
    if (contains_ci(ctx.detail, "VK_ERROR_OUT_OF_DEVICE_MEMORY") || contains_ci(ctx.detail, "out of device memory") ||
        contains_ci(ctx.detail, "VK_ERROR_OUT_OF_HOST_MEMORY")) {
        return FailureCategory::VulkanOutOfMemory;
    }
    if (contains_ci(ctx.detail, "VK_ERROR_EXTENSION_NOT_PRESENT") ||
        contains_ci(ctx.detail, "VK_ERROR_FEATURE_NOT_PRESENT") ||
        contains_ci(ctx.detail, "required device feature")) {
        return FailureCategory::VulkanExtensionOrFeatureMissing;
    }
    if (contains_ci(ctx.detail, "vkCreateInstance") || contains_ci(ctx.detail, "vkCreateDevice") ||
        contains_ci(ctx.detail, "SDL_Vulkan") || contains_ci(ctx.detail, "Vulkan initialization failed")) {
        return FailureCategory::VulkanInitFailure;
    }
    if (contains_ci(ctx.detail, "out of memory") || contains_ci(ctx.detail, "failed to allocate")) {
        return FailureCategory::OutOfMemory;
    }
    return FailureCategory::Unknown;
}

std::string build_dialog_body(const Diagnosis& diag, const std::string& report_path,
                              const std::string& technical_detail) {
    std::ostringstream oss;
    oss << "Splatoon PC Port encountered a fatal error and had to close.\n\n";
    oss << "Likely cause:\n" << diag.probable_cause << "\n\n";
    if (!diag.fix_steps.empty()) {
        oss << "How to fix:\n";
        for (size_t i = 0; i < diag.fix_steps.size(); ++i) {
            oss << "  " << (i + 1) << ". " << diag.fix_steps[i] << "\n";
        }
        oss << "\n";
    }
    if (!technical_detail.empty()) {
        std::string trimmed = technical_detail;
        if (trimmed.size() > 260) {
            trimmed.resize(257);
            trimmed += "...";
        }
        oss << "Technical detail:\n  " << trimmed << "\n\n";
    }
    if (!report_path.empty()) {
        oss << "A full crash report was saved to:\n  " << report_path << "\n";
    }
    return oss.str();
}

} // namespace

bool is_gpu_driver_module(std::string_view module_name) {
    if (module_name.empty()) return false;
    const std::string base = to_lower_ascii(basename_only(module_name));
    if (base.empty()) return false;

    static constexpr std::string_view kExactOrPrefixTokens[] = {
        // AMD Windows & Linux Vulkan / graphics drivers
        "amdvlk64",
        "amdvlk32",
        "amdvlk",
        "aticfx64",
        "aticfx32",
        "atidxx64",
        "atidxx32",
        "amdxc64",
        "amdxn64",
        "amdihk64",
        "libvulkan_radeon",
        "radv",
        "radeonsi",
        // NVIDIA Windows & Linux Vulkan / graphics drivers
        "nvoglv64",
        "nvoglv32",
        "nvwgf2umx",
        "nvwgf2um",
        "nvgpucomp64",
        "libnvidia-glcore",
        "nvidia_icd",
        // Intel Windows & Linux Vulkan / graphics drivers
        "igvk64",
        "igvk32",
        "igxelpicd64",
        "igxelpicd32",
        "igd10iumd64",
        "igd12umd64",
        "igc64",
        "libvulkan_intel",
        // Vulkan loader & Mesa ICDs
        "vulkan-1",
        "libvulkan",
        "mesa",
    };

    for (std::string_view token : kExactOrPrefixTokens) {
        if (base.find(token) != std::string::npos) return true;
    }

    // Match Intel ANV explicitly as a standalone token / library name so we do
    // not false-positive on unrelated words containing "anv" (e.g. "canvas").
    if (base == "anv" || base.rfind("anv.", 0) == 0 || base.rfind("anv_", 0) == 0 ||
        base.find("_anv") != std::string::npos || base.find("-anv") != std::string::npos ||
        base.find("anv_icd") != std::string::npos) {
        return true;
    }

    return false;
}

std::string find_gpu_driver_in_backtrace(std::string_view backtrace) {
    if (backtrace.empty()) return {};
    size_t pos = 0;
    while (pos < backtrace.size()) {
        size_t end = backtrace.find_first_of(" \t\r\n+():[]<>", pos);
        if (end == std::string_view::npos) end = backtrace.size();
        if (end > pos) {
            const std::string_view token = backtrace.substr(pos, end - pos);
            if (is_gpu_driver_module(token)) return basename_only(token);
        }
        pos = end + 1;
    }
    return {};
}

const char* category_name(FailureCategory category) {
    switch (category) {
    case FailureCategory::GpuDriverCrash: return "GPU Driver / Vulkan Crash";
    case FailureCategory::VulkanLoaderMissing: return "Vulkan Runtime Missing";
    case FailureCategory::VulkanVersionTooOld: return "Vulkan 1.3 Required (Driver Too Old)";
    case FailureCategory::VulkanDriverIncompatible: return "Incompatible Vulkan Driver";
    case FailureCategory::VulkanExtensionOrFeatureMissing: return "Missing Vulkan Feature or Extension";
    case FailureCategory::VulkanOutOfMemory: return "Out of GPU Video Memory (VRAM)";
    case FailureCategory::VulkanDeviceLost: return "GPU Driver Reset / Device Lost";
    case FailureCategory::VulkanInitFailure: return "Vulkan Initialization Failed";
    case FailureCategory::IllegalInstruction: return "Unsupported CPU Instruction";
    case FailureCategory::AccessViolation: return "Memory Access Violation";
    case FailureCategory::StackOverflow: return "Stack Overflow";
    case FailureCategory::OutOfMemory: return "Out of System Memory";
    case FailureCategory::AbortOrTerminate: return "Fatal Runtime Error";
    case FailureCategory::StartupError: return "Startup Failure";
    case FailureCategory::Unknown:
    default:
        return "Unexpected Crash";
    }
}

Diagnosis diagnose(const CrashContext& ctx) {
    Diagnosis diag{};

    // 1. Resolve effective category from explicit category, VkResult/detail, faulting module,
    //    backtrace, exception code, or POSIX signal.
    FailureCategory effective = ctx.category;
    std::string gpu_module;
    if (is_gpu_driver_module(ctx.faulting_module)) {
        gpu_module = basename_only(ctx.faulting_module);
    } else if (!ctx.backtrace.empty()) {
        gpu_module = find_gpu_driver_in_backtrace(ctx.backtrace);
    }

    if (effective == FailureCategory::Unknown) {
        effective = infer_vulkan_category(ctx);
    }

    // Check CPU illegal instruction before generic driver/access violation checks.
    const bool is_illegal_insn =
        effective == FailureCategory::IllegalInstruction || ctx.seh_code == kWinExceptionIllegalInstruction ||
#ifdef SIGILL
        ctx.signal_number == SIGILL;
#else
        ctx.signal_number == 4;
#endif

    const bool is_stack_overflow =
        effective == FailureCategory::StackOverflow || ctx.seh_code == kWinExceptionStackOverflow;

    const bool is_access_violation =
        effective == FailureCategory::AccessViolation || ctx.seh_code == kWinExceptionAccessViolation ||
#ifdef SIGSEGV
        ctx.signal_number == SIGSEGV ||
#else
        ctx.signal_number == 11 ||
#endif
#ifdef SIGBUS
        ctx.signal_number == SIGBUS;
#else
        ctx.signal_number == 7;
#endif

    if (is_illegal_insn) {
        effective = FailureCategory::IllegalInstruction;
    } else if ((effective == FailureCategory::Unknown || effective == FailureCategory::AccessViolation ||
                effective == FailureCategory::AbortOrTerminate) &&
               !gpu_module.empty()) {
        effective = FailureCategory::GpuDriverCrash;
    } else if (is_stack_overflow) {
        effective = FailureCategory::StackOverflow;
    } else if (is_access_violation) {
        effective = FailureCategory::AccessViolation;
    } else if (effective == FailureCategory::Unknown &&
               (ctx.signal_number == SIGABRT)) {
        effective = FailureCategory::AbortOrTerminate;
    }

    diag.category = effective;

    switch (effective) {
    case FailureCategory::GpuDriverCrash: {
        const std::string mod_label = gpu_module.empty() ? "your graphics driver" : ("'" + gpu_module + "'");
        diag.title = "Splatoon PC Port — Graphics Driver Crash";
        diag.probable_cause =
            "The crash occurred inside " + mod_label +
            " (your GPU Vulkan driver). This is usually caused by an outdated or corrupted GPU driver, "
            "a stale shader cache, or a third-party overlay hooking Vulkan.";
        diag.fix_steps = {
            "Update your graphics driver to the latest official release from AMD, NVIDIA, or Intel, then reboot.",
            "Delete the 'vulkan_cache' folder next to the game executable so shaders rebuild cleanly.",
            "Temporarily disable third-party overlays (RivaTuner/MSI Afterburner, Overwolf, ReShade, Discord overlay).",
            "If Realistic shaders are enabled, set shader_quality=0 (Original preset) in settings.ini and retry.",
        };
        break;
    }

    case FailureCategory::VulkanLoaderMissing:
        diag.title = "Splatoon PC Port — Vulkan Driver Missing";
        diag.probable_cause =
            "The Vulkan graphics runtime (vulkan-1.dll / libvulkan) could not be loaded. "
            "Your system either does not have a Vulkan-capable GPU driver installed or the driver installation is broken.";
        diag.fix_steps = {
            "Install the latest official graphics driver for your GPU from AMD, NVIDIA, or Intel (clean install recommended).",
            "On Linux, install the Vulkan loader and your GPU's Mesa/NVIDIA Vulkan ICD package (e.g. mesa-vulkan-drivers).",
            "Reboot your PC after installing the driver and launch the game again.",
        };
        break;

    case FailureCategory::VulkanVersionTooOld:
    case FailureCategory::VulkanDriverIncompatible:
        diag.title = "Splatoon PC Port — Graphics Driver Too Old (Vulkan 1.3 Required)";
        diag.probable_cause =
            "Your graphics driver or GPU does not support Vulkan 1.3 (VK_ERROR_INCOMPATIBLE_DRIVER / older Vulkan API). "
            "Outdated GPU drivers are the most common cause of this error.";
        diag.fix_steps = {
            "Download and install the latest graphics driver from AMD, NVIDIA, or Intel (Windows Update drivers are often too old).",
            "On laptops with dual GPUs, make sure the game is set to use the newer/discrete GPU in Windows Graphics Settings.",
            "Verify your GPU supports Vulkan 1.3 (NVIDIA Maxwell/GTX 900+, AMD GCN4/RDNA+, Intel Gen9+/Arc, or Mesa 22+).",
        };
        break;

    case FailureCategory::VulkanExtensionOrFeatureMissing:
        diag.title = "Splatoon PC Port — Missing Vulkan GPU Feature";
        diag.probable_cause =
            "Your GPU driver is missing a required Vulkan 1.3 feature or extension needed by the GX2 renderer "
            "(such as dynamicRendering, synchronization2, timelineSemaphore, or BC texture compression).";
        diag.fix_steps = {
            "Update your graphics driver to the latest official version from AMD, NVIDIA, or Intel.",
            "On laptops with hybrid graphics, switch the game to your high-performance GPU in OS graphics settings.",
            "If the error persists after updating, attach the saved crash_report.txt when reporting an issue.",
        };
        break;

    case FailureCategory::VulkanOutOfMemory:
        diag.title = "Splatoon PC Port — Out of GPU Video Memory (VRAM)";
        diag.probable_cause =
            "The GPU ran out of video memory (VK_ERROR_OUT_OF_DEVICE_MEMORY / VK_ERROR_OUT_OF_HOST_MEMORY) "
            "while allocating render targets, textures, or buffers.";
        diag.fix_steps = {
            "Close VRAM-heavy background apps (web browsers, video editors, recording tools, other games).",
            "Lower the internal render resolution to 720p (1280x720) in settings.ini or via RBREW_RENDER_RESOLUTION=720p.",
            "Set Graphics preset to Original (shader_quality=0) in settings.ini to reduce render-target VRAM usage.",
            "Update your graphics driver and ensure Windows/Linux has sufficient pagefile/swap space.",
        };
        break;

    case FailureCategory::VulkanDeviceLost:
        diag.title = "Splatoon PC Port — GPU Driver Reset (Device Lost)";
        diag.probable_cause =
            "The GPU stopped responding and the operating system reset the graphics driver (VK_ERROR_DEVICE_LOST / TDR). "
            "This is commonly caused by an outdated driver, corrupted shader cache, GPU overclock instability, or heavy shader load.";
        diag.fix_steps = {
            "Update your graphics driver to the latest official release from AMD, NVIDIA, or Intel.",
            "Delete the 'vulkan_cache' folder next to the game executable so all GPU pipelines recompile cleanly.",
            "Disable any GPU overclock/undervolt and close third-party overlay/hook apps (RTSS, Overwolf, ReShade).",
            "Try lowering Graphics preset to Original (shader_quality=0) and resolution to 720p in settings.ini.",
        };
        break;

    case FailureCategory::VulkanInitFailure:
        diag.title = "Splatoon PC Port — Vulkan Initialization Failed";
        diag.probable_cause =
            "The Vulkan renderer failed to initialize on this machine. This usually indicates an outdated or broken "
            "graphics driver, an incompatible Vulkan layer, or an unsupported GPU.";
        diag.fix_steps = {
            "Install the latest official graphics driver for your GPU (AMD, NVIDIA, or Intel) and reboot.",
            "Delete the 'vulkan_cache' folder next to the game executable.",
            "Disable third-party Vulkan layers/overlays (Overwolf, OBS Vulkan hook, RivaTuner, ReShade).",
            "Attach the saved crash_report.txt if the problem continues.",
        };
        break;

    case FailureCategory::IllegalInstruction:
        diag.title = "Splatoon PC Port — Unsupported CPU Instruction";
        diag.probable_cause =
            "Your CPU encountered an unsupported machine instruction (EXCEPTION_ILLEGAL_INSTRUCTION / SIGILL). "
            "This typically happens when running an AVX2-compiled build on an older x86-64 CPU that does not support AVX2.";
        diag.fix_steps = {
            "Use a portable build compiled without AVX2 (CMake option -DRBREW_ENABLE_AVX2=OFF, which is the default).",
            "If you built the game yourself, reconfigure CMake with -DRBREW_ENABLE_AVX2=OFF and rebuild.",
            "Check the saved crash_report.txt for the faulting module and offset if this occurred inside a driver DLL.",
        };
        break;

    case FailureCategory::StackOverflow:
        diag.title = "Splatoon PC Port — Stack Overflow Crash";
        diag.probable_cause =
            "A thread exhausted its call stack (EXCEPTION_STACK_OVERFLOW). This may be triggered by a GPU driver bug "
            "during shader compilation or an unexpected recursion in the runtime.";
        diag.fix_steps = {
            "Delete the 'vulkan_cache' folder next to the game executable and relaunch.",
            "Update your graphics driver to the latest official version.",
            "Share the saved crash_report.txt when reporting this crash.",
        };
        break;

    case FailureCategory::OutOfMemory:
        diag.title = "Splatoon PC Port — Out of System Memory";
        diag.probable_cause =
            "The system could not allocate required memory for the game runtime or guest memory arena.";
        diag.fix_steps = {
            "Close memory-intensive background programs (browsers, editors, virtual machines).",
            "Ensure at least 16 GB of RAM (or sufficient system pagefile/swap) is available.",
            "Relaunch the game and attach crash_report.txt if the issue persists.",
        };
        break;

    case FailureCategory::AccessViolation: {
        const std::string mod = ctx.faulting_module.empty() ? "unknown module" : basename_only(ctx.faulting_module);
        diag.title = "Splatoon PC Port — Crash (Access Violation)";
        diag.probable_cause =
            "The game crashed due to an invalid memory access in '" + mod +
            "'. This can be caused by outdated GPU drivers, a corrupted shader cache, or an incomplete game dump.";
        diag.fix_steps = {
            "Update your graphics driver to the latest official release from AMD, NVIDIA, or Intel.",
            "Delete the 'vulkan_cache' folder next to the game executable and relaunch.",
            "Verify your game files (content/ and meta/) are intact and not corrupted.",
            "Share the saved crash_report.txt when reporting the issue.",
        };
        break;
    }

    case FailureCategory::AbortOrTerminate:
        diag.title = "Splatoon PC Port — Fatal Runtime Error";
        diag.probable_cause =
            "The game runtime encountered an unrecoverable internal error or unhandled C++ exception and terminated.";
        diag.fix_steps = {
            "Update your graphics driver and delete the 'vulkan_cache' folder next to the game executable.",
            "Check that your game content, meta/meta.xml, and settings.ini files are valid.",
            "Share the saved crash_report.txt when reporting the issue.",
        };
        break;

    case FailureCategory::StartupError:
    case FailureCategory::Unknown:
    default:
        diag.title = "Splatoon PC Port — Unexpected Crash";
        diag.probable_cause =
            "The game encountered an unexpected fatal error and had to exit.";
        diag.fix_steps = {
            "Update your graphics driver to the latest official release.",
            "Delete the 'vulkan_cache' folder next to the game executable and try again.",
            "Share the saved crash_report.txt when reporting the issue.",
        };
        break;
    }

    return diag;
}

void record_gpu_info(const char* device_name, uint32_t vendor_id, uint32_t device_id,
                     uint32_t api_version, uint32_t driver_version,
                     const char* driver_info) {
    std::lock_guard<std::mutex> lock(s_gpu_mutex);
    std::snprintf(s_gpu.device_name, sizeof(s_gpu.device_name), "%s", device_name ? device_name : "");
    std::snprintf(s_gpu.driver_info, sizeof(s_gpu.driver_info), "%s", driver_info ? driver_info : "");
    s_gpu.vendor_id = vendor_id;
    s_gpu.device_id = device_id;
    s_gpu.api_version = api_version;
    s_gpu.driver_version = driver_version;
    s_gpu.valid.store(true, std::memory_order_release);
}

GpuInfo current_gpu_info() {
    GpuInfo info{};
    if (!s_gpu.valid.load(std::memory_order_acquire)) return info;
    std::lock_guard<std::mutex> lock(s_gpu_mutex);
    info.device_name = s_gpu.device_name;
    info.driver_info = s_gpu.driver_info;
    info.vendor_id = s_gpu.vendor_id;
    info.device_id = s_gpu.device_id;
    info.api_version = s_gpu.api_version;
    info.driver_version = s_gpu.driver_version;
    return info;
}

void clear_gpu_info() {
    std::lock_guard<std::mutex> lock(s_gpu_mutex);
    s_gpu.valid.store(false, std::memory_order_release);
    s_gpu.device_name[0] = '\0';
    s_gpu.driver_info[0] = '\0';
    s_gpu.vendor_id = 0;
    s_gpu.device_id = 0;
    s_gpu.api_version = 0;
    s_gpu.driver_version = 0;
}

void set_diagnostics_dir_override(const std::string& dir) {
    std::lock_guard<std::mutex> lock(s_config_mutex);
    s_diagnostics_dir_override = dir;
}

void set_interactive_game_process(bool interactive) {
    s_interactive_override.store(interactive ? 1 : 0, std::memory_order_relaxed);
}

void reset_interactive_game_process_override() {
    s_interactive_override.store(-1, std::memory_order_relaxed);
}

bool is_interactive_game_process() {
    const int forced = s_interactive_override.load(std::memory_order_relaxed);
    if (forced >= 0) return forced != 0;
    return !looks_like_test_or_tool_binary(current_executable_basename());
}

bool should_show_dialog() {
    if (!is_interactive_game_process()) return false;
    if (env_truthy("RBREW_HEADLESS") || env_truthy("RBREW_NO_CRASH_DIALOG")) return false;
    if (env_truthy("CI") || env_truthy("GITHUB_ACTIONS")) return false;
    if (const char* test_win = std::getenv("RBREW_TEST_WINDOW"); test_win && *test_win) return false;
#if !defined(_WIN32) && !defined(__APPLE__)
    const char* display = std::getenv("DISPLAY");
    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    if ((!display || !*display) && (!wayland || !*wayland)) return false;
#endif
    return true;
}

std::string make_crash_timestamp() {
    char buf[64]{};
    const std::time_t now = std::time(nullptr);
    std::tm tm_local{};
#if defined(_WIN32)
    localtime_s(&tm_local, &now);
#else
    localtime_r(&now, &tm_local);
#endif
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d_%02d-%02d-%02d",
                  tm_local.tm_year + 1900, tm_local.tm_mon + 1, tm_local.tm_mday,
                  tm_local.tm_hour, tm_local.tm_min, tm_local.tm_sec);
    return buf;
}

std::string format_crash_report(const CrashContext& ctx, const Diagnosis& diag,
                                const std::string& timestamp_str) {
    std::ostringstream oss;
    oss << "============================================================\n";
    oss << "SPLATOON 1 PC PORT - CRASH & FAILURE DIAGNOSTIC REPORT\n";
    oss << "============================================================\n";
    oss << "Timestamp:        " << (timestamp_str.empty() ? make_crash_timestamp() : timestamp_str) << "\n";
    oss << "Category:         " << category_name(diag.category) << "\n";
    if (ctx.seh_code != 0) {
        char code_buf[32];
        std::snprintf(code_buf, sizeof(code_buf), "0x%08X", ctx.seh_code);
        oss << "Exception Code:   " << code_buf << "\n";
    }
    if (ctx.signal_number != 0) {
        oss << "Signal:           " << ctx.signal_number << "\n";
    }
    if (ctx.vk_result != 0) {
        oss << "VkResult:         " << ctx.vk_result << "\n";
    }
    if (ctx.fault_address != 0) {
        char addr_buf[32];
        std::snprintf(addr_buf, sizeof(addr_buf), "0x%llX", static_cast<unsigned long long>(ctx.fault_address));
        oss << "Fault Address:    " << addr_buf << "\n";
    }
    if (!ctx.faulting_module.empty()) {
        oss << "Faulting Module:  " << ctx.faulting_module << "\n";
    }
    if (ctx.frame_number != 0) {
        oss << "Frame Number:     " << ctx.frame_number << "\n";
    }

    oss << "\nGPU / Driver Info:\n";
    if (s_gpu.valid.load(std::memory_order_acquire)) {
        char hex_buf[96];
        std::snprintf(hex_buf, sizeof(hex_buf), "0x%04X (%s) / Device 0x%04X",
                      s_gpu.vendor_id, vendor_name(s_gpu.vendor_id), s_gpu.device_id);
        oss << "  Device:         " << s_gpu.device_name << "\n";
        oss << "  Vendor/ID:      " << hex_buf << "\n";
        oss << "  Vulkan API:     " << format_vk_api_version(s_gpu.api_version) << "\n";
        std::snprintf(hex_buf, sizeof(hex_buf), "0x%08X", s_gpu.driver_version);
        oss << "  Driver Version: " << hex_buf << "\n";
        if (s_gpu.driver_info[0] != '\0') {
            oss << "  Driver Info:    " << s_gpu.driver_info << "\n";
        }
    } else {
        oss << "  (not yet initialized)\n";
    }

    oss << "\nDiagnosed Probable Cause:\n";
    oss << "  " << diag.probable_cause << "\n";

    if (!diag.fix_steps.empty()) {
        oss << "\nRecommended Fix Steps:\n";
        for (size_t i = 0; i < diag.fix_steps.size(); ++i) {
            oss << "  " << (i + 1) << ". " << diag.fix_steps[i] << "\n";
        }
    }

    if (!ctx.detail.empty()) {
        oss << "\nTechnical Detail:\n";
        oss << "  " << ctx.detail << "\n";
    }

    if (!ctx.backtrace.empty()) {
        oss << "\nBacktrace:\n";
        oss << ctx.backtrace;
        if (ctx.backtrace.back() != '\n') oss << "\n";
    }

    return oss.str();
}

std::string write_crash_report_to_dir(const CrashContext& ctx, const Diagnosis& diag,
                                      const std::string& crash_dir,
                                      const std::string& timestamp_str) {
    if (crash_dir.empty()) return {};
    std::error_code ec;
    std::filesystem::create_directories(crash_dir, ec);
    const std::filesystem::path report_path = std::filesystem::path(crash_dir) / "crash_report.txt";
    const std::string body = format_crash_report(ctx, diag, timestamp_str);
    std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
    if (!out) return {};
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    out.flush();
    if (!out.good()) return {};
    return report_path.string();
}

std::string write_crash_report(const CrashContext& ctx, const Diagnosis& diag,
                               const std::string& base_dir_override) {
    const std::string ts = make_crash_timestamp();
    const std::string base = resolve_diagnostics_base_dir(base_dir_override);
    std::filesystem::path crash_dir = std::filesystem::path(base) / (ts + "_crash");
    std::error_code ec;
    if (std::filesystem::exists(crash_dir, ec)) {
        for (int suffix = 1; suffix <= 99; ++suffix) {
            std::filesystem::path candidate =
                std::filesystem::path(base) / (ts + "_crash_" + std::to_string(suffix));
            if (!std::filesystem::exists(candidate, ec)) {
                crash_dir = std::move(candidate);
                break;
            }
        }
    }
    return write_crash_report_to_dir(ctx, diag, crash_dir.string(), ts);
}

bool show_error_dialog(const Diagnosis& diag, const std::string& report_path,
                       const std::string& technical_detail) {
    if (!should_show_dialog()) return false;
    const std::string title = diag.title.empty() ? "Splatoon PC Port — Error" : diag.title;
    const std::string body = build_dialog_body(diag, report_path, technical_detail);

#if defined(_WIN32)
    auto to_wide = [](const std::string& utf8) -> std::wstring {
        if (utf8.empty()) return {};
        const int needed = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), int(utf8.size()), nullptr, 0);
        if (needed <= 0) return {};
        std::wstring wide(static_cast<size_t>(needed), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), int(utf8.size()), wide.data(), needed);
        return wide;
    };
    const std::wstring wtitle = to_wide(title);
    const std::wstring wbody = to_wide(body);
    MessageBoxW(nullptr, wbody.c_str(), wtitle.c_str(), MB_OK | MB_ICONERROR | MB_TOPMOST | MB_SETFOREGROUND);
    return true;
#elif defined(HAVE_SDL2)
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, title.c_str(), body.c_str(), nullptr);
    return true;
#else
    (void)title;
    (void)body;
    return false;
#endif
}

std::string report_fatal_error(FailureCategory category, const char* detail,
                               int vk_result, const char* faulting_module,
                               bool allow_dialog) {
    CrashContext ctx{};
    ctx.category = category;
    ctx.vk_result = vk_result;
    if (detail) ctx.detail = detail;
    if (faulting_module) ctx.faulting_module = faulting_module;

    const Diagnosis diag = diagnose(ctx);
    const std::string report_path = write_crash_report(ctx, diag);

    RBREW_LOG_FATAL("crash", "%s: %s (report: %s)\n", category_name(diag.category),
                    diag.probable_cause.c_str(), report_path.empty() ? "<unavailable>" : report_path.c_str());

    if (allow_dialog) {
        show_error_dialog(diag, report_path, ctx.detail);
    }
    return report_path;
}

} // namespace rbrew::crash_diag
