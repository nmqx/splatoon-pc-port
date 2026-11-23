#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rbrew::crash_diag {

enum class FailureCategory : uint8_t {
    Unknown = 0,
    GpuDriverCrash,
    VulkanLoaderMissing,
    VulkanVersionTooOld,
    VulkanDriverIncompatible,
    VulkanExtensionOrFeatureMissing,
    VulkanOutOfMemory,
    VulkanDeviceLost,
    VulkanInitFailure,
    IllegalInstruction,
    AccessViolation,
    StackOverflow,
    OutOfMemory,
    AbortOrTerminate,
    StartupError,
};

struct GpuInfo {
    std::string device_name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t api_version = 0;
    uint32_t driver_version = 0;
    std::string driver_info;
};

struct CrashContext {
    FailureCategory category = FailureCategory::Unknown;
    uint32_t seh_code = 0;       // Windows SEH code (e.g. 0xC0000005, 0xC000001D, 0xC00000FD)
    int signal_number = 0;       // POSIX signal (e.g. SIGSEGV, SIGILL, SIGBUS, SIGFPE, SIGABRT)
    int vk_result = 0;           // Optional VkResult integer (e.g. -2 OOM, -4 DEVICE_LOST, -9 INCOMPATIBLE_DRIVER)
    uintptr_t fault_address = 0;
    std::string faulting_module; // Basename or path of faulting DLL/.so (e.g. "amdvlk64.dll")
    std::string detail;          // Technical error message / context
    std::string backtrace;       // Optional formatted stack trace
    uint32_t frame_number = 0;
};

struct Diagnosis {
    FailureCategory category = FailureCategory::Unknown;
    std::string title;
    std::string probable_cause;
    std::vector<std::string> fix_steps;
};

// Returns true when `module_name` identifies a known GPU driver or Vulkan ICD/loader library.
bool is_gpu_driver_module(std::string_view module_name);

// Scans a formatted backtrace for a known GPU driver / Vulkan ICD module name.
// Returns the first matching module token or empty string if none is found.
std::string find_gpu_driver_in_backtrace(std::string_view backtrace);

// Maps a FailureCategory enum to a stable human-readable label.
const char* category_name(FailureCategory category);

// Classifies a crash or fatal error into a plain-English diagnosis with concrete fix steps.
Diagnosis diagnose(const CrashContext& ctx);

// Records active GPU & driver metadata once Vulkan selects a physical device.
// Thread-safe; stored in fixed static buffers so crash handlers can read it safely.
void record_gpu_info(const char* device_name, uint32_t vendor_id, uint32_t device_id,
                     uint32_t api_version, uint32_t driver_version,
                     const char* driver_info = nullptr);
GpuInfo current_gpu_info();
void clear_gpu_info();

// Overrides the base Diagnostics/ directory (used by unit tests). Pass empty string to reset.
void set_diagnostics_dir_override(const std::string& dir);

// Controls whether this process may show a blocking native error dialog.
// Automated test binaries, headless runs (RBREW_HEADLESS=1 / RBREW_NO_CRASH_DIALOG=1),
// and CI environments always suppress dialogs.
void set_interactive_game_process(bool interactive);
void reset_interactive_game_process_override();
bool is_interactive_game_process();
bool should_show_dialog();

// Formats a local timestamp string suitable for Diagnostics/<timestamp>_crash folders.
std::string make_crash_timestamp();

// Formats a complete human-readable crash_report.txt body.
std::string format_crash_report(const CrashContext& ctx, const Diagnosis& diag,
                                const std::string& timestamp_str);

// Writes crash_report.txt into `crash_dir` and returns the full file path (or empty on error).
std::string write_crash_report_to_dir(const CrashContext& ctx, const Diagnosis& diag,
                                      const std::string& crash_dir,
                                      const std::string& timestamp_str);

// Creates `<base_dir>/<timestamp>_crash/crash_report.txt` and returns its path (or empty on error).
std::string write_crash_report(const CrashContext& ctx, const Diagnosis& diag,
                               const std::string& base_dir_override = {});

// Shows a native error popup (Win32 MessageBoxW on Windows, SDL_ShowSimpleMessageBox elsewhere)
// ONLY when should_show_dialog() is true. Returns true if displayed, false if suppressed.
bool show_error_dialog(const Diagnosis& diag, const std::string& report_path,
                       const std::string& technical_detail = {});

// Convenience entry point for fatal startup / Vulkan / runtime errors (e.g. from gx2_vk.cpp).
// Diagnoses the failure, logs via RBREW_LOG_FATAL, writes Diagnostics/<timestamp>_crash/crash_report.txt,
// and shows the native popup if interactive and `allow_dialog` is true.
// Returns the saved report path (empty if report creation failed).
std::string report_fatal_error(FailureCategory category, const char* detail,
                               int vk_result = 0, const char* faulting_module = nullptr,
                               bool allow_dialog = true);

} // namespace rbrew::crash_diag
