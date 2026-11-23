#include "runtime/crash_diagnostics.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "failed to open crash report file");
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

void set_env_var(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value && *value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

void test_gpu_driver_module_detection() {
    using rbrew::crash_diag::find_gpu_driver_in_backtrace;
    using rbrew::crash_diag::is_gpu_driver_module;

    const char* gpu_modules[] = {
        "amdvlk64.dll",
        "C:\\Windows\\System32\\DriverStore\\amdvlk32.dll",
        "aticfx64.dll",
        "atidxx64.dll",
        "nvoglv64.dll",
        "nvwgf2umx.dll",
        "igvk64.dll",
        "igxelpicd64.dll",
        "vulkan-1.dll",
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so",
        "/usr/lib/x86_64-linux-gnu/libvulkan_intel.so",
        "mesa_vulkan_icd.so",
        "radv_icd.x86_64.json",
        "anv_icd.x86_64.json",
    };
    for (const char* mod : gpu_modules) {
        require(is_gpu_driver_module(mod), "expected GPU/Vulkan driver module to be detected");
    }

    const char* non_gpu_modules[] = {
        "",
        "Splatoon.exe",
        "kernel32.dll",
        "ntdll.dll",
        "ucrtbase.dll",
        "libc.so.6",
        "libcanvas.so",
    };
    for (const char* mod : non_gpu_modules) {
        require(!is_gpu_driver_module(mod), "non-GPU module should not be classified as GPU driver");
    }

    const std::string bt = "  #00 amdvlk64.dll+0x12345\n  #01 Splatoon.exe+0x9999\n";
    require(find_gpu_driver_in_backtrace(bt) == "amdvlk64.dll", "backtrace GPU driver scan failed");
}

void test_diagnosis_classification() {
    using namespace rbrew::crash_diag;

    // 1. Outdated / crashing GPU driver DLL (SEH access violation inside amdvlk64.dll)
    {
        CrashContext ctx{};
        ctx.seh_code = 0xC0000005u;
        ctx.faulting_module = "C:\\Windows\\System32\\amdvlk64.dll";
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::GpuDriverCrash, "amdvlk64 crash should classify as GpuDriverCrash");
        require(contains(d.probable_cause, "amdvlk64.dll"), "cause should name faulting GPU driver DLL");
        require(!d.fix_steps.empty() && contains(d.fix_steps[0], "Update your graphics driver"),
                "GPU crash should advise updating graphics driver");
    }

    // 2. GPU driver crash detected via backtrace even when faulting module is empty
    {
        CrashContext ctx{};
        ctx.seh_code = 0xC0000005u;
        ctx.backtrace = "  #00 nvoglv64.dll+0x44A0\n  #01 Splatoon.exe+0x100\n";
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::GpuDriverCrash, "backtrace nvoglv64 should classify as GpuDriverCrash");
        require(contains(d.probable_cause, "nvoglv64.dll"), "cause should name backtrace GPU driver DLL");
    }

    // 3. VK_ERROR_INCOMPATIBLE_DRIVER (-9)
    {
        CrashContext ctx{};
        ctx.vk_result = -9;
        ctx.detail = "vkCreateInstance failed (-9)";
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::VulkanDriverIncompatible,
                "VK_ERROR_INCOMPATIBLE_DRIVER should classify as VulkanDriverIncompatible");
        require(contains(d.probable_cause, "Vulkan 1.3"), "cause should mention Vulkan 1.3");
    }

    // 4. Vulkan loader missing
    {
        CrashContext ctx{};
        ctx.detail = "FATAL: the Vulkan loader (vulkan-1.dll) is not available";
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::VulkanLoaderMissing, "missing vulkan-1.dll should be detected");
    }

    // 5. Vulkan version < 1.3
    {
        CrashContext ctx{};
        ctx.detail = "FATAL: Vulkan 1.3 is required (loader reports 1.2)";
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::VulkanVersionTooOld, "Vulkan < 1.3 should classify as VulkanVersionTooOld");
    }

    // 6. VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)
    {
        CrashContext ctx{};
        ctx.vk_result = -2;
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::VulkanOutOfMemory, "VK_ERROR_OUT_OF_DEVICE_MEMORY classification failed");
        require(contains(d.probable_cause, "video memory"), "OOM cause should explain VRAM exhaustion");
    }

    // 7. VK_ERROR_DEVICE_LOST (-4)
    {
        CrashContext ctx{};
        ctx.vk_result = -4;
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::VulkanDeviceLost, "VK_ERROR_DEVICE_LOST classification failed");
        require(contains(d.probable_cause, "TDR"), "device lost should mention TDR/reset");
    }

    // 8. VK_ERROR_EXTENSION_NOT_PRESENT (-7) / VK_ERROR_FEATURE_NOT_PRESENT (-8)
    {
        CrashContext ctx{};
        ctx.vk_result = -7;
        require(diagnose(ctx).category == FailureCategory::VulkanExtensionOrFeatureMissing,
                "VK_ERROR_EXTENSION_NOT_PRESENT classification failed");
        ctx.vk_result = -8;
        require(diagnose(ctx).category == FailureCategory::VulkanExtensionOrFeatureMissing,
                "VK_ERROR_FEATURE_NOT_PRESENT classification failed");
    }

    // 9. EXCEPTION_ILLEGAL_INSTRUCTION (0xC000001D) / SIGILL
    {
        CrashContext ctx{};
        ctx.seh_code = 0xC000001Du;
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::IllegalInstruction, "0xC000001D should classify as IllegalInstruction");
        require(contains(d.probable_cause, "AVX2"), "illegal instruction should mention AVX2");
        require(!d.fix_steps.empty() && contains(d.fix_steps[0], "RBREW_ENABLE_AVX2=OFF"),
                "illegal instruction should advise portable non-AVX2 build");
    }

    // 10. Stack overflow (0xC00000FD) and plain access violation (0xC0000005)
    {
        CrashContext ctx{};
        ctx.seh_code = 0xC00000FDu;
        require(diagnose(ctx).category == FailureCategory::StackOverflow, "0xC00000FD should classify as StackOverflow");

        ctx = {};
        ctx.seh_code = 0xC0000005u;
        ctx.faulting_module = "Splatoon.exe";
        const Diagnosis d = diagnose(ctx);
        require(d.category == FailureCategory::AccessViolation, "Splatoon.exe 0xC0000005 should be AccessViolation");
        require(contains(d.probable_cause, "Splatoon.exe"), "cause should mention faulting module");
    }
}

void test_dialog_suppression_and_report_writing() {
    using namespace rbrew::crash_diag;

    // Test binary must never allow a blocking GUI popup by default.
    reset_interactive_game_process_override();
    require(!is_interactive_game_process(), "test binary must not be treated as interactive game process");
    require(!should_show_dialog(), "dialog must be suppressed in test binary");

    // Even if interactive is forced true, RBREW_HEADLESS=1 / RBREW_NO_CRASH_DIALOG=1 must suppress dialogs.
    set_interactive_game_process(true);
    set_env_var("RBREW_NO_CRASH_DIALOG", "1");
    require(!should_show_dialog(), "RBREW_NO_CRASH_DIALOG=1 must suppress dialog");
    set_env_var("RBREW_NO_CRASH_DIALOG", nullptr);
    set_env_var("RBREW_HEADLESS", "1");
    require(!should_show_dialog(), "RBREW_HEADLESS=1 must suppress dialog");
    set_env_var("RBREW_HEADLESS", nullptr);
    set_interactive_game_process(false);

    // Record GPU info and write a crash report into an isolated temp Diagnostics directory.
    record_gpu_info("AMD Radeon 890M Graphics", 0x1002u, 0x150Eu, (1u << 22) | (3u << 12) | 292u, 0x00800150u,
                    "AMD proprietary driver 24.30.22.03");
    const GpuInfo info = current_gpu_info();
    require(info.device_name == "AMD Radeon 890M Graphics" && info.vendor_id == 0x1002u,
            "record_gpu_info round-trip failed");

    const std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "rbrew_crash_diag_test";
    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);
    set_diagnostics_dir_override(temp_dir.string());

    const std::string report_path =
        report_fatal_error(FailureCategory::VulkanDeviceLost, "vkQueueSubmit2 failed (-4)", -4, "amdvlk64.dll", true);
    require(!report_path.empty(), "report_fatal_error should write crash_report.txt");
    require(std::filesystem::exists(report_path), "crash_report.txt must exist on disk");

    const std::string body = read_file(report_path);
    require(contains(body, "AMD Radeon 890M Graphics"), "report must include recorded GPU device");
    require(contains(body, "0x1002 (AMD)"), "report must include GPU vendor");
    require(contains(body, "1.3.292"), "report must include decoded Vulkan API version");
    require(contains(body, "VkResult:         -4"), "report must include VkResult");
    require(contains(body, "Faulting Module:  amdvlk64.dll"), "report must include faulting module");
    require(contains(body, "Diagnosed Probable Cause:"), "report must include diagnosed probable cause");
    require(contains(body, "Recommended Fix Steps:"), "report must include recommended fix steps");
    require(contains(body, "vulkan_cache"), "report must include concrete fix advice");

    // Verify show_error_dialog returns false without popping any GUI dialog.
    CrashContext ctx{};
    ctx.seh_code = 0xC000001Du;
    const Diagnosis diag = diagnose(ctx);
    require(!show_error_dialog(diag, report_path, "test"), "show_error_dialog must return false when suppressed");

    // Cleanup temp directory and state.
    set_diagnostics_dir_override("");
    clear_gpu_info();
    reset_interactive_game_process_override();
    std::filesystem::remove_all(temp_dir, ec);
}

} // namespace

int main() {
    try {
        test_gpu_driver_module_detection();
        test_diagnosis_classification();
        test_dialog_suppression_and_report_writing();
        std::cout << "native_crash_diagnostics: all checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
