#include "core/telemetry.hpp"

#include <atomic>
#include <mutex>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#if defined(__APPLE__)
#include "core/telemetry_metal.hpp"
#endif
#endif

namespace waffle {

namespace {

// Subset of the NVML C API (nvml.h), declared here so the build does not depend on the CUDA Toolkit.
using nvmlReturn_t = int;
using nvmlDevice_t = struct nvmlDevice_st*;
struct nvmlUtilization_t {
    unsigned int gpu, memory;
};
struct nvmlMemory_t {
    unsigned long long total, free, used;
};
constexpr nvmlReturn_t NVML_SUCCESS = 0;
constexpr int NVML_TEMPERATURE_GPU = 0;
constexpr int NVML_CLOCK_SM = 1;

struct NvmlApi {
    nvmlReturn_t (*init)() = nullptr;
    nvmlReturn_t (*shutdown)() = nullptr;
    const char* (*error_string)(nvmlReturn_t) = nullptr;
    nvmlReturn_t (*get_count)(unsigned int*) = nullptr;
    nvmlReturn_t (*get_handle)(unsigned int, nvmlDevice_t*) = nullptr;
    nvmlReturn_t (*get_name)(nvmlDevice_t, char*, unsigned int) = nullptr;
    nvmlReturn_t (*get_utilization)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
    nvmlReturn_t (*get_memory)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
    nvmlReturn_t (*get_temperature)(nvmlDevice_t, int, unsigned int*) = nullptr;
    nvmlReturn_t (*get_power)(nvmlDevice_t, unsigned int*) = nullptr;
    nvmlReturn_t (*get_clock)(nvmlDevice_t, int, unsigned int*) = nullptr;
};

} // namespace

struct Telemetry::Impl {
    void* lib = nullptr;
    NvmlApi api;
    nvmlDevice_t device = nullptr;
    bool ok = false;
    bool apple_metal = false;
    unsigned count = 0;
    unsigned index = 0;
    std::string name;
    std::string error;
    std::mutex mutex;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    void* symbol(const char* sym) {
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), sym));
#else
        return dlsym(lib, sym);
#endif
    }

    template <class Fn> bool bind(Fn& fn, const char* sym) {
        fn = reinterpret_cast<Fn>(symbol(sym));
        return fn != nullptr;
    }

    void open() {
#if defined(__APPLE__)
        MetalProbe probe;
        if (!metal_gpu_probe(probe)) {
            error = probe.error.empty() ? "No Metal GPU found" : probe.error;
            return;
        }
        name = probe.name;
        apple_metal = true;
        ok = true;
        count = 1;
        index = 0;
        return;
#elif defined(_WIN32)
        lib = LoadLibraryA("nvml.dll");
        if (!lib) lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
#else
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_NOW);
        if (!lib) lib = dlopen("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", RTLD_NOW);
        if (!lib) lib = dlopen("/usr/lib64/libnvidia-ml.so.1", RTLD_NOW);
        if (!lib) lib = dlopen("/usr/lib/wsl/lib/libnvidia-ml.so.1", RTLD_NOW);
#endif
        if (!lib) {
            error = "NVML not found (NVIDIA driver missing?)";
            return;
        }
        const bool bound =
            bind(api.init, "nvmlInit_v2") && bind(api.shutdown, "nvmlShutdown") &&
            bind(api.error_string, "nvmlErrorString") && bind(api.get_handle, "nvmlDeviceGetHandleByIndex_v2") &&
            bind(api.get_name, "nvmlDeviceGetName") && bind(api.get_utilization, "nvmlDeviceGetUtilizationRates") &&
            bind(api.get_memory, "nvmlDeviceGetMemoryInfo") && bind(api.get_temperature, "nvmlDeviceGetTemperature") &&
            bind(api.get_power, "nvmlDeviceGetPowerUsage") && bind(api.get_clock, "nvmlDeviceGetClockInfo");
        if (!bound) {
            error = "Incompatible NVML version (missing function)";
            return;
        }
        if (nvmlReturn_t rc = api.init(); rc != NVML_SUCCESS) {
            error = std::string("nvmlInit failed: ") + api.error_string(rc);
            return;
        }
        unsigned int reported = 0;
        if (bind(api.get_count, "nvmlDeviceGetCount_v2") || bind(api.get_count, "nvmlDeviceGetCount")) {
            if (api.get_count(&reported) == NVML_SUCCESS) count = reported;
        }
        if (count == 0) count = 1;
        if (nvmlReturn_t rc = api.get_handle(0, &device); rc != NVML_SUCCESS) {
            error = std::string("No NVIDIA GPU: ") + api.error_string(rc);
            api.shutdown();
            count = 0;
            return;
        }
        char buf[96] = {};
        if (api.get_name(device, buf, sizeof buf) == NVML_SUCCESS) name = buf;
        index = 0;
        ok = true;
    }

    void close() {
        if (ok && api.shutdown) api.shutdown();
        ok = false;
        apple_metal = false;
#ifdef _WIN32
        if (lib) FreeLibrary(static_cast<HMODULE>(lib));
#else
        if (lib) dlclose(lib);
#endif
        lib = nullptr;
    }
};

Telemetry::Telemetry() : impl_(std::make_unique<Impl>()) {
    impl_->open();
}
Telemetry::~Telemetry() {
    impl_->close();
}

bool Telemetry::available() const {
    return impl_->ok;
}
std::string Telemetry::device_name() const {
    return impl_->name;
}
std::string Telemetry::error() const {
    return impl_->error;
}
unsigned Telemetry::device_count() const {
    return impl_->count;
}
unsigned Telemetry::device_index() const {
    return impl_->index;
}
bool Telemetry::reports_process_memory() const {
    return impl_->apple_metal;
}
bool Telemetry::select_device(unsigned index) {
    if (!impl_->ok) return false;
    if (impl_->apple_metal) return index == 0;
    if (index >= impl_->count) return false;
    std::lock_guard lock(impl_->mutex);
    nvmlDevice_t device = nullptr;
    if (impl_->api.get_handle(index, &device) != NVML_SUCCESS) return false;
    impl_->device = device;
    impl_->index = index;
    char buf[96] = {};
    if (impl_->api.get_name(device, buf, sizeof buf) == NVML_SUCCESS) impl_->name = buf;
    return true;
}

std::optional<GpuSample> Telemetry::sample() {
    if (!impl_->ok) return std::nullopt;
#if defined(__APPLE__)
    if (impl_->apple_metal) {
        std::lock_guard lock(impl_->mutex);
        MetalProbe probe;
        if (!metal_gpu_probe(probe)) return std::nullopt;
        GpuSample s;
        s.t_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - impl_->t0).count();
        s.vram_used = probe.used;
        s.vram_total = probe.total;
        return s;
    }
#endif
    std::lock_guard lock(impl_->mutex);
    auto& api = impl_->api;
    GpuSample s;
    s.t_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - impl_->t0).count();
    nvmlUtilization_t util{};
    if (api.get_utilization(impl_->device, &util) == NVML_SUCCESS) {
        s.gpu_util_pct = util.gpu;
        s.mem_util_pct = util.memory;
    }
    nvmlMemory_t mem{};
    if (api.get_memory(impl_->device, &mem) == NVML_SUCCESS) {
        s.vram_used = mem.used;
        s.vram_total = mem.total;
    }
    unsigned int value = 0;
    if (api.get_temperature(impl_->device, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS) s.temperature_c = value;
    if (api.get_power(impl_->device, &value) == NVML_SUCCESS) s.power_w = value / 1000.0;
    if (api.get_clock(impl_->device, NVML_CLOCK_SM, &value) == NVML_SUCCESS) s.sm_clock_mhz = value;
    return s;
}

// ---------------------------------------------------------------------------

struct VramPeakWatcher::Impl {
    std::atomic<uint64_t> peak{0};
    std::atomic<uint64_t> baseline{0};
    std::atomic<bool> have_baseline{false};
    std::jthread thread;
    std::atomic<bool> stopped{false};
};

VramPeakWatcher::VramPeakWatcher(Telemetry& telemetry, std::chrono::milliseconds interval,
                                 std::optional<uint64_t> baseline)
    : impl_(std::make_unique<Impl>()) {
    if (baseline) {
        impl_->baseline.store(*baseline);
        impl_->have_baseline.store(true);
    }
    if (!telemetry.available()) return;
    impl_->thread = std::jthread([this, &telemetry, interval](const std::stop_token& st) {
        while (!st.stop_requested()) {
            if (auto s = telemetry.sample()) {
                if (!impl_->have_baseline.load()) {
                    impl_->baseline.store(s->vram_used);
                    impl_->have_baseline.store(true);
                }
                uint64_t prev = impl_->peak.load();
                while (s->vram_used > prev && !impl_->peak.compare_exchange_weak(prev, s->vram_used)) {
                }
            }
            std::this_thread::sleep_for(interval);
        }
    });
}

VramPeakWatcher::~VramPeakWatcher() {
    stop();
}

uint64_t VramPeakWatcher::stop() {
    bool expected = false;
    if (impl_->stopped.compare_exchange_strong(expected, true)) {
        if (impl_->thread.joinable()) {
            impl_->thread.request_stop();
            impl_->thread.join();
        }
    }
    if (!impl_->have_baseline.load()) return 0;
    const uint64_t peak = impl_->peak.load();
    const uint64_t baseline = impl_->baseline.load();
    return peak > baseline ? peak - baseline : 0;
}

} // namespace waffle
