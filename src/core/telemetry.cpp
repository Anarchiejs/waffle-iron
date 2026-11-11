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
#ifdef _WIN32
        lib = LoadLibraryA("nvml.dll");
        if (!lib) lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
#else
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
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
        if (nvmlReturn_t rc = api.get_handle(0, &device); rc != NVML_SUCCESS) {
            error = std::string("No NVIDIA GPU: ") + api.error_string(rc);
            api.shutdown();
            return;
        }
        char buf[96] = {};
        if (api.get_name(device, buf, sizeof buf) == NVML_SUCCESS) name = buf;
        ok = true;
    }

    void close() {
        if (ok) api.shutdown();
        ok = false;
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

std::optional<GpuSample> Telemetry::sample() {
    if (!impl_->ok) return std::nullopt;
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
    std::jthread thread;
    bool stopped = false;
};

VramPeakWatcher::VramPeakWatcher(Telemetry& telemetry, std::chrono::milliseconds interval)
    : impl_(std::make_unique<Impl>()) {
    if (!telemetry.available()) return;
    impl_->thread = std::jthread([this, &telemetry, interval](std::stop_token st) {
        while (!st.stop_requested()) {
            if (auto s = telemetry.sample()) {
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
    if (!impl_->stopped) {
        impl_->stopped = true;
        if (impl_->thread.joinable()) {
            impl_->thread.request_stop();
            impl_->thread.join();
        }
    }
    return impl_->peak.load();
}

} // namespace waffle
