// GPU telemetry. Windows and Linux load NVML dynamically (no CUDA Toolkit dependency).
// macOS reads the Metal device instead: NVML is not shipped there.
#pragma once

#include "core/types.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <string>

namespace waffle {

class Telemetry {
public:
    Telemetry(); // never throws; available() reports whether NVML responds
    ~Telemetry();
    Telemetry(const Telemetry&) = delete;
    Telemetry& operator=(const Telemetry&) = delete;

    bool available() const;
    std::string device_name() const;   // "" if unavailable
    std::string error() const;         // reason it is unavailable
    std::optional<GpuSample> sample(); // thread-safe; nullopt if unavailable

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Samples used VRAM at a regular interval on a thread and keeps the maximum.
class VramPeakWatcher {
public:
    explicit VramPeakWatcher(Telemetry& telemetry, std::chrono::milliseconds interval = std::chrono::milliseconds(50));
    ~VramPeakWatcher(); // stops the thread
    uint64_t stop();    // stops and returns the peak in bytes (0 if NVML is unavailable)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace waffle
