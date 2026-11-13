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
    std::string device_name() const; // "" if unavailable
    std::string error() const;       // reason it is unavailable
    unsigned device_count() const;   // 0 when unavailable; 1 on Metal
    unsigned device_index() const;
    // Metal reports this process allocation, not device-wide VRAM.
    bool reports_process_memory() const;
    // Switches the NVML device. Index 0 is the only Metal device. False when out of range.
    bool select_device(unsigned index);
    std::optional<GpuSample> sample(); // thread-safe; nullopt if unavailable

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Samples used memory on a thread. stop() returns the highest reading minus the baseline
// (the optional baseline, otherwise the first sample). That is the increase over the watched
// interval, not the absolute NVML level. 0 when telemetry has no sample.
class VramPeakWatcher {
public:
    explicit VramPeakWatcher(Telemetry& telemetry, std::chrono::milliseconds interval = std::chrono::milliseconds(50),
                             std::optional<uint64_t> baseline = std::nullopt);
    ~VramPeakWatcher(); // stops the thread
    uint64_t stop();    // stops and returns the increase in bytes

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace waffle
