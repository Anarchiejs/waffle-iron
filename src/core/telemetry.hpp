// Télémétrie GPU via NVML, chargée dynamiquement (nvml.dll) : aucune dépendance de build à CUDA.
#pragma once

#include "core/types.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <string>

namespace banc {

class Telemetry {
public:
    Telemetry(); // ne lève jamais ; available() dit si NVML répond
    ~Telemetry();
    Telemetry(const Telemetry&) = delete;
    Telemetry& operator=(const Telemetry&) = delete;

    bool available() const;
    std::string device_name() const;   // "" si indisponible
    std::string error() const;         // raison de l'indisponibilité
    std::optional<GpuSample> sample(); // thread-safe ; nullopt si indisponible

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Relève la VRAM utilisée à intervalle régulier dans un thread et garde le maximum.
class VramPeakWatcher {
public:
    explicit VramPeakWatcher(Telemetry& telemetry, std::chrono::milliseconds interval = std::chrono::milliseconds(50));
    ~VramPeakWatcher(); // arrête le thread
    uint64_t stop();    // arrête et rend le pic en octets (0 si NVML indisponible)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace banc
