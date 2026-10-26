#include "core/telemetry.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("telemetry : ne lève pas, pic nul si NVML est absent", "[telemetry]") {
    banc::Telemetry telemetry;
    CHECK_NOTHROW(telemetry.device_name());
    CHECK_NOTHROW(telemetry.error());

    if (telemetry.available()) {
        const auto sample = telemetry.sample();
        REQUIRE(sample.has_value());
        CHECK(sample->vram_total > 0);
        banc::VramPeakWatcher watcher(telemetry);
        CHECK_NOTHROW(watcher.stop());
    } else {
        CHECK_FALSE(telemetry.sample().has_value());
        CHECK_FALSE(telemetry.error().empty());
        banc::VramPeakWatcher watcher(telemetry);
        CHECK(watcher.stop() == 0);
    }
}
