#include "core/telemetry_metal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace waffle {

bool metal_gpu_probe(MetalProbe& out) {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        out.error = "No Metal GPU found";
        return false;
    }
    const char* name = device.name.UTF8String;
    out.name = name ? name : "Metal GPU";
    out.total = device.recommendedMaxWorkingSetSize;
    out.used = device.currentAllocatedSize;
    if (out.total == 0) {
        out.error = "Metal device did not report a working set";
        return false;
    }
    return true;
}

} // namespace waffle
