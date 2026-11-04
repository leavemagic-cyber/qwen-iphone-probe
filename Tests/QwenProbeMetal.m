#import <Metal/Metal.h>

#include <stdint.h>

uint64_t qwen_probe_metal_recommended_max_working_set_bytes(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return 0;
    }
    return (uint64_t)device.recommendedMaxWorkingSetSize;
}

uint64_t qwen_probe_metal_current_allocated_bytes(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return 0;
    }
    return (uint64_t)device.currentAllocatedSize;
}
