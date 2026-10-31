#ifndef QWEN_PROBE_ENGINE_H
#define QWEN_PROBE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen_probe_options {
    int32_t disable_flash_attn;
    int32_t disable_mmap;
    int32_t disable_op_offload;
    int32_t cpu_output_override;
} qwen_probe_options;

typedef struct qwen_probe_metrics {
    uint64_t avail_mem_before_load_bytes;
    uint64_t phys_footprint_after_model_load_bytes;
    uint64_t peak_phys_footprint_bytes;
    double model_load_seconds;
    double image_encode_seconds;
    double image_encode_warm_seconds;
    double time_to_first_token_seconds;
    double generated_tokens_per_second;
    int32_t generated_tokens;
} qwen_probe_metrics;

uint64_t qwen_probe_available_memory(void);
uint64_t qwen_probe_phys_footprint(void);
uint64_t qwen_probe_live_peak_footprint(void);
void qwen_probe_record_live_footprint(void);
void qwen_probe_device_machine(char * buf, size_t buf_len);

int32_t qwen_probe_run(
    const char * variant_name,
    const qwen_probe_options * options,
    const char * model_path,
    const char * mmproj_path,
    const uint8_t * rgb_data,
    uint32_t image_width,
    uint32_t image_height,
    const char * prompt,
    int32_t max_new_tokens,
    qwen_probe_metrics * out_metrics,
    char * out_text,
    size_t out_text_cap,
    char * out_error,
    size_t out_error_cap
);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_PROBE_ENGINE_H */
