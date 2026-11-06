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
    int32_t clip_cpu;      /* 1: run the vision encoder (CLIP) on the CPU */
    int32_t clip_threads;  /* CLIP threads; 0 = default, -1 = every online core */
    int32_t n_ubatch;      /* 0 keeps llama_context_default_params().n_ubatch */
    int32_t resident_output; /* 1: keep CPU output head resident in anonymous host memory */
    int32_t mincore_measure; /* 1: measure mapping residency with mincore / mach vm */
} qwen_probe_options;

typedef struct qwen_probe_metrics {
    uint64_t avail_mem_before_load_bytes;
    uint64_t lowest_avail_mem_bytes;
    uint64_t phys_footprint_after_model_load_bytes;
    uint64_t peak_phys_footprint_bytes;
    int64_t decode_major_faults;
    double decode_major_faults_per_token;
    uint64_t metal_recommended_max_working_set_bytes;
    uint64_t metal_current_allocated_after_load_bytes;
    uint64_t metal_current_allocated_after_image_chunk_bytes;
    uint64_t metal_current_allocated_after_decode_bytes;
    double model_load_seconds;
    double image_encode_seconds;
    double image_encode_warm_seconds;
    double time_to_first_token_seconds;
    double total_inference_seconds;
    double generated_tokens_per_second;
    int32_t generated_tokens;
} qwen_probe_metrics;

typedef void (*qwen_probe_log_fn)(const char * message);

void qwen_probe_set_logger(qwen_probe_log_fn logger);
uint64_t qwen_probe_available_memory(void);
uint64_t qwen_probe_phys_footprint(void);
int64_t qwen_probe_major_faults(void);
void qwen_probe_reset_live_memory_stats(void);
uint64_t qwen_probe_live_peak_footprint(void);
uint64_t qwen_probe_live_lowest_available_memory(void);
uint64_t qwen_probe_live_after_load_footprint(void);
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

int32_t qwen_probe_run_with_adapter(
    const char * variant_name,
    const qwen_probe_options * options,
    const char * model_path,
    const char * mmproj_path,
    const char * adapter_path,
    float adapter_scale,
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
