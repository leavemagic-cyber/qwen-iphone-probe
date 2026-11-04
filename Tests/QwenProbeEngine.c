#include "QwenProbeEngine.h"

#include <TargetConditionals.h>
#include <llama/llama.h>
#include <llama/mtmd.h>
#include <llama/mtmd-helper.h>

#include <mach/mach.h>
#include <os/proc.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <unistd.h>

static pthread_once_t g_backend_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_peak_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_live_peak_footprint = 0;
static uint64_t g_live_lowest_avail = UINT64_MAX;
static uint64_t g_live_after_load_footprint = 0;
static qwen_probe_log_fn g_logger = NULL;

static void init_llama_backend_once(void) {
    llama_backend_init();
}

void qwen_probe_set_logger(qwen_probe_log_fn logger) {
    pthread_mutex_lock(&g_log_mutex);
    g_logger = logger;
    pthread_mutex_unlock(&g_log_mutex);
}

static void probe_logf(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
static void probe_logf(const char * fmt, ...) {
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    pthread_mutex_lock(&g_log_mutex);
    qwen_probe_log_fn logger = g_logger;
    pthread_mutex_unlock(&g_log_mutex);

    if (logger) {
        logger(buf);
    } else {
        fprintf(stderr, "%s\n", buf);
        fflush(stderr);
    }
}

uint64_t qwen_probe_available_memory(void) {
    return (uint64_t)os_proc_available_memory();
}

uint64_t qwen_probe_phys_footprint(void) {
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count);
    if (kr != KERN_SUCCESS) {
        return 0;
    }
    return (uint64_t)vm_info.phys_footprint;
}

int64_t qwen_probe_major_faults(void) {
    struct rusage ru;
    memset(&ru, 0, sizeof(ru));
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return 0;
    }
    return (int64_t)ru.ru_majflt;
}

static void record_after_load_footprint(uint64_t fp) {
    pthread_mutex_lock(&g_peak_mutex);
    g_live_after_load_footprint = fp;
    pthread_mutex_unlock(&g_peak_mutex);
}

void qwen_probe_record_live_footprint(void) {
    uint64_t cur_fp = qwen_probe_phys_footprint();
    uint64_t cur_avail = qwen_probe_available_memory();
    pthread_mutex_lock(&g_peak_mutex);
    if (cur_fp > g_live_peak_footprint) {
        g_live_peak_footprint = cur_fp;
    }
    if (cur_avail > 0 && cur_avail < g_live_lowest_avail) {
        g_live_lowest_avail = cur_avail;
    }
    pthread_mutex_unlock(&g_peak_mutex);
}

void qwen_probe_reset_live_memory_stats(void) {
    pthread_mutex_lock(&g_peak_mutex);
    g_live_peak_footprint = 0;
    g_live_lowest_avail = UINT64_MAX;
    g_live_after_load_footprint = 0;
    pthread_mutex_unlock(&g_peak_mutex);
    qwen_probe_record_live_footprint();
}

uint64_t qwen_probe_live_peak_footprint(void) {
    qwen_probe_record_live_footprint();
    pthread_mutex_lock(&g_peak_mutex);
    uint64_t val = g_live_peak_footprint;
    pthread_mutex_unlock(&g_peak_mutex);
    return val;
}

uint64_t qwen_probe_live_lowest_available_memory(void) {
    qwen_probe_record_live_footprint();
    pthread_mutex_lock(&g_peak_mutex);
    uint64_t val = (g_live_lowest_avail == UINT64_MAX) ? 0 : g_live_lowest_avail;
    pthread_mutex_unlock(&g_peak_mutex);
    return val;
}

uint64_t qwen_probe_live_after_load_footprint(void) {
    pthread_mutex_lock(&g_peak_mutex);
    uint64_t val = g_live_after_load_footprint;
    pthread_mutex_unlock(&g_peak_mutex);
    return val;
}

void qwen_probe_device_machine(char * buf, size_t buf_len) {
    if (!buf || buf_len == 0) {
        return;
    }
    buf[0] = '\0';
    size_t size = buf_len;
    if (sysctlbyname("hw.machine", buf, &size, NULL, 0) == 0 && buf[0] != '\0') {
        buf[buf_len - 1] = '\0';
        return;
    }
    struct utsname uts;
    if (uname(&uts) == 0) {
        snprintf(buf, buf_len, "%s", uts.machine);
    } else {
        snprintf(buf, buf_len, "unknown");
    }
}

typedef struct peak_sampler_state {
    pthread_mutex_t mutex;
    bool running;
    uint64_t peak_bytes;
    uint64_t min_avail_bytes;
} peak_sampler_state;

static uint64_t sampler_sample_now(peak_sampler_state * st) {
    uint64_t cur_fp = qwen_probe_phys_footprint();
    uint64_t cur_avail = qwen_probe_available_memory();

    pthread_mutex_lock(&st->mutex);
    if (cur_fp > st->peak_bytes) {
        st->peak_bytes = cur_fp;
    }
    if (cur_avail > 0 && cur_avail < st->min_avail_bytes) {
        st->min_avail_bytes = cur_avail;
    }
    pthread_mutex_unlock(&st->mutex);

    pthread_mutex_lock(&g_peak_mutex);
    if (cur_fp > g_live_peak_footprint) {
        g_live_peak_footprint = cur_fp;
    }
    if (cur_avail > 0 && cur_avail < g_live_lowest_avail) {
        g_live_lowest_avail = cur_avail;
    }
    pthread_mutex_unlock(&g_peak_mutex);
    return cur_fp;
}

static void * sampler_thread_main(void * arg) {
    peak_sampler_state * st = (peak_sampler_state *)arg;
    while (true) {
        pthread_mutex_lock(&st->mutex);
        bool keep_going = st->running;
        pthread_mutex_unlock(&st->mutex);
        if (!keep_going) {
            break;
        }
        sampler_sample_now(st);
        usleep(20000); /* 20 ms */
    }
    return NULL;
}

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
) {
    return qwen_probe_run_with_adapter(
        variant_name,
        options,
        model_path,
        mmproj_path,
        NULL,
        0.0f,
        rgb_data,
        image_width,
        image_height,
        prompt,
        max_new_tokens,
        out_metrics,
        out_text,
        out_text_cap,
        out_error,
        out_error_cap
    );
}

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
) {
    if (!out_metrics || !out_text || out_text_cap == 0 || !out_error || out_error_cap == 0) {
        return -1;
    }
    memset(out_metrics, 0, sizeof(*out_metrics));
    out_text[0] = '\0';
    out_error[0] = '\0';

    if (!model_path || !mmproj_path || !rgb_data || !prompt || image_width == 0 || image_height == 0) {
        snprintf(out_error, out_error_cap, "Invalid arguments to qwen_probe_run_with_adapter");
        return -1;
    }

    const char * vname = (variant_name && variant_name[0] != '\0') ? variant_name : "baseline";
    const bool opt_no_fa         = (options && options->disable_flash_attn != 0);
    const bool opt_no_mmap       = (options && options->disable_mmap != 0);
    const bool opt_no_op_offload = (options && options->disable_op_offload != 0);
    const bool opt_cpu_output    = (options && options->cpu_output_override != 0);
    const bool has_adapter       = (adapter_path != NULL && adapter_path[0] != '\0');

    uint64_t avail_before = qwen_probe_available_memory();
    out_metrics->avail_mem_before_load_bytes = avail_before;

    pthread_mutex_lock(&g_peak_mutex);
    g_live_peak_footprint = 0;
    g_live_lowest_avail = (avail_before > 0) ? avail_before : UINT64_MAX;
    g_live_after_load_footprint = 0;
    pthread_mutex_unlock(&g_peak_mutex);

    peak_sampler_state sampler;
    pthread_mutex_init(&sampler.mutex, NULL);
    sampler.running = true;
    sampler.peak_bytes = 0;
    sampler.min_avail_bytes = (avail_before > 0) ? avail_before : UINT64_MAX;
    uint64_t initial_fp = sampler_sample_now(&sampler);
    int64_t majflt_before_load = qwen_probe_major_faults();

    probe_logf(
        "[QWENPROBE] variant=%s step=before_load disable_fa=%d disable_mmap=%d "
        "disable_op_offload=%d cpu_output_override=%d has_adapter=%d "
        "avail_bytes=%llu phys_footprint_bytes=%llu ru_majflt=%lld",
        vname,
        opt_no_fa ? 1 : 0,
        opt_no_mmap ? 1 : 0,
        opt_no_op_offload ? 1 : 0,
        opt_cpu_output ? 1 : 0,
        has_adapter ? 1 : 0,
        (unsigned long long)out_metrics->avail_mem_before_load_bytes,
        (unsigned long long)initial_fp,
        (long long)majflt_before_load
    );

    pthread_t sampler_tid;
    bool sampler_started = (pthread_create(&sampler_tid, NULL, sampler_thread_main, &sampler) == 0);

    int32_t rc_final = 0;
    struct llama_model * model = NULL;
    struct llama_adapter_lora * lora_adapter = NULL;
    struct llama_context * lctx = NULL;
    mtmd_context * mctx = NULL;
    mtmd_bitmap * bmp = NULL;
    mtmd_input_chunks * chunks = NULL;
    struct llama_sampler * smpl = NULL;
    struct llama_batch_ext * gen_batch = NULL;
    struct llama_model_tensor_buft_override cpu_output_overrides[2];
    memset(cpu_output_overrides, 0, sizeof(cpu_output_overrides));

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int32_t n_threads = (ncpu > 2) ? (int32_t)(ncpu - 2) : 1;
    if (n_threads > 6) {
        n_threads = 6;
    }

    int64_t t_load_start = llama_time_us();
    pthread_once(&g_backend_once, init_llama_backend_once);

    struct llama_model_params mparams = llama_model_default_params();
#if TARGET_OS_SIMULATOR
    mparams.n_gpu_layers = 0;
#else
    mparams.n_gpu_layers = -1; /* offload all layers to Metal */
#endif
    if (opt_no_mmap) {
        mparams.load_mode = LLAMA_LOAD_MODE_NONE; /* use_mmap = false */
    }
    if (opt_cpu_output) {
        cpu_output_overrides[0].pattern = "^(output|token_embd)\\.weight$";
        cpu_output_overrides[0].buft    = ggml_backend_cpu_buffer_type();
        cpu_output_overrides[1].pattern = NULL;
        cpu_output_overrides[1].buft    = NULL;
        mparams.tensor_buft_overrides   = cpu_output_overrides;
    } else {
        mparams.tensor_buft_overrides   = NULL;
    }

    model = llama_model_load_from_file(model_path, mparams);
    uint64_t fp_after_model = sampler_sample_now(&sampler);
    uint64_t avail_after_model = qwen_probe_available_memory();
    out_metrics->phys_footprint_after_model_load_bytes = fp_after_model;
    record_after_load_footprint(fp_after_model);
    probe_logf(
        "[QWENPROBE] variant=%s step=after_text_model_load ok=%d phys_footprint_bytes=%llu avail_bytes=%llu",
        vname,
        model != NULL ? 1 : 0,
        (unsigned long long)fp_after_model,
        (unsigned long long)avail_after_model
    );
    if (opt_no_mmap) {
        usleep(20000);
    }
    if (!model) {
        snprintf(out_error, out_error_cap, "llama_model_load_from_file failed");
        rc_final = 1;
        goto cleanup;
    }

    if (has_adapter) {
        lora_adapter = llama_adapter_lora_init(model, adapter_path);
        uint64_t fp_after_lora = sampler_sample_now(&sampler);
        out_metrics->phys_footprint_after_model_load_bytes = fp_after_lora;
        record_after_load_footprint(fp_after_lora);
        probe_logf(
            "[QWENPROBE] variant=%s step=after_lora_init ok=%d scale=%.3f phys_footprint_bytes=%llu",
            vname,
            lora_adapter != NULL ? 1 : 0,
            (double)adapter_scale,
            (unsigned long long)fp_after_lora
        );
        if (!lora_adapter) {
            snprintf(out_error, out_error_cap, "llama_adapter_lora_init failed");
            rc_final = 18;
            goto cleanup;
        }
    }

    {
        struct llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 1024;
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
        cparams.n_threads = n_threads;
        cparams.n_threads_batch = n_threads;
        if (opt_no_fa) {
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        cparams.op_offload = !opt_no_op_offload;

        lctx = llama_init_from_model(model, cparams);
        uint64_t fp_after_ctx = sampler_sample_now(&sampler);
        out_metrics->phys_footprint_after_model_load_bytes = fp_after_ctx;
        record_after_load_footprint(fp_after_ctx);
        probe_logf(
            "[QWENPROBE] variant=%s step=after_ctx_init ok=%d op_offload=%d phys_footprint_bytes=%llu",
            vname,
            lctx != NULL ? 1 : 0,
            cparams.op_offload ? 1 : 0,
            (unsigned long long)fp_after_ctx
        );
        if (!lctx) {
            snprintf(out_error, out_error_cap, "llama_init_from_model failed");
            rc_final = 2;
            goto cleanup;
        }
    }

    if (has_adapter) {
        struct llama_adapter_lora * adapters_arr[1] = { lora_adapter };
        float scales_arr[1] = { adapter_scale };
        int32_t lora_rc = llama_set_adapters_lora(lctx, adapters_arr, 1, scales_arr);
        uint64_t fp_after_lora_set = sampler_sample_now(&sampler);
        out_metrics->phys_footprint_after_model_load_bytes = fp_after_lora_set;
        record_after_load_footprint(fp_after_lora_set);
        probe_logf(
            "[QWENPROBE] variant=%s step=after_lora_set ok=%d rc=%d scale=%.3f phys_footprint_bytes=%llu",
            vname,
            lora_rc == 0 ? 1 : 0,
            lora_rc,
            (double)adapter_scale,
            (unsigned long long)fp_after_lora_set
        );
        if (lora_rc != 0) {
            snprintf(out_error, out_error_cap, "llama_set_adapters_lora failed (rc=%d)", lora_rc);
            rc_final = 19;
            goto cleanup;
        }
    }

    {
        struct mtmd_context_params vparams = mtmd_context_params_default();
#if TARGET_OS_SIMULATOR
        vparams.use_gpu = false;
#else
        vparams.use_gpu = !(options && options->clip_cpu != 0);
#endif
        int32_t clip_threads = (options && options->clip_threads != 0) ? options->clip_threads : n_threads;
        if (clip_threads < 0) {
            clip_threads = (ncpu > 0) ? (int32_t)ncpu : 1;
        }
        vparams.n_threads = clip_threads;
        probe_logf(
            "[QWENPROBE] variant=%s step=clip_config use_gpu=%d clip_threads=%d ncpu=%ld",
            vname,
            vparams.use_gpu ? 1 : 0,
            clip_threads,
            ncpu
        );
        vparams.warmup = false;
        vparams.image_max_tokens = 512;
        if (opt_no_fa) {
            vparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }

        mctx = mtmd_init_from_file(mmproj_path, model, vparams);
        int64_t t_load_end = llama_time_us();
        out_metrics->model_load_seconds = (double)(t_load_end - t_load_start) / 1e6;

        uint64_t fp_after_mmproj = sampler_sample_now(&sampler);
        uint64_t avail_after_load = qwen_probe_available_memory();
        int64_t majflt_after_load = qwen_probe_major_faults();
        out_metrics->phys_footprint_after_model_load_bytes = fp_after_mmproj;
        record_after_load_footprint(fp_after_mmproj);
        probe_logf(
            "[QWENPROBE] variant=%s step=after_mmproj_load ok=%d load_s=%.3f "
            "phys_footprint_bytes=%llu text_model_phys_footprint_bytes=%llu "
            "avail_bytes=%llu ru_majflt=%lld",
            vname,
            mctx != NULL ? 1 : 0,
            out_metrics->model_load_seconds,
            (unsigned long long)fp_after_mmproj,
            (unsigned long long)fp_after_model,
            (unsigned long long)avail_after_load,
            (long long)majflt_after_load
        );
        probe_logf(
            "QWENPROBE_LOAD variant=%s ok=%d load_s=%.3f "
            "phys_footprint_after_load_bytes=%llu text_model_phys_footprint_bytes=%llu "
            "avail_after_load_bytes=%llu",
            vname,
            mctx != NULL ? 1 : 0,
            out_metrics->model_load_seconds,
            (unsigned long long)fp_after_mmproj,
            (unsigned long long)fp_after_model,
            (unsigned long long)avail_after_load
        );
        usleep(50000); /* Allow log buffers to flush before look allocations */
        if (!mctx) {
            snprintf(out_error, out_error_cap, "mtmd_init_from_file failed");
            rc_final = 3;
            goto cleanup;
        }
    }

    if (!mtmd_support_vision(mctx)) {
        snprintf(out_error, out_error_cap, "Loaded mmproj does not report vision support");
        rc_final = 4;
        goto cleanup;
    }

    {
        int64_t t_infer_start = llama_time_us();

        bmp = mtmd_bitmap_init(image_width, image_height, rgb_data);
        if (!bmp) {
            snprintf(out_error, out_error_cap, "mtmd_bitmap_init failed");
            rc_final = 5;
            goto cleanup;
        }

        const char * marker = mtmd_default_marker();
        char user_msg[2048];
        int user_len = snprintf(user_msg, sizeof(user_msg), "%s%s", marker, prompt);
        if (user_len <= 0 || (size_t)user_len >= sizeof(user_msg)) {
            snprintf(out_error, out_error_cap, "Prompt buffer overflow");
            rc_final = 6;
            goto cleanup;
        }

        char formatted_prompt[4096];
        const char * tmpl = llama_model_chat_template(model, NULL);
        struct llama_chat_message chat_msg;
        chat_msg.role = "user";
        chat_msg.content = user_msg;

        int32_t formatted_len = llama_chat_apply_template(
            tmpl ? tmpl : "chatml",
            &chat_msg,
            1,
            true,
            formatted_prompt,
            (int32_t)(sizeof(formatted_prompt) - 1)
        );
        if (formatted_len <= 0 || (size_t)formatted_len >= sizeof(formatted_prompt)) {
            formatted_len = snprintf(
                formatted_prompt,
                sizeof(formatted_prompt),
                "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                user_msg
            );
        } else {
            formatted_prompt[formatted_len] = '\0';
        }

        chunks = mtmd_input_chunks_init();
        if (!chunks) {
            snprintf(out_error, out_error_cap, "mtmd_input_chunks_init failed");
            rc_final = 7;
            goto cleanup;
        }

        struct mtmd_input_text inp_text;
        inp_text.text = formatted_prompt;
        inp_text.text_len = (size_t)formatted_len;
        inp_text.add_special = true;
        inp_text.parse_special = true;

        const mtmd_bitmap * bitmaps[1] = { bmp };
        int32_t tok_rc = mtmd_tokenize(mctx, chunks, &inp_text, bitmaps, 1);
        uint64_t fp_after_tok = sampler_sample_now(&sampler);
        size_t n_chunks = (tok_rc == 0) ? mtmd_input_chunks_size(chunks) : 0;
        probe_logf(
            "[QWENPROBE] variant=%s step=after_tokenize ok=%d chunks=%zu phys_footprint_bytes=%llu",
            vname,
            tok_rc == 0 ? 1 : 0,
            n_chunks,
            (unsigned long long)fp_after_tok
        );
        if (tok_rc != 0) {
            snprintf(out_error, out_error_cap, "mtmd_tokenize failed (rc=%d)", tok_rc);
            rc_final = 8;
            goto cleanup;
        }

        if (n_chunks == 0) {
            snprintf(out_error, out_error_cap, "mtmd_tokenize produced 0 chunks");
            rc_final = 9;
            goto cleanup;
        }

        llama_pos n_past = 0;
        const int32_t n_batch = 512;
        double total_img_encode_cold_s = 0.0;
        double total_img_encode_warm_s = 0.0;
        int64_t total_img_encode_warm_us = 0;

        for (size_t i = 0; i < n_chunks; i++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
            enum mtmd_input_chunk_type ctype = mtmd_input_chunk_get_type(chunk);
            bool is_last_chunk = (i + 1 == n_chunks);

            if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                llama_pos new_n_past = n_past;
                int32_t eval_rc = mtmd_helper_eval_chunk_single(
                    mctx, lctx, chunk, n_past, 0, n_batch, is_last_chunk, &new_n_past
                );
                uint64_t fp_after_txt = sampler_sample_now(&sampler);
                probe_logf(
                    "[QWENPROBE] variant=%s step=after_text_chunk_%zu ok=%d is_last=%d phys_footprint_bytes=%llu",
                    vname,
                    i,
                    eval_rc == 0 ? 1 : 0,
                    is_last_chunk ? 1 : 0,
                    (unsigned long long)fp_after_txt
                );
                if (eval_rc != 0) {
                    snprintf(out_error, out_error_cap, "mtmd_helper_eval_chunk_single failed at chunk %zu (rc=%d)", i, eval_rc);
                    rc_final = 10;
                    goto cleanup;
                }
                n_past = new_n_past;
            } else {
                int64_t t_enc_cold_0 = llama_time_us();
                int32_t enc_rc = mtmd_encode_chunk(mctx, chunk);
                int64_t t_enc_cold_1 = llama_time_us();
                total_img_encode_cold_s += (double)(t_enc_cold_1 - t_enc_cold_0) / 1e6;
                out_metrics->image_encode_seconds = total_img_encode_cold_s;
                uint64_t fp_after_enc_cold = sampler_sample_now(&sampler);
                probe_logf(
                    "[QWENPROBE] variant=%s step=after_image_encode_cold ok=%d encode_s=%.3f phys_footprint_bytes=%llu",
                    vname,
                    enc_rc == 0 ? 1 : 0,
                    total_img_encode_cold_s,
                    (unsigned long long)fp_after_enc_cold
                );
                if (enc_rc != 0) {
                    snprintf(out_error, out_error_cap, "mtmd_encode_chunk (cold) failed at chunk %zu (rc=%d)", i, enc_rc);
                    rc_final = 11;
                    goto cleanup;
                }

                int64_t t_enc_warm_0 = llama_time_us();
                int32_t enc_warm_rc = mtmd_encode_chunk(mctx, chunk);
                int64_t t_enc_warm_1 = llama_time_us();
                int64_t warm_us = t_enc_warm_1 - t_enc_warm_0;
                total_img_encode_warm_us += warm_us;
                total_img_encode_warm_s += (double)warm_us / 1e6;
                out_metrics->image_encode_warm_seconds = total_img_encode_warm_s;
                uint64_t fp_after_enc_warm = sampler_sample_now(&sampler);
                probe_logf(
                    "[QWENPROBE] variant=%s step=after_image_encode_warm ok=%d encode_warm_s=%.3f phys_footprint_bytes=%llu",
                    vname,
                    enc_warm_rc == 0 ? 1 : 0,
                    total_img_encode_warm_s,
                    (unsigned long long)fp_after_enc_warm
                );
                if (enc_warm_rc != 0) {
                    snprintf(out_error, out_error_cap, "mtmd_encode_chunk (warm) failed at chunk %zu (rc=%d)", i, enc_warm_rc);
                    rc_final = 11;
                    goto cleanup;
                }

                float * embd = mtmd_get_output_embd(mctx);
                if (!embd) {
                    snprintf(out_error, out_error_cap, "mtmd_get_output_embd returned NULL");
                    rc_final = 12;
                    goto cleanup;
                }

                llama_pos new_n_past = n_past;
                int32_t dec_img_rc = mtmd_helper_decode_image_chunk(
                    mctx, lctx, chunk, embd, n_past, 0, n_batch, &new_n_past, NULL, NULL
                );
                uint64_t fp_after_img_dec = sampler_sample_now(&sampler);
                probe_logf(
                    "[QWENPROBE] variant=%s step=after_image_decode ok=%d phys_footprint_bytes=%llu",
                    vname,
                    dec_img_rc == 0 ? 1 : 0,
                    (unsigned long long)fp_after_img_dec
                );
                if (dec_img_rc != 0) {
                    snprintf(out_error, out_error_cap, "mtmd_helper_decode_image_chunk failed (rc=%d)", dec_img_rc);
                    rc_final = 13;
                    goto cleanup;
                }
                n_past = new_n_past;
            }
        }

        out_metrics->image_encode_seconds = total_img_encode_cold_s;
        out_metrics->image_encode_warm_seconds = total_img_encode_warm_s;

        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (!smpl) {
            snprintf(out_error, out_error_cap, "llama_sampler_chain_init failed");
            rc_final = 14;
            goto cleanup;
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        const struct llama_vocab * vocab = llama_model_get_vocab(model);
        gen_batch = llama_batch_ext_init(lctx);
        if (!gen_batch) {
            snprintf(out_error, out_error_cap, "llama_batch_ext_init failed");
            rc_final = 15;
            goto cleanup;
        }

        llama_token token_id = llama_sampler_sample(smpl, lctx, -1);
        llama_synchronize(lctx);
        int64_t t_first_tok = llama_time_us();
        int64_t ttft_us = (t_first_tok - t_infer_start) - total_img_encode_warm_us;
        if (ttft_us < 0) {
            ttft_us = t_first_tok - t_infer_start;
        }
        out_metrics->time_to_first_token_seconds = (double)ttft_us / 1e6;
        uint64_t fp_after_ttft = sampler_sample_now(&sampler);
        int64_t majflt_at_first_tok = qwen_probe_major_faults();
        probe_logf(
            "[QWENPROBE] variant=%s step=first_token ttft_s=%.3f phys_footprint_bytes=%llu ru_majflt=%lld",
            vname,
            out_metrics->time_to_first_token_seconds,
            (unsigned long long)fp_after_ttft,
            (long long)majflt_at_first_tok
        );

        int32_t n_generated = 0;
        int32_t n_decode_steps = 0;
        size_t text_pos = 0;
        int64_t t_gen_start = llama_time_us();
        int64_t majflt_before_decode = qwen_probe_major_faults();

        int32_t limit = (max_new_tokens > 0) ? max_new_tokens : 64;
        for (int32_t step = 0; step < limit; step++) {
            if (llama_vocab_is_eog(vocab, token_id)) {
                break;
            }

            char piece[512];
            int32_t n_piece = llama_token_to_piece(vocab, token_id, piece, (int32_t)sizeof(piece), 0, false);
            if (n_piece > 0) {
                size_t avail = (out_text_cap - 1 > text_pos) ? (out_text_cap - 1 - text_pos) : 0;
                size_t to_copy = ((size_t)n_piece < avail) ? (size_t)n_piece : avail;
                if (to_copy > 0) {
                    memcpy(out_text + text_pos, piece, to_copy);
                    text_pos += to_copy;
                    out_text[text_pos] = '\0';
                }
            }
            n_generated++;

            if (step + 1 >= limit) {
                break;
            }

            llama_batch_ext_clear(gen_batch);
            int32_t idx = llama_batch_ext_add_token(gen_batch, 0, token_id);
            if (idx < 0) {
                snprintf(out_error, out_error_cap, "llama_batch_ext_add_token failed (idx=%d)", idx);
                rc_final = 16;
                break;
            }
            llama_pos pos = n_past++;
            llama_batch_ext_set_pos(gen_batch, idx, &pos);
            llama_batch_ext_set_output_logits(gen_batch, idx, true);

            int32_t dec_rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, gen_batch);
            sampler_sample_now(&sampler);
            if (dec_rc != 0) {
                snprintf(out_error, out_error_cap, "llama_process decode failed at step %d (rc=%d)", step, dec_rc);
                rc_final = 17;
                break;
            }
            n_decode_steps++;
            token_id = llama_sampler_sample(smpl, lctx, -1);
        }

        llama_synchronize(lctx);
        int64_t t_gen_end = llama_time_us();
        int64_t majflt_after_decode = qwen_probe_major_faults();
        uint64_t fp_after_gen = sampler_sample_now(&sampler);

        int64_t decode_majflt = (majflt_after_decode >= majflt_before_decode)
            ? (majflt_after_decode - majflt_before_decode)
            : 0;
        out_metrics->decode_major_faults = decode_majflt;
        if (n_decode_steps > 0) {
            out_metrics->decode_major_faults_per_token = (double)decode_majflt / (double)n_decode_steps;
        } else if (n_generated > 0) {
            out_metrics->decode_major_faults_per_token = (double)decode_majflt / (double)n_generated;
        } else {
            out_metrics->decode_major_faults_per_token = 0.0;
        }

        int64_t total_infer_us = (t_gen_end - t_infer_start) - total_img_encode_warm_us;
        if (total_infer_us < 0) {
            total_infer_us = t_gen_end - t_infer_start;
        }
        out_metrics->total_inference_seconds = (double)total_infer_us / 1e6;

        out_metrics->generated_tokens = n_generated;
        double decode_elapsed_s = (double)(t_gen_end - t_gen_start) / 1e6;
        if (n_decode_steps > 0 && decode_elapsed_s > 0.0) {
            out_metrics->generated_tokens_per_second = (double)n_decode_steps / decode_elapsed_s;
        } else if (n_generated > 0 && out_metrics->time_to_first_token_seconds > 0.0) {
            out_metrics->generated_tokens_per_second = (double)n_generated / out_metrics->time_to_first_token_seconds;
        } else {
            out_metrics->generated_tokens_per_second = 0.0;
        }

        probe_logf(
            "[QWENPROBE] variant=%s step=after_generation tokens=%d decode_steps=%d tok_per_s=%.3f "
            "decode_major_faults=%lld decode_major_faults_per_token=%.3f "
            "total_s=%.3f phys_footprint_bytes=%llu",
            vname,
            n_generated,
            n_decode_steps,
            out_metrics->generated_tokens_per_second,
            (long long)out_metrics->decode_major_faults,
            out_metrics->decode_major_faults_per_token,
            out_metrics->total_inference_seconds,
            (unsigned long long)fp_after_gen
        );
    }

cleanup:
    sampler_sample_now(&sampler);
    pthread_mutex_lock(&sampler.mutex);
    sampler.running = false;
    pthread_mutex_unlock(&sampler.mutex);
    if (sampler_started) {
        pthread_join(sampler_tid, NULL);
    }
    pthread_mutex_lock(&sampler.mutex);
    out_metrics->peak_phys_footprint_bytes = sampler.peak_bytes;
    out_metrics->lowest_avail_mem_bytes = (sampler.min_avail_bytes == UINT64_MAX)
        ? out_metrics->avail_mem_before_load_bytes
        : sampler.min_avail_bytes;
    pthread_mutex_unlock(&sampler.mutex);
    pthread_mutex_destroy(&sampler.mutex);

    if (gen_batch) {
        llama_batch_ext_free(gen_batch);
    }
    if (smpl) {
        llama_sampler_free(smpl);
    }
    if (chunks) {
        mtmd_input_chunks_free(chunks);
    }
    if (bmp) {
        mtmd_bitmap_free(bmp);
    }
    if (mctx) {
        mtmd_free(mctx);
    }
    if (lctx) {
        llama_free(lctx);
    }
    if (lora_adapter) {
        llama_adapter_lora_free(lora_adapter);
    }
    if (model) {
        llama_model_free(model);
    }

    uint64_t fp_after_cleanup = qwen_probe_phys_footprint();
    probe_logf(
        "[QWENPROBE] variant=%s step=after_cleanup phys_footprint_bytes=%llu "
        "peak_phys_footprint_bytes=%llu lowest_avail_bytes=%llu",
        vname,
        (unsigned long long)fp_after_cleanup,
        (unsigned long long)out_metrics->peak_phys_footprint_bytes,
        (unsigned long long)out_metrics->lowest_avail_mem_bytes
    );

    return rc_final;
}
