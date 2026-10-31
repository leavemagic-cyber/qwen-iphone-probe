#include "QwenProbeEngine.h"

#include <TargetConditionals.h>
#include <llama/llama.h>
#include <llama/mtmd.h>
#include <llama/mtmd-helper.h>

#include <mach/mach.h>
#include <os/proc.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <unistd.h>

static pthread_once_t g_backend_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_peak_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_live_peak_footprint = 0;

static void init_llama_backend_once(void) {
    llama_backend_init();
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

void qwen_probe_record_live_footprint(void) {
    uint64_t cur = qwen_probe_phys_footprint();
    pthread_mutex_lock(&g_peak_mutex);
    if (cur > g_live_peak_footprint) {
        g_live_peak_footprint = cur;
    }
    pthread_mutex_unlock(&g_peak_mutex);
}

uint64_t qwen_probe_live_peak_footprint(void) {
    qwen_probe_record_live_footprint();
    pthread_mutex_lock(&g_peak_mutex);
    uint64_t val = g_live_peak_footprint;
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
} peak_sampler_state;

static uint64_t sampler_sample_now(peak_sampler_state * st) {
    uint64_t cur = qwen_probe_phys_footprint();
    pthread_mutex_lock(&st->mutex);
    if (cur > st->peak_bytes) {
        st->peak_bytes = cur;
    }
    pthread_mutex_unlock(&st->mutex);

    pthread_mutex_lock(&g_peak_mutex);
    if (cur > g_live_peak_footprint) {
        g_live_peak_footprint = cur;
    }
    pthread_mutex_unlock(&g_peak_mutex);
    return cur;
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
    if (!out_metrics || !out_text || out_text_cap == 0 || !out_error || out_error_cap == 0) {
        return -1;
    }
    memset(out_metrics, 0, sizeof(*out_metrics));
    out_text[0] = '\0';
    out_error[0] = '\0';

    if (!model_path || !mmproj_path || !rgb_data || !prompt || image_width == 0 || image_height == 0) {
        snprintf(out_error, out_error_cap, "Invalid arguments to qwen_probe_run");
        return -1;
    }

    const char * vname = (variant_name && variant_name[0] != '\0') ? variant_name : "baseline";
    const bool opt_no_fa         = (options && options->disable_flash_attn != 0);
    const bool opt_no_mmap       = (options && options->disable_mmap != 0);
    const bool opt_no_op_offload = (options && options->disable_op_offload != 0);
    const bool opt_cpu_output    = (options && options->cpu_output_override != 0);

    pthread_mutex_lock(&g_peak_mutex);
    g_live_peak_footprint = 0;
    pthread_mutex_unlock(&g_peak_mutex);

    out_metrics->avail_mem_before_load_bytes = qwen_probe_available_memory();

    peak_sampler_state sampler;
    pthread_mutex_init(&sampler.mutex, NULL);
    sampler.running = true;
    sampler.peak_bytes = 0;
    uint64_t initial_fp = sampler_sample_now(&sampler);

    fprintf(stderr,
            "[QWENPROBE] variant=%s step=before_load disable_fa=%d disable_mmap=%d "
            "disable_op_offload=%d cpu_output_override=%d avail_bytes=%llu phys_footprint_bytes=%llu\n",
            vname,
            opt_no_fa ? 1 : 0,
            opt_no_mmap ? 1 : 0,
            opt_no_op_offload ? 1 : 0,
            opt_cpu_output ? 1 : 0,
            (unsigned long long)out_metrics->avail_mem_before_load_bytes,
            (unsigned long long)initial_fp);
    fflush(stderr);

    pthread_t sampler_tid;
    bool sampler_started = (pthread_create(&sampler_tid, NULL, sampler_thread_main, &sampler) == 0);

    int32_t rc_final = 0;
    struct llama_model * model = NULL;
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
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    if (opt_cpu_output) {
        cpu_output_overrides[0].pattern = "^(output|token_embd)\\.weight$";
        cpu_output_overrides[0].buft    = ggml_backend_cpu_buffer_type();
        cpu_output_overrides[1].pattern = NULL;
        cpu_output_overrides[1].buft    = NULL;
        mparams.tensor_buft_overrides   = cpu_output_overrides;
    }

    model = llama_model_load_from_file(model_path, mparams);
    uint64_t fp_after_model = sampler_sample_now(&sampler);
    out_metrics->phys_footprint_after_model_load_bytes = fp_after_model;
    fprintf(stderr,
            "[QWENPROBE] variant=%s step=after_text_model_load ok=%d phys_footprint_bytes=%llu\n",
            vname, model != NULL ? 1 : 0, (unsigned long long)fp_after_model);
    fflush(stderr);
    if (!model) {
        snprintf(out_error, out_error_cap, "llama_model_load_from_file failed");
        rc_final = 1;
        goto cleanup;
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
        if (opt_no_op_offload) {
            cparams.op_offload = false;
        }

        lctx = llama_init_from_model(model, cparams);
        uint64_t fp_after_ctx = sampler_sample_now(&sampler);
        fprintf(stderr,
                "[QWENPROBE] variant=%s step=after_ctx_init ok=%d phys_footprint_bytes=%llu\n",
                vname, lctx != NULL ? 1 : 0, (unsigned long long)fp_after_ctx);
        fflush(stderr);
        if (!lctx) {
            snprintf(out_error, out_error_cap, "llama_init_from_model failed");
            rc_final = 2;
            goto cleanup;
        }
    }

    {
        struct mtmd_context_params vparams = mtmd_context_params_default();
#if TARGET_OS_SIMULATOR
        vparams.use_gpu = false;
#else
        vparams.use_gpu = true;
#endif
        vparams.n_threads = n_threads;
        vparams.warmup = false;
        vparams.image_max_tokens = 512;
        if (opt_no_fa) {
            vparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }

        mctx = mtmd_init_from_file(mmproj_path, model, vparams);
        int64_t t_load_end = llama_time_us();
        out_metrics->model_load_seconds = (double)(t_load_end - t_load_start) / 1e6;

        uint64_t fp_after_mmproj = sampler_sample_now(&sampler);
        fprintf(stderr,
                "[QWENPROBE] variant=%s step=after_mmproj_load ok=%d load_s=%.3f phys_footprint_bytes=%llu\n",
                vname, mctx != NULL ? 1 : 0, out_metrics->model_load_seconds, (unsigned long long)fp_after_mmproj);
        fflush(stderr);
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
        fprintf(stderr,
                "[QWENPROBE] variant=%s step=after_tokenize ok=%d chunks=%zu phys_footprint_bytes=%llu\n",
                vname, tok_rc == 0 ? 1 : 0, n_chunks, (unsigned long long)fp_after_tok);
        fflush(stderr);
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
                fprintf(stderr,
                        "[QWENPROBE] variant=%s step=after_text_chunk_%zu ok=%d is_last=%d phys_footprint_bytes=%llu\n",
                        vname, i, eval_rc == 0 ? 1 : 0, is_last_chunk ? 1 : 0, (unsigned long long)fp_after_txt);
                fflush(stderr);
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
                fprintf(stderr,
                        "[QWENPROBE] variant=%s step=after_image_encode_cold ok=%d encode_s=%.3f phys_footprint_bytes=%llu\n",
                        vname, enc_rc == 0 ? 1 : 0, total_img_encode_cold_s, (unsigned long long)fp_after_enc_cold);
                fflush(stderr);
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
                fprintf(stderr,
                        "[QWENPROBE] variant=%s step=after_image_encode_warm ok=%d encode_warm_s=%.3f phys_footprint_bytes=%llu\n",
                        vname, enc_warm_rc == 0 ? 1 : 0, total_img_encode_warm_s, (unsigned long long)fp_after_enc_warm);
                fflush(stderr);
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
                fprintf(stderr,
                        "[QWENPROBE] variant=%s step=after_image_decode ok=%d phys_footprint_bytes=%llu\n",
                        vname, dec_img_rc == 0 ? 1 : 0, (unsigned long long)fp_after_img_dec);
                fflush(stderr);
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
        int64_t t_first_tok = llama_time_us();
        int64_t ttft_us = (t_first_tok - t_infer_start) - total_img_encode_warm_us;
        if (ttft_us < 0) {
            ttft_us = t_first_tok - t_infer_start;
        }
        out_metrics->time_to_first_token_seconds = (double)ttft_us / 1e6;
        uint64_t fp_after_ttft = sampler_sample_now(&sampler);
        fprintf(stderr,
                "[QWENPROBE] variant=%s step=first_token ttft_s=%.3f phys_footprint_bytes=%llu\n",
                vname, out_metrics->time_to_first_token_seconds, (unsigned long long)fp_after_ttft);
        fflush(stderr);

        int32_t n_generated = 0;
        int32_t n_decode_steps = 0;
        size_t text_pos = 0;
        int64_t t_gen_start = t_first_tok;

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
        uint64_t fp_after_gen = sampler_sample_now(&sampler);

        out_metrics->generated_tokens = n_generated;
        double decode_elapsed_s = (double)(t_gen_end - t_gen_start) / 1e6;
        if (n_decode_steps > 0 && decode_elapsed_s > 0.0) {
            out_metrics->generated_tokens_per_second = (double)n_decode_steps / decode_elapsed_s;
        } else if (n_generated > 0 && out_metrics->time_to_first_token_seconds > 0.0) {
            out_metrics->generated_tokens_per_second = (double)n_generated / out_metrics->time_to_first_token_seconds;
        } else {
            out_metrics->generated_tokens_per_second = 0.0;
        }

        fprintf(stderr,
                "[QWENPROBE] variant=%s step=after_generation tokens=%d tok_per_s=%.3f phys_footprint_bytes=%llu\n",
                vname, n_generated, out_metrics->generated_tokens_per_second, (unsigned long long)fp_after_gen);
        fflush(stderr);
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
    if (model) {
        llama_model_free(model);
    }

    uint64_t fp_after_cleanup = qwen_probe_phys_footprint();
    fprintf(stderr,
            "[QWENPROBE] variant=%s step=after_cleanup phys_footprint_bytes=%llu peak_phys_footprint_bytes=%llu\n",
            vname,
            (unsigned long long)fp_after_cleanup,
            (unsigned long long)out_metrics->peak_phys_footprint_bytes);
    fflush(stderr);

    return rc_final;
}
