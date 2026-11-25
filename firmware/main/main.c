// micro-qwen3 on the ESP32-S3: web demo.
// The board starts its own Wi-Fi network "micro-qwen3" and serves a page at http://192.168.4.1: a chat where the story
// is generated on the chip and streams to the browser token by token (Server-Sent Events).
// Inference is the same microqwen.c as on the Mac. The USB console works too: a line in, a story out.
// Core split: CORE 1 runs only the model (the generation task, one core, matrix work not split);
// CORE 0 runs Wi-Fi, the web server and the console. A ring buffer of token bytes sits between them, so generation never waits for the network.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_netif.h"
#include "esp_partition.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/message_buffer.h"
#include "lwip/sockets.h"
#include "freertos/idf_additions.h"
#include "freertos/task.h"
#include "microqwen.h"
#include "nvs_flash.h"

#define AP_SSID "micro-qwen3"
#define AP_PASS "microqwen"

extern const char index_html_start[] asm("_binary_index_html_start");
extern const char index_html_end[] asm("_binary_index_html_end");

static MQModel M;
static SemaphoreHandle_t model_lock;   // one generation at a time

// ---------------------------------------------------------------- generation with a callback per token
typedef int (*TokenCb)(void *ctx, const uint8_t *bytes, int len);   // return 0 to stop
typedef struct { int n; double seconds, tps; int words_used, words_total; uint32_t seed; } GenStats;

// Default word order: cascade 2 -> 4 -> 6 (tuned on 200 TinyStoriesInstruct prompts), ramp over time
#define WB_BASE 2.0f
#define WB_TAU 100.0f
// The loop penalty is ON. It was off for a while: without a tail cut the model dodged it with mangled words
// ("warnerer", "tighterer"). With min-p 0.02 and the spelling automaton that no longer happens. Measured on 300
// stories from orders the model had never seen (4-bit weights, min-p 0.02): stories with a made-up word 2.3% -> 0.7%,
// looping stories 13% -> 6%, all requested words in 296 -> 298 of 300. To switch it off: LP_IMM 0, LP_MULT 0.
#define LP_IMM 3.0f
#define LP_MULT 1.0f

static GenStats generate(const char *prompt, const char *words, float wbase, float wtau,
                         int steps, float temp, int topk, uint64_t seed, TokenCb cb, void *ctx) {
    GenStats st = {0};
    st.seed = (uint32_t)seed;
    int ids[256];
    int n = mq_encode(&M, prompt, ids, 200);
    if (n == 0) return st;
    // bonus for requested words: the answer text and tokens tell which words are already there
    static MQWordBias wb;
    // automaton: a requested word is written whole, without junk suffixes; ~6 KB in PSRAM so it does not push out the KV cache
    static MQWordFSM *fsm_p;
    if (!fsm_p) fsm_p = heap_caps_malloc(sizeof(MQWordFSM), MALLOC_CAP_SPIRAM);
    MQWordFSM *fsm = fsm_p;
    static char text[2048];
    static int gen[256];
    int tl = 0, ng = 0;
    mq_wb_init(&wb, &M, words ? words : "", wbase, wtau, 0.5f);   // cascade start u0 = (3 - n)/2: 1 word 99.5%, 2 words 100%, NLL almost as without the bonus
    mq_fsm_init(fsm, &M, &wb, 2, 4.0f);
    uint64_t rng = seed;
    for (int i = 0; i < n - 1; i++) mq_forward(&M, ids[i], i);
    int64_t t1 = esp_timer_get_time();
    int tok = ids[n - 1], pos = n - 1;
    while (st.n < steps && pos < M.max_pos - 1) {
        mq_forward(&M, tok, pos++);
        int bt[MQ_WB_MAXW + 1]; float bb[MQ_WB_MAXW + 1]; int nb = 0, form = -1;
        int forced = wb.n ? mq_fsm_step(fsm, &wb, &M, text, tl, gen, ng, bt, bb, &nb, MQ_WB_MAXW + 1, &form) : -1;
        for (int q = 0; q < nb; q++) M.logits[bt[q]] += bb[q];
        if (form >= 0) {   // a requested word was just completed: do not continue it with letters ("warnerer")
            int nx[16]; int nn = mq_fsm_next(fsm, form, nx, 16);
            for (int t = 0; t < M.vocab; t++) {
                if (!mq_tok_letter(&M, t)) continue;
                int ok = 0;
                for (int i = 0; i < nn; i++) ok |= nx[i] == t;
                if (!ok) M.logits[t] = -1e30f;
            }
        }
        if (wb.n && (LP_IMM != 0 || LP_MULT != 0)) {   // loop penalty; tokens the word bonus is pushing right now are exempt
            static int lt[48]; static float lv[48];
            int nl = mq_loop_step(&M, gen, ng, LP_IMM, LP_MULT, 1.75f, 32, lt, lv, 48);
            for (int q = 0; q < nl; q++) {
                int pushed = 0;
                for (int r = 0; r < nb; r++) pushed |= bt[r] == lt[q];
                if (!pushed) M.logits[lt[q]] += lv[q];
            }
        }
        tok = forced >= 0 ? forced : mq_sample(&M, temp, topk, &rng);   // spelling a word: the automaton sets the piece
        if (tok <= 2) break;   // <|endoftext|>, <|im_start|>, <|im_end|>: the story or the answer is over
        st.n++;
        int bl; const uint8_t *b = mq_token_bytes(&M, tok, &bl);
        if (ng < 256) gen[ng++] = tok;
        if (tl + bl < (int)sizeof text) { memcpy(text + tl, b, bl); tl += bl; }
        if (cb && !cb(ctx, b, bl)) break;
        if ((st.n & 15) == 0) vTaskDelay(1);
    }
    st.words_total = wb.n;
    st.words_used = mq_wb_used(&wb, text, tl);
    st.seconds = (esp_timer_get_time() - t1) / 1e6;
    st.tps = st.seconds > 0 ? st.n / st.seconds : 0;
    return st;
}

// ---------------------------------------------------------------- generation task (core 1)
typedef struct { char prompt[640]; char words[80]; float wbase, wtau; float temp; int topk, steps; uint32_t seed; int to_console; } GenJob;   // 640: a chat prompt with options
static QueueHandle_t job_q;               // generation jobs (core 0 -> core 1)
static MessageBufferHandle_t tok_mb;      // tokens (core 1 -> core 0): one message = one token
static volatile int gen_cancel, gen_done, gen_tokens;
static GenStats gen_last;

static int to_buffer(void *ctx, const uint8_t *b, int len) {
    if (gen_cancel) return 0;
    xMessageBufferSend(tok_mb, b, len, portMAX_DELAY);  // buffer full means a slow client, so wait
    gen_tokens++;
    return !gen_cancel;
}

static int to_console(void *ctx, const uint8_t *b, int len) { fwrite(b, 1, len, stdout); fflush(stdout); gen_tokens++; return 1; }

#ifdef MQ_PROFILE
static uint64_t prof_now(void) { return (uint64_t)esp_timer_get_time(); }
#endif
static void gen_worker(void *arg) {
#ifdef MQ_PROFILE
    mq_now_us = prof_now;
#endif
    GenJob job;
    for (;;) {
        xQueueReceive(job_q, &job, portMAX_DELAY);
        gen_cancel = 0; gen_tokens = 0;
        // seed 0 means "pick one": a random seed from the chip. A given seed repeats the same story exactly,
        // which is how the page offers tested orders and lets a guest share one.
        uint32_t seed = job.seed ? job.seed : 1 + esp_random() % 99999;
        gen_last = generate(job.prompt, job.words, job.wbase, job.wtau, job.steps, job.temp, job.topk, seed,
                            job.to_console ? to_console : to_buffer, NULL);
        gen_done = 1;
    }
}

// Queue a job and (for the console) wait until it ends. Called with model_lock held.
static void submit(const char *prompt, const char *words, float wbase, float wtau, int steps, float temp, int topk, uint32_t seed, int to_cons) {
    static GenJob job;
    strlcpy(job.prompt, prompt, sizeof job.prompt);
    strlcpy(job.words, words ? words : "", sizeof job.words);
    job.wbase = wbase; job.wtau = wtau;
    job.steps = steps; job.temp = temp; job.topk = topk; job.seed = seed; job.to_console = to_cons;
    xMessageBufferReset(tok_mb);
    gen_done = 0;
    xQueueSend(job_q, &job, portMAX_DELAY);
}

// ---------------------------------------------------------------- HTTP
static esp_err_t h_index(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    return httpd_resp_send(req, index_html_start, index_html_end - index_html_start);
}

static esp_err_t h_info(httpd_req_t *req) {
    char buf[384];
    int d = M.dim, hd = M.head_dim;
    int per_layer = d * M.heads * hd * 2 + 2 * d * M.kv_heads * hd + 3 * d * M.inter + 2 * d + 2 * hd;
    int head_params = M.has_head ? M.vocab * d : 0;   // separate output layer (model with a baked-in rotation)
    int params = M.vocab * d + head_params + M.layers * per_layer + d;
    snprintf(buf, sizeof buf,
             "{\"chip\":\"ESP32-S3\",\"cpu_mhz\":240,\"format\":\"%s\",\"params\":%d,\"vocab\":%d,\"hidden\":%d,"
             "\"head_params\":%d,\"layers\":%d,\"heads\":%d,\"head_dim\":%d,\"context\":%d,\"free_internal_kb\":%u,\"free_psram_kb\":%u}",
             M.bits == 8 ? "int8" : (M.rowscale ? "int4 W4R" : "int4"), params, M.vocab, d, head_params, M.layers, M.heads, hd,
             M.max_pos, (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, buf);
}

static void url_decode(char *s) {
    char *o = s;
    for (; *s; s++, o++) {
        if (*s == '+') *o = ' ';
        else if (*s == '%' && s[1] && s[2]) { char h[3] = {s[1], s[2], 0}; *o = (char)strtol(h, NULL, 16); s += 2; }
        else *o = *s;
    }
    *o = 0;
}

// Send a piece of text as an SSE event: data: ["<json string>", token count]\n\n
static int sse_send(httpd_req_t *req, const uint8_t *b, int len, int ntok) {
    static char out[1600];
    int o = snprintf(out, sizeof out, "data: [\"");
    for (int i = 0; i < len && o < (int)sizeof out - 24; i++) {
        unsigned char ch = b[i];
        if (ch == '"' || ch == '\\') { out[o++] = '\\'; out[o++] = ch; }
        else if (ch == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
        else if (ch == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
        else if (ch == '\t') { out[o++] = '\\'; out[o++] = 't'; }
        else if (ch < 0x20) o += snprintf(out + o, sizeof out - o, "\\u%04x", ch);
        else out[o++] = ch;
    }
    o += snprintf(out + o, sizeof out - o, "\",%d]\n\n", ntok);
    return httpd_resp_send_chunk(req, out, o) == ESP_OK;
}

// ---- request queue "like Gradio": tickets in order, the client gets its position while it waits
#define MAX_WAITING 4
static SemaphoreHandle_t q_mutex;
static uint32_t next_ticket, now_serving;
static int waiting;

typedef struct { httpd_req_t *req; char prompt[640]; char words[80]; float wbase, wtau; float temp; int topk, steps; uint32_t seed; uint32_t ticket; } WebJob;

static void web_job_task(void *arg) {
    WebJob *j = arg;
    httpd_req_t *req = j->req;
    int alive = 1;
    // 1) wait for our turn, reporting the position every half second
    for (;;) {
        xSemaphoreTake(q_mutex, portMAX_DELAY);
        uint32_t ahead = j->ticket - now_serving;
        xSemaphoreGive(q_mutex);
        if (ahead == 0) break;
        if (alive) {
            char ev[64];
            snprintf(ev, sizeof ev, "event: queue\ndata: {\"pos\":%lu}\n\n", (unsigned long)ahead + 1);
            if (httpd_resp_sendstr_chunk(req, ev) != ESP_OK) alive = 0;   // left the queue: we still skip its turn
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
    // 2) our turn: generation on core 1, token stream to the browser
    GenStats st = {0};
    if (alive) {
        xSemaphoreTake(model_lock, portMAX_DELAY);
        printf("[web] ticket %lu: \"%s\" t=%.2f k=%d n=%d\n", (unsigned long)j->ticket, j->prompt, j->temp, j->topk, j->steps);
        submit(j->prompt, j->words, j->wbase, j->wtau, j->steps, j->temp, j->topk, j->seed, 0);
        // each token is its own event, sent as soon as it is ready: generation on core 1 does not wait for the network,
        // so sending in small pieces does not slow the model and the text flows smoothly
        int one = 1;
        setsockopt(httpd_req_to_sockfd(req), IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);   // do not merge small packets
        uint8_t tokb[64];
        for (;;) {
            size_t got = xMessageBufferReceive(tok_mb, tokb, sizeof tokb, pdMS_TO_TICKS(40));
            if (got && alive && !sse_send(req, tokb, got, 1)) { alive = 0; gen_cancel = 1; }   // "Stop"
            if (gen_done && xMessageBufferIsEmpty(tok_mb)) break;
        }
        st = gen_last;
        xSemaphoreGive(model_lock);
        printf("[web] %d tokens in %.2f s = %.1f tok/s%s\n", st.n, st.seconds, st.tps, alive ? "" : " (stopped)");
    }
    // 3) pass the turn to the next client and close the request
    xSemaphoreTake(q_mutex, portMAX_DELAY);
    now_serving++;
    waiting--;
    xSemaphoreGive(q_mutex);
    if (alive) {
        char done[160];
        snprintf(done, sizeof done, "event: done\ndata: {\"n\":%d,\"s\":%.3f,\"tps\":%.2f,\"wu\":%d,\"wn\":%d,\"seed\":%lu}\n\n",
                 st.n, st.seconds, st.tps, st.words_used, st.words_total, (unsigned long)st.seed);
        httpd_resp_sendstr_chunk(req, done);
        httpd_resp_send_chunk(req, NULL, 0);
    }
    httpd_req_async_handler_complete(req);
    free(j);
    vTaskDeleteWithCaps(NULL);
}

static esp_err_t h_generate(httpd_req_t *req) {
    WebJob *j = heap_caps_calloc(1, sizeof *j, MALLOC_CAP_SPIRAM);
    // the encoded query string (%3C%7C... for chat tokens) is up to CONFIG_HTTPD_MAX_URI_LEN, kept in PSRAM, not on the stack
    const size_t QN = 1024;
    char *q = heap_caps_calloc(1, QN, MALLOC_CAP_SPIRAM), v[16];
    strcpy(j->prompt, "Once upon a time");
    j->temp = 0.8f; j->topk = 40; j->steps = 160; j->wbase = WB_BASE; j->wtau = WB_TAU;
    if (q && httpd_req_get_url_query_str(req, q, QN) == ESP_OK) {
        if (httpd_query_key_value(q, "prompt", j->prompt, sizeof j->prompt) == ESP_OK) url_decode(j->prompt);
        if (httpd_query_key_value(q, "t", v, sizeof v) == ESP_OK) j->temp = strtof(v, NULL);
        if (httpd_query_key_value(q, "k", v, sizeof v) == ESP_OK) j->topk = atoi(v);
        if (httpd_query_key_value(q, "n", v, sizeof v) == ESP_OK) j->steps = atoi(v);
        // requested words (comma-separated) and bonus: wb is the cascade base, wt the ramp τ (0: no ramp)
        if (httpd_query_key_value(q, "w", j->words, sizeof j->words) == ESP_OK) url_decode(j->words);
        if (httpd_query_key_value(q, "wb", v, sizeof v) == ESP_OK) j->wbase = strtof(v, NULL);
        if (httpd_query_key_value(q, "wt", v, sizeof v) == ESP_OK) j->wtau = strtof(v, NULL);
        if (httpd_query_key_value(q, "seed", v, sizeof v) == ESP_OK) j->seed = (uint32_t)strtoul(v, NULL, 10);   // 0 or absent: a random one
    }
    free(q);
    if (j->steps < 1) j->steps = 1;
    if (j->steps > 240) j->steps = 240;
    httpd_resp_set_type(req, "text/event-stream");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");

    xSemaphoreTake(q_mutex, portMAX_DELAY);
    int full = waiting >= MAX_WAITING;
    if (!full) { j->ticket = next_ticket++; waiting++; }
    xSemaphoreGive(q_mutex);
    if (full) {                                   // queue full: say so right away
        free(j);
        httpd_resp_sendstr_chunk(req, "event: busy\ndata: {}\n\n");
        return httpd_resp_send_chunk(req, NULL, 0);
    }
    // the rest runs in its own task: the server stays free to serve the page to others while this client waits
    httpd_req_t *copy = NULL;
    httpd_req_async_handler_begin(req, &copy);
    j->req = copy;
    xTaskCreatePinnedToCoreWithCaps(web_job_task, "mq_web", 6144, j, 5, NULL, 0, MALLOC_CAP_SPIRAM);
    return ESP_OK;
}

static void start_web(void) {
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t wcfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wcfg));
    wifi_config_t ap = {.ap = {.ssid = AP_SSID, .ssid_len = sizeof(AP_SSID) - 1, .password = AP_PASS,
                               .channel = 6, .max_connection = 4, .authmode = WIFI_AUTH_WPA2_PSK}};
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_start());

    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 6144;     // generation runs in its own task on core 1, so the server needs a smaller stack
    cfg.core_id = 0;
    cfg.lru_purge_enable = true;
    httpd_handle_t srv = NULL;
    ESP_ERROR_CHECK(httpd_start(&srv, &cfg));
    httpd_uri_t u1 = {.uri = "/", .method = HTTP_GET, .handler = h_index};
    httpd_uri_t u2 = {.uri = "/api/info", .method = HTTP_GET, .handler = h_info};
    httpd_uri_t u3 = {.uri = "/api/generate", .method = HTTP_GET, .handler = h_generate};
    httpd_register_uri_handler(srv, &u1);
    httpd_register_uri_handler(srv, &u2);
    httpd_register_uri_handler(srv, &u3);
    printf("\nWi-Fi: network \"%s\", password \"%s\" -> open http://192.168.4.1\n", AP_SSID, AP_PASS);
}

// ---------------------------------------------------------------- console
static void pie_selftest(void) {
    int8_t *a = heap_caps_aligned_alloc(16, 336, MALLOC_CAP_INTERNAL), *b = heap_caps_aligned_alloc(16, 336, MALLOC_CAP_INTERNAL);
    int bad = !a || !b;
    for (int trial = 0; trial < 100 && !bad; trial++) {
        for (int i = 0; i < 336; i++) { a[i] = (int8_t)(esp_random() & 0xff); b[i] = (int8_t)(esp_random() & 0xff); }
        mq_pie_enabled = 0; int32_t r1 = mq_dot_s8(a, b, 336), r2 = mq_dot_u4row((const uint8_t *)a, b, 10);
        mq_pie_enabled = 1; int32_t g1 = mq_dot_s8(a, b, 336), g2 = mq_dot_u4row((const uint8_t *)a, b, 10);
        bad = (r1 != g1) || (r2 != g2);
    }
    free(a); free(b);
    mq_pie_enabled = !bad;
    printf("PIE self-test: %s\n", bad ? "FAILED, vector kernel disabled" : "OK");
}

static void heap_report(const char *when) {
    printf("[memory %s] internal free %u KB (largest block %u KB), PSRAM %u KB\n", when,
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
           (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
}

void app_main(void) {
    usb_serial_jtag_driver_config_t ucfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_serial_jtag_driver_install(&ucfg);
    usb_serial_jtag_vfs_use_driver();
    setvbuf(stdin, NULL, _IONBF, 0);
    vTaskDelay(pdMS_TO_TICKS(1000));
    printf("\n=== micro-qwen3 on ESP32-S3: web demo ===\n");
    model_lock = xSemaphoreCreateMutex();
    q_mutex = xSemaphoreCreateMutex();
    job_q = xQueueCreate(1, sizeof(GenJob));
    tok_mb = xMessageBufferCreate(4096);

    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    const void *flash_ptr; esp_partition_mmap_handle_t h;
    if (!part || esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &flash_ptr, &h) != ESP_OK ||
        memcmp(flash_ptr, "MQW3", 4) != 0) { printf("no model in the model partition\n"); return; }
    size_t len = part->size;   // the whole partition: the model file size is not stored there, and mq_load checks bounds itself
    uint8_t *psram = heap_caps_aligned_alloc(16, len, MALLOC_CAP_SPIRAM);
    if (!psram) { printf("not enough PSRAM (%u KB) for a copy of the model\n", (unsigned)(len / 1024)); return; }
    memcpy(psram, flash_ptr, len);

    heap_report("before Wi-Fi");
    start_web();   // Wi-Fi and the server start before the model: they need internal SRAM, the KV cache takes the rest
    heap_report("after Wi-Fi");
    int rc = mq_load(&M, psram, len);
    if (rc) {      // not enough internal SRAM for the int8 KV cache: retry with the KV cache in PSRAM
        printf("KV cache did not fit in internal SRAM (%d), using PSRAM\n", rc);
        mq_kv_in_psram = 1;
        rc = mq_load(&M, psram, len);
    }
    if (rc) { printf("failed to load the model: %d\n", rc); return; }
    printf("model: W%d%s, vocab %d, hidden %d, layers %d; KV cache: %d of %d blocks in internal SRAM\n",
           M.bits, M.rowscale ? "R" : "", M.vocab, M.dim, M.layers, M.kv_blocks_internal, 2 * M.kv_blocks);
    heap_report("after the model");
    pie_selftest();
    mq_parallel = NULL;   // the whole model on one core: no sync between cores
    xTaskCreatePinnedToCore(gen_worker, "mq_gen", 6144, NULL, configMAX_PRIORITIES - 3, NULL, 1);
    printf("the model runs on core 1; Wi-Fi, the web server and the console on core 0\n");

    xSemaphoreTake(model_lock, portMAX_DELAY);
    submit("Once upon a time", NULL, 0, 0, 64, 0, 40, 0, 1);
    while (!gen_done) vTaskDelay(pdMS_TO_TICKS(20));
    xSemaphoreGive(model_lock);
    GenStats st = gen_last;
    printf("\nbenchmark: %d tokens, %.1f tok/s\n", st.n, st.tps);

    printf("\nConsole: type the start of a story and press Enter, or \"/words cat moon cake\" to order a story\n> ");
    fflush(stdout);
    char line[200];
    for (;;) {
        if (!fgets(line, sizeof line, stdin)) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) { printf("> "); fflush(stdout); continue; }
        // "/seed 35": the next story uses this seed (once), so a story can be repeated exactly
        // "/words cat moon cake": a story order in the Qwen chat format, as on the page
        static char chat[640];
        const char *prompt = line, *words = NULL;
        static uint32_t next_seed;
#ifdef MQ_PROFILE
        // "/prof": time per token by part of the model since the last /prof (profiling builds only)
        if (!strncmp(line, "/prof", 5)) {
            static const char *nm[MQP_N] = {"qkv", "attn", "o", "mlp", "head", "other"};
            uint64_t tot = 0; for (int k = 0; k < MQP_N; k++) tot += mq_prof_us[k];
            for (int k = 0; k < MQP_N; k++) printf("%-6s %8llu us %5.1f%%\n", nm[k], (unsigned long long)mq_prof_us[k], tot ? 100.0 * mq_prof_us[k] / tot : 0.0);
            memset(mq_prof_us, 0, sizeof mq_prof_us); printf("> "); fflush(stdout); continue;
        }
#endif
        if (!strncmp(line, "/seed ", 6)) { next_seed = (uint32_t)strtoul(line + 6, NULL, 10); printf("seed %lu for the next story\n> ", (unsigned long)next_seed); fflush(stdout); continue; }
        if (!strncmp(line, "/words ", 7)) {
            words = line + 7;
            char w[3][17] = {{0}};
            int nw = sscanf(line + 7, "%16s %16s %16s", w[0], w[1], w[2]);
            snprintf(chat, sizeof chat, "<|im_start|>user\nWrite a short story. Use the words: %s%s%s%s%s."
                     "<|im_end|>\n<|im_start|>assistant\n",
                     w[0], nw > 1 ? ", " : "", w[1], nw > 2 ? ", " : "", w[2]);
            prompt = chat;
        }
        xSemaphoreTake(model_lock, portMAX_DELAY);
        printf("%s", prompt == chat ? "[order] " : line);
        submit(prompt, words, WB_BASE, WB_TAU, words ? 160 : 240, 0.6f, 40, next_seed, 1);   // as the page: 160 tokens, temperature 0.6
        next_seed = 0;
        while (!gen_done) vTaskDelay(pdMS_TO_TICKS(20));
        st = gen_last;
        xSemaphoreGive(model_lock);
        if (st.words_total) printf("\n[%d tokens, %.1f tok/s, words %d/%d, seed %lu]\n> ", st.n, st.tps, st.words_used, st.words_total, (unsigned long)st.seed);
        else printf("\n[%d tokens, %.1f tok/s]\n> ", st.n, st.tps);
        fflush(stdout);
    }
}
