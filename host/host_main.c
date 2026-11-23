// Host program (Mac/Linux): generation and modes for checking against MLX.
//   ./microqwen model.bin gen "Once upon a time" [tokens=200] [temperature=0.8] [top_k=40] [seed=1]
//   ./microqwen model.bin logits 12,345,67        - logits of the last position
//   ./microqwen model.bin greedy 12,345,67 60     - greedy continuation (comma-separated ids)
//   ./microqwen model.bin encode "text"           - tokens
//   ./microqwen model.bin wbias "cat,moon" 2 50 12,345,67 - per-step word bonuses (check against word_bias.py)
//   ./microqwen model.bin ppl val.bin 200         - bits/token on the first N windows of max_pos tokens (uint16)
//   environment variable MQ_KV=float: float32 KV cache instead of int8
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "microqwen.h"

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*len);
    if (fread(b, 1, *len, f) != *len) { free(b); b = NULL; }
    fclose(f);
    return b;
}

static int parse_ids(const char *s, int *ids, int max) {
    int n = 0;
    for (const char *p = s; *p && n < max;) { ids[n++] = (int)strtol(p, (char **)&p, 10); if (*p == ',') p++; }
    return n;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.bin gen|logits|greedy|encode ARG ...\n", argv[0]); return 1; }
    const char *kv = getenv("MQ_KV");
    if (kv && !strcmp(kv, "float")) mq_kv_int8 = 0;
    size_t len; uint8_t *buf = read_file(argv[1], &len);
    MQModel m; int rc = buf ? mq_load(&m, buf, len) : -100;
    if (rc) { fprintf(stderr, "failed to load the model: %d\n", rc); return 1; }
    int ids[1024], n = 0;
    const char *mode = argv[2];
    if (!strcmp(mode, "wbias") && argc >= 7) {
        MQWordBias wb; mq_wb_init(&wb, &m, argv[3], atof(argv[4]), atof(argv[5]), argc > 7 ? (float)atof(argv[7]) : 0.0f);
        n = parse_ids(argv[6], ids, 1024);
        static char text[16384]; int tl = 0;
        for (int i = 0; i <= n; i++) {
            int tk[MQ_WB_MAXW]; float b[MQ_WB_MAXW];
            int nb = mq_wb_step(&wb, text, tl, ids, i, tk, b);
            printf("%d", i);
            for (int q = 0; q < nb; q++) printf(" %d:%.4f", tk[q], b[q]);
            printf("\n");
            if (i < n) { int bl; const uint8_t *bb = mq_token_bytes(&m, ids[i], &bl); memcpy(text + tl, bb, bl); tl += bl; }
        }
        return 0;
    }
    if (!strcmp(mode, "fsm") && argc >= 8) {         // model.bin fsm WORDS BASE TAU START ids: the word automaton at every step
        static MQWordBias wb; static MQWordFSM f;
        mq_wb_init(&wb, &m, argv[3], atof(argv[4]), atof(argv[5]), (float)atof(argv[6]));
        mq_fsm_init(&f, &m, &wb, 2, 4.0f);
        n = parse_ids(argv[7], ids, 1024);
        static char text[16384]; int tl = 0;
        for (int i = 0; i <= n; i++) {
            int tk[8]; float b[8]; int nb, form;
            int forced = mq_fsm_step(&f, &wb, &m, text, tl, ids, i, tk, b, &nb, 8, &form);
            printf("%d", i);
            if (forced >= 0) printf(" F%d", forced);
            else {
                for (int q = 0; q < nb; q++) printf(" %d:%.4f", tk[q], b[q]);
                if (form >= 0) {                 // banned letter pieces: their count and the sum of their ids
                    long cnt = 0, sum = 0;
                    for (int t = 0; t < m.vocab; t++) if (mq_tok_letter(&m, t) && !mq_fsm_allowed(&f, form, t)) { cnt++; sum += t; }
                    printf(" X%ld/%ld", cnt, sum);
                }
            }
            printf("\n");
            if (i < n) { int bl; const uint8_t *bb = mq_token_bytes(&m, ids[i], &bl); memcpy(text + tl, bb, bl); tl += bl; }
        }
        return 0;
    }
    if (!strcmp(mode, "loop") && argc >= 6) {        // model.bin loop IMM MULT ids: the penalty at every step
        n = parse_ids(argv[5], ids, 1024);
        for (int i = 0; i <= n; i++) {
            int tk[64]; float p[64];
            int np = mq_loop_step(&m, ids, i, atof(argv[3]), atof(argv[4]), 1.75f, 32, tk, p, 64);
            printf("%d", i);
            for (int q = 0; q < np; q++) printf(" %d:%.4f", tk[q], p[q]);
            printf("\n");
        }
        return 0;
    }
    if (!strcmp(mode, "encode")) {
        n = mq_encode(&m, argv[3], ids, 1024);
        for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? "," : "\n");
    } else if (!strcmp(mode, "ppl")) {
        size_t tl; uint8_t *tb = read_file(argv[3], &tl);
        const uint16_t *toks = (const uint16_t *)tb;
        int T = m.max_pos, nwin = argc > 4 ? atoi(argv[4]) : 100;
        double nll = 0; long cnt = 0;
        for (int w = 0; w < nwin && (size_t)(w + 1) * T * 2 <= tl; w++) {
            const uint16_t *win = toks + (size_t)w * T;
            for (int p = 0; p < T - 1; p++) {
                float *lg = mq_forward(&m, win[p], p);
                float mx = lg[0]; for (int i = 1; i < m.vocab; i++) if (lg[i] > mx) mx = lg[i];
                double se = 0; for (int i = 0; i < m.vocab; i++) se += exp(lg[i] - mx);
                nll += -(lg[win[p + 1]] - mx - log(se)); cnt++;
            }
        }
        printf("KV %s: %ld tokens, %.4f bits/token\n", m.kv8 ? "int8" : "float32", cnt, nll / cnt / log(2.0));
        free(tb);
    } else if (!strcmp(mode, "logits")) {
        n = parse_ids(argv[3], ids, 1024);
        float *lg = NULL;
        for (int i = 0; i < n; i++) lg = mq_forward(&m, ids[i], i);
        for (int i = 0; i < m.vocab; i++) printf("%.6f%c", lg[i], i + 1 < m.vocab ? ' ' : '\n');
    } else if (!strcmp(mode, "greedy") || !strcmp(mode, "gen")) {
        int greedy = !strcmp(mode, "greedy");
        n = greedy ? parse_ids(argv[3], ids, 1024) : mq_encode(&m, argv[3], ids, 1024);
        int steps = argc > 4 ? atoi(argv[4]) : (greedy ? 60 : 200);
        float temp = greedy ? 0 : (argc > 5 ? (float)atof(argv[5]) : 0.8f);
        int topk = argc > 6 ? atoi(argv[6]) : 40;
        uint64_t rng = argc > 7 ? strtoull(argv[7], NULL, 10) : 1;
        if (!greedy) printf("%s", argv[3]);
        for (int i = 0; i < n - 1; i++) mq_forward(&m, ids[i], i);
        int tok = ids[n - 1], pos = n - 1, made = 0;
        double t0 = now();
        while (made < steps && pos < m.max_pos - 1) {
            mq_forward(&m, tok, pos++);
            tok = mq_sample(&m, temp, topk, &rng);
            if (tok <= 2) break;                         // <|endoftext|>, <|im_start|>, <|im_end|>
            made++;
            if (greedy) printf("%d%s", tok, made < steps ? "," : "");
            else { int bl; const uint8_t *b = mq_token_bytes(&m, tok, &bl); fwrite(b, 1, bl, stdout); fflush(stdout); }
        }
        double dt = now() - t0;
        printf("\n");
        if (!greedy) fprintf(stderr, "[%d tokens in %.2f s = %.0f tok/s]\n", made, dt, made / dt);
    }
    mq_free(&m); free(buf);
    return 0;
}
