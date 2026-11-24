// micro-qwen3 for ESP32-S3 — Copyright 2026 Alex Anokhin — Apache License 2.0 (LICENSE, NOTICE)
// A story order on a computer, with the same loop as the board (firmware/main/main.c, generate()): the chat prompt,
// the word bonus, the spelling automaton, the loop penalty, top-k with min-p. Used to pick tested seeds for the page.
//   ./order model.bin "dragon, cookie, school" SEED [temperature=0.6] [top_k=40] [tokens=160]
//   ./order model.bin "dragon, cookie, school" FROM..TO      a scan: one line per seed, no text
// Line format: seed, tokens, words used/total, ended (1: the model stopped by itself), then the story.
// Note: this runs the same integer core, but a computer and the chip round a few float steps differently. Measured
// on 7 seeds: 3 stories matched the board letter for letter, 4 went another way after 100 to 480 characters (and
// -ffp-contract=off on the computer did not change that). So a seed found here is only a candidate: run it on a real
// board before publishing it.
#include "microqwen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WB_BASE 2.0f
#define WB_TAU 100.0f
#define LP_IMM 3.0f     // keep these two equal to firmware/main/main.c
#define LP_MULT 1.0f

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *len = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*len); if (fread(b, 1, *len, f) != *len) { fclose(f); free(b); return NULL; }
    fclose(f); return b;
}

static MQModel M;

static int run(const char *words, uint64_t seed, float temp, int topk, int steps, char *text, int cap, int *used, int *total, int *ended) {
    static char prompt[640];
    snprintf(prompt, sizeof prompt, "<|im_start|>user\nWrite a short story. Use the words: %s.<|im_end|>\n<|im_start|>assistant\n", words);
    int ids[256], n = mq_encode(&M, prompt, ids, 200), tl = 0, ng = 0, count = 0;
    static MQWordBias wb; static MQWordFSM fsm; static int gen[256];
    char w[80]; int wl = 0;                      // the board gets the words comma-separated, without spaces
    for (const char *c = words; *c && wl < 79; c++) if (*c != ' ') w[wl++] = *c;
    w[wl] = 0;
    mq_wb_init(&wb, &M, w, WB_BASE, WB_TAU, 0.5f);
    mq_fsm_init(&fsm, &M, &wb, 2, 4.0f);
    uint64_t rng = seed;
    for (int i = 0; i < n - 1; i++) mq_forward(&M, ids[i], i);
    int tok = ids[n - 1], pos = n - 1; *ended = 0;
    while (count < steps && pos < M.max_pos - 1) {
        mq_forward(&M, tok, pos++);
        int bt[MQ_WB_MAXW + 1]; float bb[MQ_WB_MAXW + 1]; int nb = 0, form = -1;
        int forced = wb.n ? mq_fsm_step(&fsm, &wb, &M, text, tl, gen, ng, bt, bb, &nb, MQ_WB_MAXW + 1, &form) : -1;
        for (int q = 0; q < nb; q++) M.logits[bt[q]] += bb[q];
        if (form >= 0) {
            int nx[16]; int nn = mq_fsm_next(&fsm, form, nx, 16);
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
        tok = forced >= 0 ? forced : mq_sample(&M, temp, topk, &rng);
        if (tok <= 2) { *ended = 1; break; }
        count++;
        int bl; const uint8_t *b = mq_token_bytes(&M, tok, &bl);
        if (ng < 256) gen[ng++] = tok;
        if (tl + bl < cap - 1) { memcpy(text + tl, b, bl); tl += bl; }
    }
    text[tl] = 0;
    *total = wb.n; *used = mq_wb_used(&wb, text, tl);
    return count;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.bin \"word, word, word\" SEED|FROM..TO [temperature] [top_k] [tokens]\n", argv[0]); return 1; }
    size_t len; uint8_t *buf = read_file(argv[1], &len);
    if (!buf || mq_load(&M, buf, len) != 0) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.6f; int topk = argc > 5 ? atoi(argv[5]) : 40, steps = argc > 6 ? atoi(argv[6]) : 160;
    unsigned long a = strtoul(argv[3], NULL, 10), b = a; const char *dd = strstr(argv[3], "..");
    if (dd) b = strtoul(dd + 2, NULL, 10);
    static char text[4096];
    for (unsigned long seed = a; seed <= b; seed++) {
        int used, total, ended, n = run(argv[2], seed, temp, topk, steps, text, sizeof text, &used, &total, &ended);
        for (char *c = text; *c; c++) if (*c == '\n') *c = ' ';
        printf("%lu\t%d\t%d/%d\t%d\t%s\n", seed, n, used, total, ended, text);
    }
    return 0;
}
