// micro-qwen3: inference of the Qwen3 architecture in plain C (Mac and ESP32-S3).
// Weights are int8 with one scale per row; activations are quantized to int8 at the input of every matrix multiply
// (W8A8, the format the ESP32-S3 vector unit supports natively). Everything else is float32.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef MQ_ESP32                // ESP32-S3: aligned activation buffers live in internal SRAM
#include "esp_heap_caps.h"
#define MQ_MALLOC_A16(n) heap_caps_aligned_alloc(16, (n), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define MQ_MALLOC_A16_EXT(n) heap_caps_aligned_alloc(16, (n), MALLOC_CAP_SPIRAM)
#endif
#ifndef MQ_MALLOC_A16_EXT      // aligned buffer in external memory (KV cache, if internal memory is not enough)
#define MQ_MALLOC_A16_EXT(n) MQ_MALLOC_A16(n)
#endif
#ifndef MQ_MALLOC
#include <stdlib.h>
#define MQ_MALLOC(n) malloc(n)
#define MQ_FREE(p) free(p)
#endif
#ifndef MQ_MALLOC_A16          // buffer aligned to 16 bytes (PIE vector loads on ESP32-S3)
#define MQ_MALLOC_A16(n) MQ_MALLOC(n)
#endif

// y = W x, W: [out][in]. W8: q is int8 [out][in], s is f32 [out]. W4: q4 is [out][groups][16] nibbles
// (u = q + 8, low nibbles are weights 0..15 of the group, high nibbles 16..31), s4 is f16 [out][groups], group = 32 inputs.
// W4R (rowscale=1): q4 uses the same nibble packing, but there is one scale per row: s (f32 [out]).
typedef struct { const int8_t *q; const float *s; const uint8_t *q4; const uint16_t *s4; int out, in, groups, bits, rowscale; } MQMat;

typedef struct {
    const float *in_norm, *post_norm, *q_norm, *k_norm;
    MQMat q, k, v, o, gate, up, down;
} MQLayer;

typedef struct {
    int vocab, dim, inter, layers, heads, kv_heads, head_dim, max_pos;
    float rope_theta, eps;
    MQMat emb;                 // embedding table; if has_head == 0 it is also the output layer (tied)
    MQMat head; int has_head;  // separate output layer (version 4)
    const float *final_norm;
    MQLayer *L;
    // tokenizer
    const uint8_t **tok_bytes; uint8_t *tok_len;
    int n_merges; uint16_t *merges;   // triples (a, b, merged) by rank
    // work buffers
    float *x, *xn, *q, *k, *v, *att, *hb, *hb2, *logits, *scores, *rope_cos, *rope_sin;
    int8_t *xq;
    int32_t *gsum;             // W4: per-group sums of the quantized input (correction for the zero point 8)
    int bits;                  // 8 or 4
    int rowscale;              // W4R: one scale per row
    float *kcache, *vcache;    // float32: [layers][max_pos][kv_heads*head_dim]
    // int8 KV cache (mq_kv_int8 = 1): K is [layers][max_pos][kvd], V is transposed [layers][kv_heads][head_dim][max_pos]
    int kv8;
    int8_t *qq, *wq;
    // KV cache with one block per (layer, kv head): K is [pos][head_dim], V is transposed [head_dim][pos].
    // Small blocks (~16 KB) are easier to place in fragmented internal SRAM.
    int8_t *kq_h[64], *vqT_h[64];
    int kv_blocks, kv_blocks_internal;
    float *ks, *vs;            // scales: [layers][max_pos][kv_heads]
} MQModel;

// Load from a buffer (the whole file in memory, or a flash partition mapped into the address space).
// Weights are not copied: pointers point straight into buf. Returns 0 on success.
int mq_load(MQModel *m, const uint8_t *buf, size_t len);
void mq_free(MQModel *m);

// One step: token at position pos -> logits in m->logits[vocab].
float *mq_forward(MQModel *m, int token, int pos);

// Tokenizer (Qwen3-style pre-tokenization for ASCII + BPE by merges). Returns the number of tokens.
int mq_encode(const MQModel *m, const char *text, int *out, int max_out);
// Bytes of a token (for output). Special tokens are an empty string.
const uint8_t *mq_token_bytes(const MQModel *m, int id, int *len);

// "Bonus" for the requested words (cascade + ramp over time), as micro_qwen3/word_bias.py, mode="escalate",
// multi_token=True: an unused word adds base·(u+1)·(1 + t/tau) to the logit of its (next) piece,
// u is how many requested words are already in the text, t is tokens since the last one appeared (tau = 0: no ramp).
// start: cascade start as a share of a full order, u starts at start·(MQ_WB_MAXW − n). 1 is like a full order,
// 0.5 is the board rule u0 = (3 − n)/2 (one word 2·base, two words 1.5·base), 0 is no correction.
#define MQ_WB_MAXW 3
#define MQ_WB_MAXTOK 12
typedef struct {
    int n, used_prev, since;
    float base, tau, u0;
    char word[MQ_WB_MAXW][24]; int wlen[MQ_WB_MAXW];
    int seq[MQ_WB_MAXW][MQ_WB_MAXTOK]; int slen[MQ_WB_MAXW];
} MQWordBias;
// words: separated by commas or spaces, up to 3 words (Latin letters, hyphen, apostrophe). Returns the number of words.
int mq_wb_init(MQWordBias *wb, const MQModel *m, const char *words, float base, float tau, float start);
// text/text_len is the text generated so far (bytes), gen/ngen are its tokens. Writes (token, bonus) pairs and
// returns how many (≤ MQ_WB_MAXW). Equal tokens are not merged: the caller simply adds them to the logits.
int mq_wb_step(MQWordBias *wb, const char *text, int text_len, const int *gen, int ngen, int *tok, float *bias);
// How many requested words are already in the text (for statistics).
int mq_wb_used(const MQWordBias *wb, const char *text, int text_len);

// Trace for visualization (only in builds with -DMQ_TRACE, the board has none of this code): if mq_trace != NULL,
// mq_forward copies the intermediate values of the current step into it.
#ifdef MQ_TRACE
typedef struct {
    float *resid;      // [layers + 1][dim]: residual stream before each layer and after the last one
    float *attn_out;   // [layers][dim]: attention contribution (after the o projection)
    float *mlp_act;    // [layers][inter]: silu(gate) · up, the MLP "neurons"
    float *mlp_out;    // [layers][dim]: MLP contribution (after the down projection)
    float *attn_p;     // [layers][heads][max_pos]: attention weights of the current token over positions 0..pos
    float *final_norm; // [dim]: input of the output layer (after the final RMSNorm)
    float *head_out;   // [layers][heads*head_dim]: attention head outputs (before the o projection)
    float *v_out;      // [layers][kv_heads*head_dim]: value vector of the current token (what is later read from it)
} MQTrace;
extern MQTrace *mq_trace;
// Logit lens: final RMSNorm + output layer applied to any residual stream vector (out is vocab numbers).
void mq_project(MQModel *m, const float *x, float *out);
// Layer weights in float (for flow lines in the visualization): which 0 is o_proj [dim × heads*head_dim], 1 is down [dim × inter].
void mq_dequant(const MQModel *m, int layer, int which, float *out);
#endif

// State machine for the requested words, as micro_qwen3/word_fsm.py: soft about WHEN to insert a word (WordBias), strict about HOW
// to spell it. If k ≥ lock leading pieces of a word match, the next piece is forced; k = 1 gives start_bonus to the
// second piece; if the text ends with a normal form of the word (+s/es/ed/d/ing, capitalized, no space), pieces that
// continue the word with letters are banned, except those leading to another normal form ("warnerer", "examineineine" are not allowed).
#define MQ_FSM_MAXF 120
typedef struct {
    int nf, lock; float start_bonus;
    int flen[MQ_FSM_MAXF]; int fseq[MQ_FSM_MAXF][MQ_WB_MAXTOK];
#ifdef MQ_TRACE
    int fword[MQ_FSM_MAXF];   // which requested word the form belongs to (visualization only)
#endif
} MQWordFSM;
int mq_fsm_init(MQWordFSM *f, const MQModel *m, const MQWordBias *wb, int lock, float start_bonus);
// Returns a forced token (≥ 0), and then nothing else is needed, or −1: then tok/val/*nb are the bonuses (already
// including WordBias), and *form is the index of the word form at the end of the text (≥ 0: apply mq_fsm_allowed to all "letter" pieces).
int mq_fsm_step(MQWordFSM *f, MQWordBias *wb, const MQModel *m, const char *text, int tl, const int *gen, int ng,
                int *tok, float *val, int *nb, int max_out, int *form);
int mq_fsm_allowed(const MQWordFSM *f, int form, int tok);   // can tok follow form `form`
// All tokens that can follow form `form` (continuations to other normal forms); there are few of them.
int mq_fsm_next(const MQWordFSM *f, int form, int *out, int max_out);
int mq_tok_letter(const MQModel *m, int id);                 // the piece starts with a letter (continues a word)

// Loop penalty, as micro_qwen3/loop_penalty.py: an immediate repeat inside a phrase ("hot hot", "tall and tall",
// "very tall and very tall") -> −imm; a repeat of a sequence of k ≥ 2 tokens within window (DRY) -> −mult·base^(k−2).
// Punctuation and newlines break matches. Writes (token, penalty) pairs, unique tokens, ≤ max_out.
#define MQ_LP_MAXK 8
int mq_loop_step(const MQModel *m, const int *gen, int ngen, float imm, float mult, float base, int window,
                 int *tok, float *pen, int max_out);

// Parallel matrix multiply (on ESP32-S3: the second core). If NULL, runs sequentially.
// fn computes output rows [r0, r1) of y from the already quantized input xq with scale sx.
typedef void (*MQRowsFn)(float *y, const MQMat *W, const int8_t *xq, float sx, int r0, int r1);
extern void (*mq_parallel)(MQRowsFn fn, float *y, const MQMat *W, const int8_t *xq, float sx);
// PIE vector kernel (only when built with MQ_PIE): can be switched off at runtime for comparison.
extern int mq_pie_enabled;
// KV cache in int8 (default) or float32; set BEFORE mq_load.
extern int mq_kv_int8;
// KV cache in external memory (PSRAM) instead of internal SRAM; set BEFORE mq_load.
extern int mq_kv_in_psram;
extern uint32_t mq_pie_fallbacks;   // how many calls fell back to scalar code due to alignment/length (should be 0)
int32_t mq_dot_s8(const int8_t *w, const int8_t *x, int n);
int32_t mq_dot_u4(const uint8_t *w, const int8_t *x);
int32_t mq_dot_u4row(const uint8_t *w, const int8_t *x, int groups);   // W4R: whole row, for the self-test   // W4: group of 32 (16 bytes of nibbles), for the self-test

// Profiling (only when built with MQ_PROFILE): microseconds per category, accumulated across calls.
enum { MQP_QKV, MQP_ATTN, MQP_O, MQP_MLP, MQP_HEAD, MQP_OTHER, MQP_N };
extern uint64_t mq_prof_us[MQP_N];
extern uint64_t (*mq_now_us)(void);

// Next token choice: temperature <= 0 is argmax; otherwise top-k sampling. rng is the xorshift state.
// Tail cut after top-k: min-p drops candidates less likely than mq_min_p times the best one (default 0.02: 50 times
// less likely); top-p keeps the most likely ones until their sum reaches mq_top_p (1 = off).
extern float mq_min_p, mq_top_p;
int mq_sample(const MQModel *m, float temperature, int top_k, uint64_t *rng);
