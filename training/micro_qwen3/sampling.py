import mlx.core as mx

from .model import Qwen3ForCausalLM


def generate(
    model: Qwen3ForCausalLM,
    prompt_ids: list[int],
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    stop_ids: set[int] | None = None,
    top_p: float | None = None,
    logit_bias=None,
):
    """Autoregressive generation with a KV cache. Yields tokens one at a time.

    logit_bias: a function (tokens generated so far) -> {token: logit bonus}; called at every step
    before temperature and cutoffs. This is how requested words get a bonus that is removed once they are used.
    """
    generated: list[int] = []
    max_len = model.cfg.max_position_embeddings
    prompt_ids = prompt_ids[-(max_len - 1):]
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids]), cache)[:, -1, :]
    for _ in range(min(max_tokens, max_len - len(prompt_ids))):
        if logit_bias is not None:
            bias = logit_bias(generated)
            if bias:
                ids = mx.array(list(bias))
                logits = logits.at[:, ids].add(mx.array(list(bias.values()), dtype=logits.dtype))
        if temperature == 0:
            tok = mx.argmax(logits, axis=-1)
        else:
            logits = logits / temperature
            if top_k:
                kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = mx.where(logits < kth, -mx.inf, logits)
            if top_p and top_p < 1:
                # nucleus: keep the most likely tokens until their total probability reaches top_p
                order = mx.argsort(-logits, axis=-1)
                probs = mx.take_along_axis(mx.softmax(logits, axis=-1), order, axis=-1)
                drop = (mx.cumsum(probs, axis=-1) - probs) >= top_p          # the first token always stays
                cut = mx.zeros_like(logits).at[mx.arange(logits.shape[0])[:, None], order].add(drop.astype(logits.dtype))
                logits = mx.where(cut > 0, -mx.inf, logits)
            tok = mx.random.categorical(logits)
        t = tok.item()
        if stop_ids and t in stop_ids:
            return
        generated.append(t)
        yield t
        logits = model(tok[:, None], cache)[:, -1, :]
