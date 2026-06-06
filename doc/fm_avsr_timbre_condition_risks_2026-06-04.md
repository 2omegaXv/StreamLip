# FM-AVSR Timbre Conditioning Risks and Next Options - 2026-06-04

This note records the current issue with the strongest timbre/audio-prompt
conditioning path and possible follow-up designs.

## Current conditioning path

The final checkpoint uses two audio-derived conditions:

```text
audio_prompt.npy  : (38, 512)
timbre_cond.npy   : (1024,)
```

`audio_prompt.npy` is the first 38 normalized Mimi latent frames, about 3.04
seconds. It is injected as a sequence-level condition/prompt path. In the
current config, prompt pooling is also enabled, so prompt information can enter
both as prompt tokens and as a pooled prompt condition.

`timbre_cond.npy` is a fixed-size global summary:

```text
timbre_cond = concat(mean(audio_prompt), std(audio_prompt))
```

This global summary is closer to a true timbre condition. The sequence prompt is
stronger, but it also carries much more than timbre.

## Observed problem: prompt content leakage

The sequence audio prompt is not a pure speaker/timbre representation. It
contains:

- speaker and recording color;
- lexical content from the reference segment;
- local prosody, timing, rhythm, and pauses;
- frame-level acoustic trajectory that is directly decodable by Mimi.

Because it lives in the same latent space as the target, the model can learn to
copy the prompt region rather than use it only as a speaker/style reference.
This is now visible in silent-reference demos: the generated waveform can copy
reference audio into the beginning of the output.

The current export-time workaround is to skip the first 38 Mimi frames
(`3.04s`) in silent-reference output. This is only a hack. The proper fix should
prevent prompt leakage inside the model or condition design.

## Principle: timbre should not depend on sequence length

For speaker/timbre control, the condition should ideally be:

- fixed dimensional;
- weakly dependent on reference duration;
- insensitive to spoken content;
- unable to reconstruct the reference phrase directly;
- stable when reference audio is sampled from different valid segments of the
  same speaker.

The current `(38, 512)` prompt violates this principle because it is a timed
audio-latent sequence. A longer or shorter reference would change the condition
shape or the amount of content available for copying.

## Options to fix or improve timbre control

### Option A: use statistics-only timbre conditioning

Keep `timbre_cond.npy` and remove or disable sequence-level `audio_prompt.npy`
as a default path.

Pros:

- fixed size;
- simple;
- already implemented;
- difficult to copy literal speech content.

Cons:

- may lose useful speaker and recording detail;
- mean/std over Mimi latents may be too weak for high-fidelity voice identity.

This is the safest short-term direction if the priority is avoiding content
leakage.

### Option B: random runtime reference segment, then pool

At inference, randomly choose a valid reference segment from the available
unmasked audio, encode it with Mimi, and immediately reduce it to a fixed-size
condition such as mean/std or learned pooling:

```text
ref audio segment -> Mimi latent -> pooling/statistics -> fixed timbre vector
```

This can be run multiple times and averaged:

```text
q = mean_k(pool(Mimi(ref_segment_k)))
```

Pros:

- does not require the reference to be the first 3 seconds;
- can use any unmasked speech region;
- averaging multiple random segments reduces dependence on specific words;
- keeps the condition fixed-size.

Cons:

- if the raw sequence latent is still exposed to the model, leakage remains;
- random sampling adds nondeterminism unless the seed is fixed;
- needs a voiced-segment selector to avoid silence/noise-only references.

Recommendation: if this is used, expose only the pooled/statistical vector to
the model, not the full sequence prompt.

### Option C: learned content-invariant audio timbre encoder

Train or reuse an audio encoder that outputs a fixed speaker/style embedding:

```text
ref audio -> speaker/timbre encoder -> fixed embedding
```

Candidate encoders:

- ECAPA-TDNN speaker embedding;
- WavLM/HuBERT speaker-style pooled embedding;
- a small learned adapter on top of Mimi latents with aggressive temporal
  pooling and content dropout.

Pros:

- better speaker identity target than raw mean/std;
- condition shape is independent of reference length;
- less direct copying risk than raw Mimi sequence tokens.

Cons:

- requires integrating another model or training an adapter;
- speaker embeddings may not preserve recording color/prosody;
- needs experiments to confirm it improves decoded audio.

### Option D: keep sequence prompt but bottleneck it

If sequence prompt information is still needed, restrict it:

- project `(T, 512)` to one or a few prompt tokens;
- add dropout/noise to prompt latents during training;
- randomly mask prompt time spans;
- use only pooled prompt tokens;
- add an anti-copy loss or prompt-region exclusion.

Pros:

- keeps some benefit of reference audio;
- less disruptive than removing prompt conditioning entirely.

Cons:

- still has leakage risk;
- needs careful ablation;
- may become a brittle compromise.

## Recommended next experiment

The next clean experiment should compare:

1. final current model with sequence prompt, using post-3s export crop;
2. statistics-only `timbre_cond.npy`, no sequence `audio_prompt`;
3. random reference segment at runtime, pooled to fixed `timbre_cond`;
4. optional learned/ECAPA/WavLM speaker embedding if integration time permits.

Primary checks:

- listening quality;
- whether the first generated seconds copy reference content;
- latent corr on matched validation;
- robustness when reference segment changes within the same speaker;
- robustness when reference speaker changes.

The expected direction is that fixed-size pooled/statistical timbre control will
reduce prompt-copying artifacts, while possibly losing some speaker detail. If
the quality loss is large, use a stronger fixed-size speaker/timbre encoder
rather than reintroducing raw time-aligned Mimi prompt tokens.
