# Dense-to-MoE conversion and routing

This document explains how the pinned 1.5B Qwen checkpoint is expanded into the
released 8.94B Qwen2-MoE checkpoint.

“Copying learned language representations” means copying pretrained weights. It
does not mean copying a separate language database.

## 1. Scope

The base checkpoint is:

```text
Qwen/Qwen2.5-1.5B-Instruct
revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

The conversion preserves the base model’s embeddings, attention, normalization,
and output behavior while replacing each dense feed-forward block.

The released model is domain-adapted. It is not a 9B model trained from random
initialization.

## 2. Notation and dimensions

For one token in one transformer layer:

| Symbol | Meaning | Shape |
|---|---|---:|
| $h$ | token hidden state | $1536$ |
| $W_g$ | dense gate projection | $8960 \times 1536$ |
| $W_u$ | dense up projection | $8960 \times 1536$ |
| $W_d$ | dense down projection | $1536 \times 8960$ |
| $W_r$ | MoE router matrix | $28 \times 1536$ |
| $E$ | routed expert count | $28$ |
| $K$ | experts selected per token | $2$ |

Each routed expert has intermediate width 2,048. The shared expert retains the
base model’s intermediate width of 8,960.

## 3. Original dense MLP

The base Qwen layer uses a SwiGLU feed-forward block:

$$
D(h)
=
W_d
\left[
\operatorname{SiLU}(W_g h)
\odot
(W_u h)
\right].
$$

Here, $\odot$ is element-wise multiplication.

The block is only one part of the learned language function. Knowledge is
distributed across embeddings, attention, normalization, MLP weights, and the
language-model head.

## 4. Tensors copied without modification

Every compatible tensor outside the dense MLP is copied directly:

```python
converted = {
    name: tensor.clone()
    for name, tensor in dense_state.items()
    if ".mlp." not in name
}
```

This preserves:

- the 151,936-token embedding table;
- all 28 attention blocks;
- rotary-position behavior;
- all normalization weights;
- final normalization;
- the tied language-model output head.

The tied `lm_head.weight` does not need an independent copy. It is tied back to
the token embedding after loading.

## 5. New shared-plus-routed block

Each dense MLP becomes one always-active shared expert plus 28 routed experts:

```text
                           ┌── shared expert ───────────────┐
token hidden state ────────┤                                ├── sum ── output
                           └── router → top-2 experts ──────┘
```

The converted block is:

$$
M(h)
=
g_s(h)\,S(h)
+
\sum_{i \in \mathcal{K}(h)}
\alpha_i(h)\,E_i(h).
$$

Definitions:

- $S(h)$ is the shared expert;
- $g_s(h)$ is the shared-expert gate;
- $E_i(h)$ is routed expert $i$;
- $\mathcal{K}(h)$ is the top-2 expert set;
- $\alpha_i(h)$ is the normalized routing weight.

## 6. Behavior-preserving initialization

The shared expert starts from the original dense MLP:

```text
shared gate projection = W_g
shared up projection   = W_u
shared down projection = 2 W_d
shared gate vector     = 0
```

The shared gate is:

$$
g_s(h) = \sigma(w_s^\top h).
$$

Because $w_s = 0$ at initialization:

$$
g_s(h) = \sigma(0) = \frac{1}{2}.
$$

The initialized shared expert is:

$$
S(h)
=
2W_d
\left[
\operatorname{SiLU}(W_g h)
\odot
(W_u h)
\right].
$$

Therefore:

$$
g_s(h)\,S(h)
=
\frac{1}{2}
\cdot
2W_d
\left[
\operatorname{SiLU}(W_g h)
\odot
(W_u h)
\right]
=
D(h).
$$

Every routed expert starts with a zero down projection:

$$
W_{d,i} = 0.
$$

For routed expert $i$:

$$
E_i(h)
=
W_{d,i}
\left[
\operatorname{SiLU}(W_{g,i}h)
\odot
(W_{u,i}h)
\right]
=
0.
$$

Thus, before training:

$$
M(h) = D(h).
$$

This equality is the core upcycling invariant. The new routed capacity begins
as a zero residual around the pretrained dense function.

## 7. Routed expert initialization

Each routed expert receives a deterministic 2,048-row slice of the original
dense gate and up projections.

For layer $\ell$, expert $e$, and expert row $k$:

$$
j_{\ell,e,k}
=
(\ell \cdot 997 + e \cdot 2048 + k)
\bmod 8960.
$$

The selected row is copied into both routed projections:

$$
W_{g,e}[k] = W_g[j_{\ell,e,k}] + \epsilon_{g,\ell,e,k},
$$

$$
W_{u,e}[k] = W_u[j_{\ell,e,k}] + \epsilon_{u,\ell,e,k}.
$$

The noise terms are small, deterministic, and seed-controlled. They prevent all
experts from beginning with identical feature detectors.

The fused gate/up tensor for one layer has shape:

```text
(28 experts, 4096 gate-plus-up rows, 1536 hidden dimensions)
```

Each routed down projection has shape:

```text
(1536 hidden dimensions, 2048 expert dimensions)
```

It starts at zero and learns a residual projection during adaptation.

## 8. Internal token routing

Routing occurs independently for every token in every transformer layer.

For $T$ non-padding tokens, hidden states form:

$$
H \in \mathbb{R}^{T \times 1536}.
$$

The layer router produces 28 logits per token:

$$
Z = HW_r^\top,
\qquad
Z \in \mathbb{R}^{T \times 28}.
$$

For one token, router probabilities are:

$$
p_i(h)
=
\frac{\exp(z_i)}
{\sum_{j=1}^{28}\exp(z_j)}.
$$

The router selects:

$$
\mathcal{K}(h)
=
\operatorname{TopK}\!\left(p(h), 2\right).
$$

Selected probabilities are renormalized:

$$
\alpha_i(h)
=
\frac{p_i(h)}
{\sum_{j \in \mathcal{K}(h)} p_j(h)},
\qquad
i \in \mathcal{K}(h).
$$

The routed residual is:

$$
R(h)
=
\sum_{i \in \mathcal{K}(h)}
\alpha_i(h)\,E_i(h).
$$

If experts 7 and 19 are selected with probabilities 0.42 and 0.26:

$$
\alpha_7 = \frac{0.42}{0.42 + 0.26} \approx 0.618,
$$

$$
\alpha_{19} = \frac{0.26}{0.42 + 0.26} \approx 0.382.
$$

The routed residual is then:

$$
R(h) \approx 0.618E_7(h) + 0.382E_{19}(h).
$$

Another token or layer may select different experts. Experts are not manually
labeled as cards, loans, transfers, or other intents.

## 9. Trainable parameters and gradient flow

The adaptation policy freezes:

- embeddings and tied output head;
- attention and normalization;
- the shared expert;
- routed expert gate and up projections.

It trains:

- every layer’s router matrix;
- every routed expert down projection.

Exact model counts:

| Quantity | Parameters |
|---|---:|
| Total checkpoint | 8,943,713,792 |
| Estimated active per token | 2,073,443,840 |
| Routed down projections | 2,466,250,752 |
| Router matrices | 1,204,224 |
| Total trainable | 2,467,454,976 |

At initialization, routed down matrices receive gradients because their input
features are nonzero:

$$
\frac{\partial \mathcal{L}}{\partial W_{d,i}} \neq 0.
$$

The language-model gradient to routed gate/up weights and router choices is zero
while every routed output is exactly zero.

The auxiliary routing loss can still train the router immediately. Once routed
down projections become nonzero, the language-model loss also informs routing.

## 10. Load balancing and expert-health gates

The training objective includes a router load-balancing loss:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{LM}}
+
0.01\,\mathcal{L}_{\text{aux}}.
$$

Starting at step 250, every layer must satisfy:

- each expert receives at least 0.5% of assignments;
- no expert receives more than 20%;
- normalized routing entropy is at least 0.75;
- auxiliary loss is finite;
- routed down projections receive finite, nonzero gradients.

Counts exclude padding tokens. Gates are evaluated every 250 steps over the
final 50 steps of the interval.

The released 1,000-step run passed every per-layer expert-health gate at steps
250, 500, 750, and 1,000.

## 11. Internal and external routing

The system contains two independent routers.

### External domain and intent router

```text
customer message
    ├── supported banking → run the language model
    └── OOD               → return the exact stock response
```

This CPU DistilBERT model has a binary domain head and a 77-way Banking77 intent
head. It enforces the application boundary before model inference.

### Internal MoE router

```text
token hidden state → select 2 of 28 experts in this layer
```

This router distributes neural computation. It does not decide whether a user
request belongs to the banking domain.

The external router enforces policy. The internal router allocates model
capacity.

## 12. Evidence and limitations

The full conversion and 1,000-step adaptation run completed. The released BF16
checkpoint contains 8,943,713,792 parameters.

The repository tests the dense-equivalence calculation and requires maximum
absolute logit error no greater than $10^{-5}$ before training.

The expansion does not create 9B parameters of pretrained knowledge. At
initialization, the model is approximately:

```text
pretrained 1.5B language capability
+ 7.4B parameters of mostly inactive routed capacity
```

The restricted banking corpus teaches residual domain behavior. It cannot give
the model the breadth of a 9B checkpoint pretrained on a large general corpus.
