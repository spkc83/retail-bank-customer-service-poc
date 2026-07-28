# Dense-to-MoE conversion and routing

This document explains how the pinned 1.5B Qwen checkpoint is expanded into the
released 8.94B Qwen2-MoE checkpoint. All equations use plain text so they render
consistently in GitHub, Hugging Face, terminals, and basic Markdown viewers.

“Copying learned language representations” means copying pretrained weights. It
does not mean copying a separate language database.

## 1. Scope

The base checkpoint is:

```text
Qwen/Qwen2.5-1.5B-Instruct
revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

The conversion preserves the base model’s embeddings, attention,
normalization, and initial output behavior while replacing each dense
feed-forward block.

The released model is domain-adapted. It is not a 9B model trained from random
initialization.

## 2. Notation and dimensions

For one token in one transformer layer:

| Symbol | Meaning | Shape |
|---|---|---:|
| `h` | token hidden state | `1536` |
| `W_gate` | dense gate projection | `8960 × 1536` |
| `W_up` | dense up projection | `8960 × 1536` |
| `W_down` | dense down projection | `1536 × 8960` |
| `W_router` | MoE router matrix | `28 × 1536` |
| `E` | routed expert count | `28` |
| `K` | experts selected per token | `2` |

Each routed expert has intermediate width 2,048. The shared expert retains the
base model’s intermediate width of 8,960.

## 3. Original dense MLP

The base Qwen layer uses a SwiGLU feed-forward block:

```text
D(h) = W_down × [SiLU(W_gate × h) element-wise-multiplied-by (W_up × h)]
```

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

```text
M(h) = shared_gate(h) × shared_expert(h)
     + sum of [routing_weight_i(h) × expert_i(h)]
       for the two selected experts
```

Definitions:

- `shared_expert(h)` is the always-active shared expert;
- `shared_gate(h)` controls the shared expert’s contribution;
- `expert_i(h)` is routed expert `i`;
- `top2(h)` is the set of two experts selected for token `h`;
- `routing_weight_i(h)` is expert `i`’s normalized routing weight.

## 6. Behavior-preserving initialization

The shared expert starts from the original dense MLP:

```text
shared gate projection = W_gate
shared up projection   = W_up
shared down projection = 2 × W_down
shared gate vector     = 0
```

The shared gate is a sigmoid:

```text
shared_gate(h) = sigmoid(shared_gate_vector dot h)
```

Because the shared gate vector starts at zero:

```text
shared_gate(h) = sigmoid(0) = 0.5
```

The initialized shared expert is:

```text
shared_expert(h)
  = 2 × W_down
    × [SiLU(W_gate × h) element-wise-multiplied-by (W_up × h)]
```

Therefore:

```text
shared_gate(h) × shared_expert(h)
  = 0.5 × 2 × W_down
    × [SiLU(W_gate × h) element-wise-multiplied-by (W_up × h)]
  = D(h)
```

Every routed expert starts with a zero down projection:

```text
expert_down_i = 0
```

For routed expert `i`:

```text
expert_i(h)
  = expert_down_i
    × [SiLU(expert_gate_i × h)
       element-wise-multiplied-by
       (expert_up_i × h)]
  = 0
```

Thus, before training:

```text
M(h) = D(h)
```

This equality is the core upcycling invariant. The new routed capacity begins
as a zero residual around the pretrained dense function.

## 7. Routed expert initialization

Each routed expert receives a deterministic 2,048-row slice of the original
dense gate and up projections.

For layer `layer`, expert `expert`, and expert row `row`:

```text
source_row
  = (layer × 997 + expert × 2048 + row) modulo 8960
```

The selected source row is copied into both routed projections with small,
deterministic noise:

```text
expert_gate[expert, row]
  = W_gate[source_row] + gate_noise[layer, expert, row]

expert_up[expert, row]
  = W_up[source_row] + up_noise[layer, expert, row]
```

The seed-controlled noise prevents all experts from beginning with identical
feature detectors.

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

For `T` non-padding tokens, the hidden-state matrix has shape:

```text
H shape = T × 1536
```

The layer router produces 28 logits per token:

```text
Z = H × transpose(W_router)
Z shape = T × 28
```

For one token, expert `i` receives a softmax probability:

```text
p_i(h) = exp(logit_i)
         ÷ sum(exp(logit_j) for j from 1 through 28)
```

The router selects the two largest probabilities:

```text
top2(h) = the indices of the two largest values in p(h)
```

The two selected probabilities are renormalized:

```text
routing_weight_i(h)
  = p_i(h) ÷ sum(p_j(h) for j in top2(h))
```

Only selected experts receive a routing weight. Their routed residual is:

```text
routed_residual(h)
  = sum(routing_weight_i(h) × expert_i(h) for i in top2(h))
```

Example: experts 7 and 19 are selected with probabilities 0.42 and 0.26.

```text
routing_weight_7  = 0.42 ÷ (0.42 + 0.26) ≈ 0.618
routing_weight_19 = 0.26 ÷ (0.42 + 0.26) ≈ 0.382

routed_residual(h)
  ≈ 0.618 × expert_7(h) + 0.382 × expert_19(h)
```

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

At initialization, routed down matrices receive nonzero gradients because
their input features are nonzero:

```text
gradient(total_loss, expert_down_i) is not zero
```

The language-model gradient to routed gate/up weights and router choices is
zero while every routed output is exactly zero.

The auxiliary routing loss can still train the router immediately. Once routed
down projections become nonzero, the language-model loss also informs routing.

## 10. Load balancing and expert-health gates

The training objective includes router load balancing:

```text
total_loss = language_model_loss + 0.01 × router_auxiliary_loss
```

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
absolute logit error no greater than `0.00001` before training.

The expansion does not create 9B parameters of pretrained knowledge. At
initialization, the model is approximately:

```text
pretrained 1.5B language capability
+ 7.4B parameters of mostly inactive routed capacity
```

The restricted banking corpus teaches residual domain behavior. It cannot give
the model the breadth of a 9B checkpoint pretrained on a large general corpus.
