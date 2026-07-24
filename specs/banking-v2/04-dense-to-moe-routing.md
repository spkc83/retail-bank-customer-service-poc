# Dense-to-MoE conversion and routing

“Copies its learned language representations” means copying the pretrained
model’s weights, not copying a separate language database or set of embeddings.

The pretrained 1.5B model has already learned useful transformations for
syntax, vocabulary, dialogue structure, and common concepts. The conversion
preserves those transformations while replacing each dense feed-forward block
with a shared-plus-routed MoE block.

## 1. What the pretrained model contains

Each Qwen transformer layer is approximately:

```text
hidden state
    │
    ├── self-attention
    │
    └── dense MLP
            gate projection
            up projection
            activation
            down projection
```

The dense MLP computes:

\[
\operatorname{DenseMLP}(h)
=
W_{\text{down}}
\left(
\operatorname{SiLU}(W_{\text{gate}}h)
\odot
W_{\text{up}}h
\right)
\]

Here, `h` is the contextual representation of one token.

Language knowledge is distributed throughout:

- token embeddings;
- attention projections;
- layer-normalization weights;
- MLP projections;
- final normalization;
- tied language-model output head.

There is no single “English-language component.” The behavior emerges from all
these matrices working together.

## 2. What is copied unchanged

The base is pinned to `Qwen/Qwen2.5-1.5B-Instruct`. The MoE model deliberately
keeps the same core dimensions:

- vocabulary: 151,936
- hidden size: 1,536
- layers: 28
- query heads: 12
- key/value heads: 2
- dense intermediate size: 8,960

Every tensor outside the dense MLPs is copied directly:

```python
converted = {
    key: clone(value)
    for key, value in dense_state.items()
    if ".mlp." not in key
}
```

That preserves:

- the complete tokenizer embedding table;
- all 28 self-attention blocks;
- positional/RoPE behavior;
- normalization weights;
- the final language-model head.

The tied `lm_head.weight` is omitted from the converted state and tied back to
the token embedding after loading.

## 3. Replacing the dense MLP with MoE

Each of the 28 layers becomes:

```text
                       ┌── shared dense expert ─────────┐
hidden state ──────────┤                                ├── add ── output
                       └── router → selected experts ───┘
```

The MoE block contains:

- one shared expert, always active;
- 28 routed experts;
- a router that selects two routed experts for each token;
- a learned gate controlling the shared expert.

Mathematically:

\[
\operatorname{MoE}(h)
=
\sigma(w_s h)\operatorname{SharedExpert}(h)
+
\sum_{i \in \operatorname{Top2}(h)}
p_i(h)\operatorname{Expert}_i(h)
\]

## 4. How the original dense behavior is preserved

The shared expert receives the original dense MLP weights:

```text
shared gate projection = original gate projection
shared up projection   = original up projection
shared down projection = 2 × original down projection
shared gate weight     = 0
```

Because the shared gate starts at zero:

\[
\sigma(0) = 0.5
\]

Therefore:

\[
0.5 \times
\left(
2W_{\text{down}}
\left[
\operatorname{SiLU}(W_{\text{gate}}h)
\odot W_{\text{up}}h
\right]
\right)
=
\operatorname{DenseMLP}(h)
\]

So the shared branch initially reproduces the original dense MLP.

Meanwhile, every routed expert’s down projection starts at zero:

\[
W_{\text{expert-down}} = 0
\]

Consequently:

\[
\operatorname{Expert}_i(h)=0
\]

regardless of which experts the router selects. At initialization:

\[
\operatorname{MoE}(h)
\approx
\operatorname{DenseMLP}(h)
\]

This is the essential upcycling mechanism: the model starts as the pretrained
1.5B model’s function, with additional expert branches initially contributing
nothing.

The full 9B conversion has not yet been executed, so exact full-checkpoint logit
equivalence remains an acceptance test rather than completed evidence.

## 5. How routed experts are initialized

Each routed expert is smaller than the shared expert:

```text
hidden size:                1,536
routed intermediate size:  2,048
shared intermediate size:  8,960
experts per layer:         28
```

The fused expert gate/up tensor has shape:

```text
(28 experts, 4096 gate+up rows, 1536 hidden dimensions)
```

For each expert, the converter chooses 2,048 rows from the original dense gate
projection and 2,048 corresponding rows from the dense up projection:

```python
offset = (layer * 997 + expert * 2048) % 8960
expert_gate = dense_gate[offset : offset + 2048]
expert_up   = dense_up[offset : offset + 2048]
```

Wrapping is used when the slice crosses the end of the matrix. Small
deterministic noise is added so experts do not begin completely identical.

The expert’s down projection has shape:

```text
(1536, 2048)
```

and starts at zero.

Thus every expert initially has pretrained feature detectors in its gate/up
side, but cannot affect the model until its down projection learns a useful
residual.

## 6. How internal MoE routing works

Routing happens independently:

- for every token;
- in every transformer layer;
- during both training and inference.

For a batch containing `T = batch × sequence length` tokens, the layer’s hidden
states have shape:

```text
(T, 1536)
```

Each layer has a router matrix:

```text
W_router: (28, 1536)
```

The router calculates:

\[
r = h W_{\text{router}}^T
\]

producing 28 logits per token:

```text
router logits: (T, 28)
```

It then applies softmax:

\[
p_i = \frac{e^{r_i}}{\sum_j e^{r_j}}
\]

and selects the two largest probabilities:

```python
probabilities = softmax(router_logits)
weights, expert_ids = topk(probabilities, k=2)
weights = weights / weights.sum()
```

For example, one token might produce:

```text
expert 7:  0.42
expert 19: 0.26
expert 3:  0.08
...
```

After selecting and renormalizing the top two:

```text
expert 7:  0.618
expert 19: 0.382
```

The routed output becomes:

\[
0.618E_7(h) + 0.382E_{19}(h)
\]

The next token can select completely different experts. Layer 12 can also
route a token differently from layer 11.

Experts are not manually labeled as “cards,” “loans,” or “transfers.” Any
specialization must emerge from training.

## 7. What is trained

The current policy freezes the original language model and most new expert
parameters.

Trainable:

- 28 router matrices per layer;
- each routed expert’s down projection.

Frozen:

- embeddings and output head;
- attention layers;
- layer normalization;
- shared expert;
- routed expert gate/up projections.

Parameter totals:

```text
Entire model:       8,943,713,792
Active per token:  ~2,073,443,840
Trainable:          2,467,454,976
```

Although 28 experts exist per layer, only two routed experts plus the shared
expert execute for each token.

## 8. How the experts begin learning

At the first step, routed outputs are zero because their down matrices are
zero. However, the down matrices still receive gradients:

\[
\frac{\partial L}{\partial W_{\text{down}}}
\neq 0
\]

Their gate/up features are already nonzero, so the optimizer can learn how to
project those fixed features back into the model’s hidden representation.

As routed down matrices become nonzero:

- selected experts begin affecting token predictions;
- language-model loss begins training the router toward useful selections;
- different experts can specialize around different patterns.

The router also receives a load-balancing auxiliary loss with coefficient
`0.01`. This discourages routing every token to the same one or two experts.

## 9. Preventing expert collapse

After the first 250 steps, every layer must pass these gates:

- every expert receives at least 0.5% of assignments;
- no expert receives more than 20%;
- normalized routing entropy is at least 0.75;
- auxiliary loss is finite;
- routed down projections receive nonzero gradients.

Training stops if the router collapses.

## 10. Two different kinds of routing

The project has two unrelated routers.

### External domain router

This runs before language-model inference:

```text
user query
   │
   ├── banking → run language model
   └── OOD     → exact canned response, no generation
```

It enforces the financial-services boundary and exact stock response. The
current implementation is a deterministic integration fixture, not yet the
trained production router.

### Internal MoE router

This operates inside every transformer layer:

```text
token hidden state → select 2 of 28 experts
```

It does not decide whether a question is banking-related and cannot guarantee
the canned OOD response.

The external router handles application policy; the internal router distributes
neural computation.

## Important limitation

Upcycling does not transform 1.5B worth of learned knowledge into 9B worth of
learned knowledge instantly.

At initialization, this is effectively:

```text
pretrained 1.5B language capability
+ approximately 7.4B parameters of mostly inactive/untrained expert capacity
```

The banking corpus would teach the residual experts domain behavior, but five
million tokens are still limited. The additional experts can specialize; they
cannot acquire the breadth of a genuinely pretrained 9B model from that corpus
alone.
