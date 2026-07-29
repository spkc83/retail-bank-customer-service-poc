# Retail banking model specification

The banking specification is the only active model-development contract in this
repository.

Active release documents:

1. [Banking v3 tool-use SFT plan](banking-v3/01-tool-use-sft-plan.md)
2. [Dual-head domain and intent classifier](banking-v2/05-dual-head-router.md)
3. [POC runtime and operating instructions](../poc/retail-bank-customer-service-poc/README.md)

Banking v3 is the active generative-model contract. It governs the synthetic
tool-use corpus, native Granite architecture, BF16 LoRA adaptation, evaluation,
and ZeroGPU release gates.

The `banking-v2/` model conversion, dense-to-MoE routing, and Bitext corpus
documents are retained as historical control-design records. They do not
describe the active Granite release.

Release documentation:

- [Tool-use SFT dataset card](../data_cards/retail-bank-agent-sft.md)
- [Retail Bank Agent 9B model card](../model_cards/retail-bank-agent-9b.md)
- [Dual-head classifier model card](../model_cards/retail-bank-domain-intent-router.md)
