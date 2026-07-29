# Retail banking model specification

The banking specification is the only active model-development contract in this
repository.

Read in order:

1. [Banking v3 tool-use SFT plan](banking-v3/01-tool-use-sft-plan.md)
2. [Corpus, governance, and acceptance gates](banking-v2/README.md)
3. [Model conversion and training](banking-v2/02-model-training.md)
4. [Evaluation and serving](banking-v2/03-evaluation-serving.md)
5. [Dense-to-MoE initialization and routing](banking-v2/04-dense-to-moe-routing.md)
6. [Dual-head domain and intent router](banking-v2/05-dual-head-router.md)

Banking v2 remains the released implementation contract. Banking v3 is the
reviewed next-iteration proposal and becomes the release contract only after
its data, model bakeoff, training, evaluation, and ZeroGPU gates pass.

The released model card is
[Retail Bank Servicing MoE 9B](../model_cards/retail-bank-servicing-moe-9b.md).
The deployable application contract and operating instructions are in the
[POC README](../poc/retail-bank-customer-service-poc/README.md).
