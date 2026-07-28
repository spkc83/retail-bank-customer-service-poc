# Retail banking model specification

The banking specification is the only active model-development contract in this
repository.

Read in order:

1. [Corpus, governance, and acceptance gates](banking-v2/README.md)
2. [Model conversion and training](banking-v2/02-model-training.md)
3. [Evaluation and serving](banking-v2/03-evaluation-serving.md)
4. [Dense-to-MoE initialization and routing](banking-v2/04-dense-to-moe-routing.md)
5. [Dual-head domain and intent router](banking-v2/05-dual-head-router.md)

The released model card is
[Retail Bank Servicing MoE 9B](../model_cards/retail-bank-servicing-moe-9b.md).
The deployable application contract and operating instructions are in the
[POC README](../poc/retail-bank-customer-service-poc/README.md).
