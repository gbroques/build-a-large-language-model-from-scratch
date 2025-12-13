# Coding Attention Mechanisms

We'll code the attention mechanism in 4 stages:

1. Simple self-attention
2. Self-attention with trainable weights
3. Causal attention mechanism 
  * Adds a mask to self-attention that allows the llm to generate one word at a time.
4. Multi-head attention
  * Organizes the attention mechanism into multiple heads, allowing the model to capture various aspects of the input data in parallel.

![4 stages of coding the attention mechanism](./3-2-stages-of-self-attention-mechanism.png)

## Motivating The Attention Mechanism

When translating between languagues, it's not possible to translate word by word due to the grammatical structures within each language.

To solve this problem, it's common to use a deep neural network with two submodules:

1. an encoder - to process the input text
2. a decoder - to translate the input text

Before transformers, recurrent neural networks (RNNs) were the most popular encoder-decoder architecture for translating text.

An RNN is a type of neural network where outputs from previous steps are fed as inputs to the current step making them well-suited for sequential data like text.

The encoder updates its hidden state (the internal values at the hidden layers) at each step, trying to capture the entire meaning of the input sentence in the final hidden state acting as a memory cell. The decoder then takes this final hidden state to start generating the translated sentence, one word at a time. It also updates its hidden state at each step, which is supposed to carry the context necessary for the next-word prediction.

The big limitation of encoder–decoder RNNs is that the RNN can't directly access earlier hidden states from the encoder during the decoding phase. Consequently, it relies solely on the current hidden state, which encapsulates all relevant information. This can lead to a loss of context, especially in complex sentences where dependencies might span long distances.

## Capturing Data Dependencies with Attention Mechanisms

RNNs work fine for translating short sentences, but don't work well for longer texts as they don't have access to previous words in the input, and must remember the entire encoded input in a single hidden state before passing it to the decoder.

To address this, researchers developed the [Bahdanau attention mechanism](https://arxiv.org/abs/1409.0473) for RNNs in 2014 which modifies its architecture such that the decoder can selectively access different parts of the input sequence at each decoding.

The transformer architecture's (2017) self-attention mechanism is inspired by the Bahdanau attention mechanism.

Self-attention is a mechanism that allows each position in the input sequence to consider the relevancy of, or "attend to," all other positions in the same sequence when computing the representation of a sequence.

## Self-attention

Self-attention serves as the cornerstone of every LLM based on the transformer architecture.

The "self" in self-attention refers to the mechanism's ability to compute attention weights by relating different positions within a single input sequence. It learns the relationships and dependencies between various parts of the input itself, such as words in a sentence.

This is in contrast to traditional attention mechanisms, where the focus is on the relationships between elements of two different sequences, such as in sequence-to-sequence models where the attention might be between an input sequence and an output sequence.

### Simplified Version

The goal of self-attention is to compute a *context vector* for each input element that combines information from all other input elements.

A context vector can be interpreted as an enriched embedding vector.

A dot product measures the similarity of two vectors. In the context of self-attention mechanisms, the dot product determines the extent to which each element in a sequence "attends to" any other element: the higher the dot product, the higher the similarity and attention score between them.

Computing a dot product requires summing the element-wise [product](https://en.wikipedia.org/wiki/Product_(mathematics)) for two vectors.

Next, the attention scores are normalized via a `softmax` function to obtain attention weights that sum up to 1 which is useful for interpretation and maintaining training stability in an LLM.

The `softmax` function ensures that the attention weights are always positive, avoids extreme values, and offers more favorable gradient properties during training.

### Self-attention with trainable weights

Next is implementing the self-attention mechanism with trainable weights from the "Attention Is All You Need" paper known as [scaled dot-product attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)#Standard_scaled_dot-product_attention).

We will introduce three trainable weight matrices Wq, Wk, and Wv, to project the embedded input tokens, x(i), into query, key, and value vectors, respectively.

The terms "key," "query," and "value" are borrowed from the domain of information retrieval and databases, where similar concepts are used to store, search, and retrieve information.

A *query* is analogous to a search query in a database. It represents the current item (e.g., a word or token in a sentence) the model focuses on or tries to understand. The query is used to probe the other parts of the input sequence to determine how much attention to pay to them.

The *key* is like a database key used for indexing and searching. In the attention mechanism, each item in the input sequence (e.g., each word in a sentence) has an associated key. These keys are used to match the query.

The *value* in this context is similar to the value in a key-value pair in a database. It represents the actual content or representation of the input items. Once the model determines which keys (and thus which parts of the input) are most relevant to the query (the current focus item), it retrieves the corresponding values.

The attention score computation is a dot-product computation similar to the simplified self-attention mechanism. The new aspect here is that we are not directly computing the dot-product between the input elements, but using the query and key obtained by transforming the inputs via the respective weight matrices.

The reason for the normalization by the embedding dimension size is to improve the training performance by avoiding small gradients near zero. These small gradients can drastically slow down learning or cause training to stagnate.


