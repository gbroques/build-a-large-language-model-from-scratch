# Working with text data

* **Tokenization** - splitting text into individual word and subword tokens.
* **Byte pair encoding** - advanced tokenization algorithm.
* **Embedding** - the concept of converting data into a vector format.
  * Can embed different data types (e.g. video, audio, text).
  * Different data types require different embedding models (e.g. a text embedding model isn't suitable for audio).
* Word2Vec
  * Words in similar context tend to have similar meaning.
  * Neural network architecture.
* Higher dimensionality might capture more nuanced meaning at the cost of computational efficiency.
* LLMs commonly produce their own embeddings as part of the input layer, and are updated during training.
  * The advantage of this approach, as opposed to using something like Word2Vec, is that the embeddings are optimized to the task and data at hand.

![two dimensional word embeddings plot](./2-3-two-dimensional-word-embeddings-plot.png)

Data preparation steps:
1. Spltting text into words
  * Spltting text into words could be a simple regular expression to split on whitespace and punctuation characters.
  * You may want to filter whitespace tokens depending upon the task.
  * For Python code, whitespace matters.
2. Converting words into tokens
3. Converting tokens into embeddings

After tokenizing the text, a vocabulary is constructed by:
1. Sorting the tokens alphabetically.
2. Removing duplicates.

The vocabulary is then used to map tokens from the input text to **token IDs**.

![token vocabulary](./2-6-token-vocabulary.png)

Depending on the LLM, some researchers also consider additional special tokens such as the following:

* [BOS] (beginning of sequence) — This token marks the start of a text. It signifies to the LLM where a piece of content begins.
* [EOS] (end of sequence) — This token is positioned at the end of a text and is especially useful when concatenating multiple unrelated texts, similar to <|endoftext|>. For instance, when combining two different Wikipedia articles or books, the [EOS] token indicates where one ends and the next begins.
* [PAD] (padding) — When training LLMs with batch sizes larger than one, the batch might contain texts of varying lengths. To ensure all texts have the same length, the shorter texts are extended or "padded" using the [PAD] token, up to the length of the longest text in the batch.

The tokenizer used for GPT models does not need any of these tokens; it only uses an <|endoftext|> token for simplicity. <|endoftext|> is analogous to the [EOS] token. <|endoftext|> is also used for padding. However, as we’ll explore in subsequent chapters, when training on batched inputs, we typically use a mask, meaning we don’t attend to padded tokens. Thus, the specific token chosen for padding becomes inconsequential.

Moreover, the tokenizer used for GPT models also doesn’t use an <|unk|> token for out-of-vocabulary words. Instead, GPT models use a byte pair encoding tokenizer, which breaks words down into subword units, which we will discuss next.

## Byte pair encoding

Byte pair encoding (BPE) is a more sophisticated tokenization algorithm used by GPT-2, GPT-3 and the original model used in ChatGPT.

It has a couple desirable properties:

1. It's reversible and lossless, so you can convert tokens back into the original text
2. It works on arbitrary text, even text that is not in the tokeniser's training data
3. It compresses the text: the token sequence is shorter than the bytes corresponding to the original text. On average, in practice, each token corresponds to about 4 bytes.
4. It attempts to let the model see common subwords. For instance, "ing" is a common subword in English, so BPE encodings will often split "encoding" into tokens like "encod" and "ing" (instead of e.g. "enc" and "oding"). Because the model will then see the "ing" token again and again in different contexts, it helps models generalise and better understand grammar.

[openai/tiktoken](https://github.com/openai/tiktoken) implements BPE efficiently in Rust.

Run [`./byte-pair-encoding.py`](./byte-pair-encoding.py).

Two notable obervations include:
1. The `<|endoftext|>` token is assigned `50256` as its token ID.
  * The BPE tokenizer used to train models such as GPT-2, GPT-3, and the original model used in ChatGPT, has a total vocabulary size of 50,257, with `<|endoftext|>` being assigned the largest token ID.
2. The BPE tokenizer encodes and decodes unknown words, such as **someunknownPlace**, correctly without using `<|unk|>` tokens.
  * The algorithm breaks down words that aren't in its predefined vocabulary into smaller subword units or even individual characters, enabling it to handle out-of-vocabulary words.

![byte pair encoding breaking down text](./2-11-byte-pair-encoding-breaking-down-text.png)

BPE builds its vocabulary by iteratively merging frequent characters into subwords, and frequent subwords into words. For example, BPE starts with adding all individual single characters to its vocabulary ("a," "b," etc.). Next, it merges character combinations that frequently occur together into subwords. For example, "d" and "e" may be merged into the subword "de," which is common in many English words like "define," "depend," and "decide,". The merges are determined by a frequency cutoff.

`stride` controls the number of positions the inputs shift across batches.

![stride parameter](./2-14-stride-parameter.png)

