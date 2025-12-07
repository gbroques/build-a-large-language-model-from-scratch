# Build a Large Language Model (From Scratch)

Source: [sebastianraschka.com/books](https://sebastianraschka.com/books/)

## Resources

- [Free early access PDF](https://book.caibitim.duckdns.org/download/683/pdf/683.pdf)
- [YouTube playlist](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu) of a guy who went through it and based videos on it (each ~30 min to an hour)
- [Official video course](https://www.manning.com/livevideo/master-and-build-large-language-models)
- [Free official supplementary YouTube playlist](https://www.youtube.com/playlist?list=PLQRyiBCWmqp5twpd8Izmaxu5XRkxd5yC-)

## Introduction & Fundamentals

Video: [https://www.youtube.com/watch?v=3dWzNZXA8DY&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=2](https://www.youtube.com/watch?v=3dWzNZXA8DY&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=2)

A large language model is a neural network designed to understand, generate, and respond to human-like text.

### AI Hierarchy

Artificial Intelligence -> Machine Learning -> Deep Learning -> Large Language Models

Generative AI includes Deep Learning and Large Language Models since it includes other modalities such as images and audio.

### Why "Large"?

Large language models have billions of parameters:

- GPT-1 - 512 token size, 12 decoders, 117M parameters
- GPT-2 - 1024 token size, 48 decoders, 1.5B parameters
- GPT-3 - 2048 token size, 96 decoders, 175B parameters

GPT-2 is roughly 10x the parameters as GPT-1, and GPT-3 is roughly 100x the parameters of GPT-2.

![Nature The Drive to Bigger AI Models](https://media.nature.com/lw767/magazine-assets/d41586-023-00777-9/d41586-023-00777-9_24620916.jpg)

The chart shows exponential increase in parameters from 1950 to 2020 (log scale).

### Why "Language" Models?

They do a large variety of natural language processing (NLP) tasks such as question-answering, translation, sentiment analysis, and more.

LLMs are general-purpose. Before LLMs, earlier language models were designed for specific tasks.

### LLM Applications

- Content creation
- Chatbots / virtual assistants
- Machine translation
- Sentiment analysis

## Transformer Architecture

Video: [https://www.youtube.com/watch?v=NLn4eetGmf8&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=4](https://www.youtube.com/watch?v=NLn4eetGmf8&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=4)

### What are Transformers?

Deep neural network architecture introduced in 2017: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - more than 100k citations in 5 years. 15 pages.

![Transformer architecture](https://arxiv.org/html/1706.03762v7/extracted/1706.03762v7/Figures/ModalNet-21.png)

Originally developed for machine translation: English to German / French

### Simplified Transformer Architecture

The transformer architecture consists of two submodules:

1. **Encoder** - generates embeddings - words with similar meaning are closer to each other in space
   - ![vector embeddings](https://causewriter.ai/wp-content/uploads/2023/08/image-2-1024x483.png)
2. **Decoder** - generates translated text one word at a time using an embedding for the input text + partial output text

![figure 1.4 - simplified transformer architecture](./1-4-simplified-transformer-architecture.png)

### Self-Attention Mechanism

- Allows the model to weigh the importance of different words or tokens in a sequence relative to each other.
- Enables the model to capture long-range dependencies and contextual relationships within the input data.
- Enhances the model's ability to generate coherent and contextually relevant output.

Explained in greater detail during chapter 3.

### Encoder vs Decoder: BERT vs GPT

**Generative Pre-trained Transformers (GPT)** - generates a new word
- Has a **decoder** and no encoder
- Example: "This is an example of how LLM can __"
- Does better in machine translation, text summarization, fiction writing, writing computer code, and more.

**Bidirectional Encoder Representations from Transformers (BERT)** - predicts hidden words in a sentence (masking or masked word prediction)
- Has an **encoder** and no decoder
- Tokens are randomly masked in training
- Example: "This is an __ of how LLM __ perform"
- Does better in text classification tasks such as sentiment analysis and document categorization.
- X (formerly Twitter) uses BERT to detect "toxic" content.
- Bidirectional looks at context on the left and right
- Understands the nuance of "bank" in a financial context vs. a river bank

![figure 1.5 - BERT vs. GPT](./1-5-bert-vs-gpt.png)

### Evolution of Neural Network Architectures

- 1980 - Recurrent neural networks (RNN) have feedback loops
- 1997 - Long short-term memory networks (LSTM) - 2 paths: one for short-term memories and one for long-term memories
- 2017 - Transformer

### Transformers vs LLMs

Not all transformers are LLMs:
- Transformers can be used for other tasks like computer vision (Vision Transformers - ViT)
- ViT achieves remarkable results compared to CNNs with less computational resources during pre-training

Not all LLMs are transformers:
- LLMs can be based on recurrent or convolutional architectures as well

Don't use transformers and LLMs interchangeably.

## Training Methodology

Video: [https://www.youtube.com/watch?v=-bsa3fCNGg4&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=3](https://www.youtube.com/watch?v=-bsa3fCNGg4&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=3)

Creating an LLM consists of two stages:

### Pre-training

Training on a large and diverse dataset.

- Unsupervised or unlabeled dataset
- GPT-3 was trained on:
  - [Common Crawl](https://commoncrawl.org/) - 60% web crawl data (250 billion pages spanning 17 years)
  - [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest/) / WebText2 - 22% (Reddit submissions from 2005 - 2020)
  - Books - 16%
  - Wikipedia - 3% high-quality data
- Hundreds of billions of tokens
- The authors of the GPT-3 paper did not share the training dataset, but a comparable dataset that is publicly available is Dolma: An Open Corpus of Three Trillion Tokens for LLM Pretraining Research by Soldaini et al. 2024 (https://arxiv.org/abs/2402.00159).
- Total pre-training cost for GPT-3 was $4.6 million:
  - [Reddit discussion](https://www.reddit.com/r/MachineLearning/comments/hwfjej/d_the_cost_of_training_gpt3/)
  - [Lambda AI blog](https://lambda.ai/blog/demystifying-gpt-3)
  - [VentureBeat article](https://venturebeat.com/ai/openai-launches-an-api-to-commercialize-its-research/)
- Source: [https://openai.com/index/language-unsupervised/](https://openai.com/index/language-unsupervised/)
- "We also noticed we can use the underlying language model to begin to perform tasks without ever training on them."

Pre-trained models are base / foundational models which can be used for finetuning.

Many pre-training LLMs are available as open-weight models which can be used to write, extract, and edit text not included in the training data.

Source: [https://www.researchgate.net/figure/Performance-comparison-of-closed-source-and-open-weight-large-language-models-on-the_fig2_386193568](https://www.researchgate.net/figure/Performance-comparison-of-closed-source-and-open-weight-large-language-models-on-the_fig2_386193568)

### Fine-tuning

Training on a narrower dataset, specific to a task or domain.

- Supervised or labeled dataset
- Uses a pre-trained or foundational model
- [OpenAI model optimization guide](https://platform.openai.com/docs/guides/model-optimization)
- [OpenAI fine-tuning improvements](https://openai.com/index/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program/)

Industry examples:
- Airline industry - SK Telecom
- Healthcare
- Law - harvey.ai
- JPMorgan Chase LLM suite

Two types of fine-tuning:
- **Instruction fine-tuning** - prompt-response pairs
- **Fine-tuning for classification tasks** - text & labels (e.g. 10k emails labelled spam and not spam)

### Zero-Shot, One-Shot, and Few-Shot Learning

- **Zero shot** - generalize to completely unseen tasks without any prior examples
- **One shot** - learns from one example
- **Few shot** - learning from a minimum number of examples from the user

Examples from "Language Models are Few-Shot Learners" paper:
- GPT-3 is a few-shot learner
- GPT-4 is a few-shot learner, and capable of zero shot learning but does better with some examples

## GPT Evolution & Research

Video: [https://www.youtube.com/watch?v=xbaYCf2FHSY&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=5](https://www.youtube.com/watch?v=xbaYCf2FHSY&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=5)

### Progression of Research Papers

**2017 - Attention Is All You Need**
- Introduced transformer architecture
- Self-attention mechanism capturing long-range dependencies

**2018 - Improving Language Understanding by Generative Pre-Training**
- GPT-1
- Introduced the concept of "generative pre-training"
- Unsupervised learning
- Labeled data is scarce
- Diverse corpus of unlabeled text
- Blog: [https://openai.com/index/language-unsupervised/](https://openai.com/index/language-unsupervised/)

**2019 - Language Models are Unsupervised Multitask Learners**
- GPT-2
- Increased data
- Blog: [https://openai.com/index/better-language-models/](https://openai.com/index/better-language-models/)

**2020 - Language Models are Few-Shot Learners**
- GPT-3
- Paper: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
- Blog: [https://openai.com/index/language-models-are-few-shot-learners/](https://openai.com/index/language-models-are-few-shot-learners/)

### GPT Architecture

- Only a decoder. No encoder.
- Only trained on next word prediction (e.g. "The lion roams in the ____")
- Auto-regressive model - appending the output to the input and re-running the model until a stop token is generated
- Can do a wide-range of tasks such as spelling correction and translation
- Pre-training is unsupervised
- Weights in the neural network are optimized to make it more likely to predict the next word correctly through a process called backpropagation

### Emergent Behavior

GPT can perform tasks like language translation even though it was only trained on predicting the next word. GPT can perform tasks that it wasn't explicitly trained to perform.

## Stages of Building an LLM from Scratch

Video: [https://www.youtube.com/watch?v=z9fgKz1Drlc&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=6](https://www.youtube.com/watch?v=z9fgKz1Drlc&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=6)

![figure 1.9 - three main stages of coding an LLM](./1-9-three-main-stages-of-coding-an-llm.png)

### Stage 1: Architecture and Fundamentals

**Data Preparation & Sampling**
- **Tokenization** - breaking down text into units
- **Embeddings** - capture semantic meaning of units
- **Batching** - how to handle a large amount of data. What is "context"?
  - [How GPT3 Works - Visualizations and Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

**Attention Mechanism**
- Multi-head attention
- Masked multi-head attention

**LLM Architecture**

### Stage 2: Pretraining

**Build foundational model on unlabeled data:**
- Training loop (see page 171 of book):
- Model evaluation
- Load pre-trained weights

### Stage 3: Finetuning

- Classifier - dataset with class labels
- Personal Assistant - dataset with instructions

### Prompt Style Templates

**Phi-3 prompt style template**

Source: [https://huggingface.co/microsoft/Phi-3-mini-128k-instruct#chat-format](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct#chat-format)

```
<|system|>
You are a helpful assistant.<|end|>
<|user|>
Question?<|end|>
<|assistant|>
```

**Alpaca prompt style template**

