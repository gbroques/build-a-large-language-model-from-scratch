import re
import pprint

print("Reading the-verdict.txt")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of characters:", len(raw_text))
print(raw_text[:99])
print()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Total number of tokens:", len(preprocessed))
print(preprocessed[:10])
print()

all_tokens = sorted(set(preprocessed))
vocab_size = len(all_tokens)
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print("Total vocabulary size:", vocab_size)
pprint.pprint(list(enumerate(vocab))[:12])
print()


class SimpleTokenizerV1:
    """A simple tokenizer that converts text to token IDs and back."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Splits text into tokens and converts them to token IDs via the vocabulary."""
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Converts token IDs back to text tokens and concatenates them into natural text."""
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Encoding passage from the-verdict.txt:")
print(text)
print(ids)
print()

print("Decoding sample text token IDs:")
print(tokenizer.decode(ids))
print()

text = "Hello, do you like tea?"
print("Encoding passage NOT in the-verdict.txt:")
print(text)
try:
    print(tokenizer.encode(text))
except KeyError as error:
    print("KeyError:", error)
    print("SimpleTokenizerV1 does'nt handle unknown words not in the training set.")
print()

print('Extending vocabulary with <|endoftext|> and <|unk|> tokens.')
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print("Total vocabulary size:", len(vocab.items()))
pprint.pprint(list(vocab.items())[-5:])
print()

class SimpleTokenizerV2:
    """A simple tokenizer that converts text to token IDs and back."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Splits text into tokens and converts them to token IDs via the vocabulary.
        
        Unknown tokens not in the vocabulary are replaced with <|unk|> token."""
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Converts token IDs back to text tokens and concatenates them into natural text."""
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print('Encoding:')
print(text)
tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode(text)

print('Decoding:')
print(ids)
print(tokenizer.decode(ids))
print()
