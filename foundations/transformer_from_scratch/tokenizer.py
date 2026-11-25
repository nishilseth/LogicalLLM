"""GPT-2 byte-level BPE tokenizer, written from scratch.

Only `json` and `regex` (for the \\p{L} unicode classes in GPT-2's pre-tokenisation
pattern) are used; no tokenizer library.
"""
import json
import regex as re

PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def bytes_to_unicode():
    """Map each of the 256 byte values to a printable unicode char so BPE can
    work on text without ever hitting whitespace/control characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class BPETokenizer:
    def __init__(self, vocab_file, merges_file):
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as f:
            merges = [tuple(l.split()) for l in f.read().split("\n")[1:] if l]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.byte_enc = bytes_to_unicode()
        self.byte_dec = {v: k for k, v in self.byte_enc.items()}
        self.cache = {}

    def _bpe(self, token):
        """Repeatedly merge the adjacent pair with the lowest merge rank."""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        self.cache[token] = word
        return word

    def encode(self, text):
        ids = []
        for chunk in PAT.findall(text):
            mapped = "".join(self.byte_enc[b] for b in chunk.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self._bpe(mapped))
        return ids

    def decode(self, ids):
        text = "".join(self.decoder[i] for i in ids)
        return bytearray(self.byte_dec[c] for c in text).decode("utf-8", errors="replace")
