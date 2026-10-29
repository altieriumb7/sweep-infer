import re
from typing import List, Tuple

class SimpleSpeculativeLookahead:
    def __init__(self, prompt_text: str):
        self.prompt_text = prompt_text
        self.tokens_stream = []
        self.order1_map = {}  # w1 -> {w2: count}
        self.order2_map = {}  # (w1, w2) -> {w3: count}
        self.seed(prompt_text)

    def seed(self, text: str):
        words = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_'\u00C0-\u017F]+\b", text)]
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1 not in self.order1_map:
                self.order1_map[w1] = {}
            self.order1_map[w1][w2] = self.order1_map[w1].get(w2, 0) + 1

        for i in range(len(words) - 2):
            k = (words[i], words[i+1])
            w3 = words[i+2]
            if k not in self.order2_map:
                self.order2_map[k] = {}
            self.order2_map[k][w3] = self.order2_map[k].get(w3, 0) + 1

    def absorb(self, word: str):
        w = word.strip().lower()
        if not w:
            return
        self.tokens_stream.append(w)
        if len(self.tokens_stream) >= 2:
            w1, w2 = self.tokens_stream[-2], self.tokens_stream[-1]
            if w1 not in self.order1_map:
                self.order1_map[w1] = {}
            self.order1_map[w1][w2] = self.order1_map[w1].get(w2, 0) + 1
        if len(self.tokens_stream) >= 3:
            k = (self.tokens_stream[-3], self.tokens_stream[-2])
            w3 = self.tokens_stream[-1]
            if k not in self.order2_map:
                self.order2_map[k] = {}
            self.order2_map[k][w3] = self.order2_map[k].get(w3, 0) + 1

    def propose(self, current_word: str) -> List[str]:
        cw = current_word.strip().lower()
        candidates = []
        # Try order-2 first
        if len(self.tokens_stream) >= 1:
            prev = self.tokens_stream[-1]
            k = (prev, cw)
            if k in self.order2_map and self.order2_map[k]:
                best_w = max(self.order2_map[k].items(), key=lambda x: x[1])[0]
                candidates.append(best_w)
                # Try next step
                k2 = (cw, best_w)
                if k2 in self.order2_map and self.order2_map[k2]:
                    best_w2 = max(self.order2_map[k2].items(), key=lambda x: x[1])[0]
                    candidates.append(best_w2)
                return candidates

        # Try order-1
        if cw in self.order1_map and self.order1_map[cw]:
            best_w = max(self.order1_map[cw].items(), key=lambda x: x[1])[0]
            candidates.append(best_w)
            if best_w in self.order1_map and self.order1_map[best_w]:
                best_w2 = max(self.order1_map[best_w].items(), key=lambda x: x[1])[0]
                candidates.append(best_w2)
        return candidates

# Test on Logic prompt
look = SimpleSpeculativeLookahead(
    "Davanti a te ci sono tre scrigni: d'Oro, d'Argento e di Piombo. "
    "Scrigno d'Oro: 'Il tesoro si trova qui dentro.' "
    "Scrigno d'Argento: 'Il tesoro non si trova qui dentro.' "
    "Scrigno di Piombo: 'Il tesoro non si trova nello scrigno d'Oro.'"
)

print("Proposals for 'tesoro':", look.propose("tesoro"))
print("Proposals for 'scrigno':", look.propose("scrigno"))
print("Proposals for 'non':", look.propose("non"))
print("Proposals for 'si':", look.propose("si"))
