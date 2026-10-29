import collections

class TokenNgramSpeculator:
    def __init__(self, prompt_tokens):
        self.unigram_map = collections.defaultdict(lambda: collections.defaultdict(int))
        self.bigram_map = collections.defaultdict(lambda: collections.defaultdict(int))
        self.history = list(prompt_tokens)
        self.seed(prompt_tokens)

    def seed(self, tokens):
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            self.unigram_map[t1][t2] += 1
        for i in range(len(tokens) - 2):
            t1, t2, t3 = tokens[i], tokens[i+1], tokens[i+2]
            self.bigram_map[(t1, t2)][t3] += 1

    def observe(self, tok: int):
        self.history.append(tok)
        if len(self.history) >= 2:
            self.unigram_map[self.history[-2]][self.history[-1]] += 1
        if len(self.history) >= 3:
            self.bigram_map[(self.history[-3], self.history[-2])][self.history[-1]] += 1

    def propose(self, root_tok: int, max_draft: int = 2):
        proposals = []
        curr = root_tok
        # Check bigram first if we have at least 1 previous token
        if len(self.history) >= 1:
            prev = self.history[-1]
            k = (prev, curr)
            if k in self.bigram_map and self.bigram_map[k]:
                best_t, cnt = max(self.bigram_map[k].items(), key=lambda x: x[1])
                total = sum(self.bigram_map[k].values())
                if cnt / total >= 0.35:
                    proposals.append(best_t)
                    # Check next step
                    k2 = (curr, best_t)
                    if k2 in self.bigram_map and self.bigram_map[k2]:
                        best_t2, cnt2 = max(self.bigram_map[k2].items(), key=lambda x: x[1])
                        total2 = sum(self.bigram_map[k2].values())
                        if cnt2 / total2 >= 0.35:
                            proposals.append(best_t2)
                    return proposals

        # Unigram fallback
        if curr in self.unigram_map and self.unigram_map[curr]:
            best_t, cnt = max(self.unigram_map[curr].items(), key=lambda x: x[1])
            total = sum(self.unigram_map[curr].values())
            if cnt / total >= 0.35:
                proposals.append(best_t)
                if best_t in self.unigram_map and self.unigram_map[best_t]:
                    best_t2, cnt2 = max(self.unigram_map[best_t].items(), key=lambda x: x[1])
                    total2 = sum(self.unigram_map[best_t].values())
                    if cnt2 / total2 >= 0.35:
                        proposals.append(best_t2)
        return proposals

print("TokenNgramSpeculator compiled successfully!")
