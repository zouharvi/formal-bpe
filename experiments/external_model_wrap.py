import sentencepiece as spm
import io

class SentencePiece:
    pass
    def fit_greedy(self, tokens, vocab_size, progress_bar=False):
        tokens = tokens.split("\n")
        while True:
            try:
                model = io.BytesIO()
                spm.SentencePieceTrainer.train(
                    sentence_iterator=(x for x in tokens),
                    model_writer=model,
                    vocab_size=vocab_size,
                    num_threads=1,
                )

                sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
                tokens = [sp.encode(line) for line in tokens]
                return tokens
            except Exception:
                vocab_size -= 500 
