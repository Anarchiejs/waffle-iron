#!/usr/bin/env python3
"""Génère tests/data/tiny-llama.gguf : llama minuscule, poids aléatoires, chargeable par llama.cpp b11907.

Architecture : n_embd 64, 2 couches, 4 têtes (2 KV), n_ff 128, contexte 256.
Vocabulaire SPM (« llama ») : <unk>, <s>, </s>, 256 octets, puis des morceaux fr/en.
Les tenseurs 1D restent en f32, les matrices en f16 (general.file_type = MOSTLY_F16).
Les formes NumPy sont l'inverse des dimensions ggml : ne[0] est le dernier axe NumPy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "llama.cpp" / "gguf-py"))

try:
    import numpy as np
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "numpy"])
    import numpy as np

import gguf

N_EMBD = 64
N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 2
N_FF = 128
N_CTX = 256
HEAD_DIM = N_EMBD // N_HEAD
SEED = 11907
LIMIT = 10 * 1024 * 1024

# Mots courts fr/en. Le préfixe ▁ (U+2581) marque le début de mot, comme SentencePiece.
WORDS = """
le la les un une des et ou mais donc car je tu il elle on nous vous ils elles
ce cet cette ces mon ma mes ton ta tes son sa ses notre votre leur
pas plus moins très tout tous toute toutes dans sur sous avec sans pour par
est suis es sont était être avoir faire aller dire voir pouvoir vouloir
bonjour merci oui non bon bien mal peu ici là quand où qui que quoi
the a an of to and in on for with from by as at it is be
are was were this that these those you we they not or but if then
hello thanks yes no good bad small large time day year man woman child
model token layer head cache memory bench local gpu cpu file type name
chat code data test train value number word text line file path context
prompt speed power temperature sample result campaign history export plan scan
français anglais modèle couche tête mémoire contexte fichier mesure résultat
""".split()


def vocab() -> tuple[list[str], list[float], list[int]]:
    tokens = ["<unk>", "<s>", "</s>"]
    scores = [0.0, 0.0, 0.0]
    types = [int(gguf.TokenType.UNKNOWN), int(gguf.TokenType.CONTROL), int(gguf.TokenType.CONTROL)]
    for byte in range(256):
        tokens.append(f"<0x{byte:02X}>")
        scores.append(-1000.0)
        types.append(int(gguf.TokenType.BYTE))
    seen = set(tokens)
    pieces: list[str] = []
    for word in WORDS:
        pieces.append("▁" + word)
        pieces.append(word)
    for word in pieces:
        if word in seen or not word:
            continue
        seen.add(word)
        tokens.append(word)
        # Score un peu plus haut pour la forme ▁mot, afin qu'elle gagne au découpage.
        scores.append(-0.2 if word.startswith("▁") else -1.0 - 0.01 * len(tokens))
        types.append(int(gguf.TokenType.NORMAL))
    return tokens, scores, types


def matrix(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    # rows = dimension ggml ne[1], cols = ne[0].
    data = rng.normal(0.0, 0.02, size=(rows, cols)).astype(np.float16)
    return np.ascontiguousarray(data)


def vector(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.ascontiguousarray(rng.normal(1.0, 0.01, size=(n,)).astype(np.float32))


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "data" / "tiny-llama.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens, scores, types = vocab()
    n_vocab = len(tokens)
    rng = np.random.default_rng(SEED)
    n_q = N_HEAD * HEAD_DIM
    n_kv = N_HEAD_KV * HEAD_DIM

    writer = gguf.GGUFWriter(out, "llama")
    writer.add_name("tiny-llama")
    writer.add_description("Modèle llama minuscule à poids aléatoires, réservé aux tests de Banc Local.")
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16))
    writer.add_vocab_size(n_vocab)
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_key_length(HEAD_DIM)
    writer.add_value_length(HEAD_DIM)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)

    writer.add_tensor("token_embd.weight", matrix(rng, n_vocab, N_EMBD))
    for layer in range(N_LAYER):
        prefix = f"blk.{layer}"
        writer.add_tensor(f"{prefix}.attn_norm.weight", vector(rng, N_EMBD))
        writer.add_tensor(f"{prefix}.attn_q.weight", matrix(rng, n_q, N_EMBD))
        writer.add_tensor(f"{prefix}.attn_k.weight", matrix(rng, n_kv, N_EMBD))
        writer.add_tensor(f"{prefix}.attn_v.weight", matrix(rng, n_kv, N_EMBD))
        writer.add_tensor(f"{prefix}.attn_output.weight", matrix(rng, N_EMBD, n_q))
        writer.add_tensor(f"{prefix}.ffn_norm.weight", vector(rng, N_EMBD))
        writer.add_tensor(f"{prefix}.ffn_gate.weight", matrix(rng, N_FF, N_EMBD))
        writer.add_tensor(f"{prefix}.ffn_down.weight", matrix(rng, N_EMBD, N_FF))
        writer.add_tensor(f"{prefix}.ffn_up.weight", matrix(rng, N_FF, N_EMBD))
    writer.add_tensor("output_norm.weight", vector(rng, N_EMBD))
    writer.add_tensor("output.weight", matrix(rng, n_vocab, N_EMBD))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = out.stat().st_size
    if size >= LIMIT:
        raise SystemExit(f"{out} fait {size} octets, au-dessus de 10 Mio")
    print(f"{out}  {n_vocab} jetons  {size} octets")


if __name__ == "__main__":
    main()
