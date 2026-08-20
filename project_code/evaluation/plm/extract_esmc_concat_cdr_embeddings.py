#!/usr/bin/env python3
"""Extract ESM-C 300M penultimate-layer residue embeddings for concat_CDR sequences."""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_METADATA = "/opig-shared/users/lina4783/structures_final/metadata.csv"
DEFAULT_OUTDIR = (
    "/opig-shared/users/lina4783/abb4_experiments/plm/embeddings/esmc_300m_concat_cdr"
)
DEFAULT_MANIFEST = (
    "/opig-shared/users/lina4783/abb4_experiments/plm/esmc_300m_concat_cdr.csv"
)
DEFAULT_MODEL = "biohub/ESMC-300M"
DEFAULT_EMBED_DIM = 960


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ESM-C penultimate-layer residue embeddings for concat_CDR."
    )
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--seq-col", default="concat_CDR")
    parser.add_argument("--id-col", default="pdb_name")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip sequences whose output .pt already exists.",
    )
    return parser.parse_args()


def sanitize_pdb_name(pdb_name: str) -> str:
    return pdb_name.replace(":", "_")


def embedding_path(outdir: Path, pdb_name: str) -> Path:
    return outdir / f"{sanitize_pdb_name(pdb_name)}.pt"


def make_record(
    pdb_name: str,
    sequence: str,
    out_path: Path,
    embed_dim: int,
    seq_col: str,
) -> dict[str, object]:
    return {
        "pdb_name": pdb_name,
        seq_col: sequence,
        "embedding_path": str(out_path),
        "seq_len": len(sequence),
        "embed_dim": embed_dim,
    }


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        import torch as torch_mod

        raise RuntimeError(
            "CUDA is not available. Common causes on this cluster:\n"
            f"  - PyTorch build: {torch_mod.__version__} (CUDA {torch_mod.version.cuda})\n"
            "  - Installed PyTorch must match the node driver (use cu124, not cu130)\n"
            "  - Job must run on a GPU node with --gres=gpu:1\n"
            "Check nvidia-smi output at the top of the job log."
        )
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print(f"CUDA driver capability: {torch.cuda.get_device_capability(device)}", flush=True)
    return device


def get_penultimate_layer(model: AutoModelForMaskedLM) -> torch.nn.Module:
    blocks = model.esmc.transformer.blocks
    if len(blocks) < 2:
        raise ValueError("Expected at least two transformer blocks.")
    return blocks[-2]


def extract_residue_embeddings(
    penultimate_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    sequences: list[str],
) -> list[torch.Tensor]:
    embeddings: list[torch.Tensor] = []

    for i, sequence in enumerate(sequences):
        valid_len = int(attention_mask[i].sum().item())
        residue_emb = penultimate_hidden[i, 1 : valid_len - 1, :].detach().cpu().float().clone()
        if residue_emb.shape[0] != len(sequence):
            raise ValueError(
                f"Embedding length mismatch for sequence index {i}: "
                f"expected {len(sequence)}, got {residue_emb.shape[0]}"
            )
        embeddings.append(residue_emb)

    return embeddings


def run_batch(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    batch_seqs: list[str],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    penultimate_hidden: dict[str, torch.Tensor] = {}
    penultimate_layer = get_penultimate_layer(model)

    def capture_penultimate(_module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        penultimate_hidden["states"] = hidden

    handle = penultimate_layer.register_forward_hook(capture_penultimate)
    try:
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                compute_sae=False,
            )

        if "states" not in penultimate_hidden:
            raise RuntimeError("Failed to capture penultimate-layer hidden states.")

        batch_embeddings = extract_residue_embeddings(
            penultimate_hidden["states"],
            inputs["attention_mask"],
            batch_seqs,
        )
        return batch_embeddings, inputs["attention_mask"]
    finally:
        handle.remove()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    print(args.seq_col)
    if args.seq_col not in df.columns:
        raise KeyError(f"Column {args.seq_col!r} not found in {args.metadata}")
    if args.id_col not in df.columns:
        raise KeyError(f"Column {args.id_col!r} not found in {args.metadata}")

    pending_ids: list[str] = []
    pending_seqs: list[str] = []
    skipped = 0
    saved = 0
    embed_dim = DEFAULT_EMBED_DIM

    for _, row in df.iterrows():
        pdb_name = str(row[args.id_col])
        sequence = str(row[args.seq_col])
        out_path = embedding_path(outdir, pdb_name)

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        pending_ids.append(pdb_name)
        pending_seqs.append(sequence)

    print(f"Loaded {len(df)} sequences from {args.metadata}", flush=True)
    print(f"Skipping {skipped} existing embeddings", flush=True)
    print(f"Processing {len(pending_seqs)} sequences", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)

    device = require_cuda()

    if pending_seqs:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        embed_dim = int(getattr(model.config, "d_model", DEFAULT_EMBED_DIM))

        num_batches = (len(pending_seqs) + args.batch_size - 1) // args.batch_size
        for batch_idx, start in enumerate(
            tqdm(
                range(0, len(pending_seqs), args.batch_size),
                desc="Extracting embeddings",
                file=sys.stdout,
            ),
            start=1,
        ):
            batch_ids = pending_ids[start : start + args.batch_size]
            batch_seqs = pending_seqs[start : start + args.batch_size]

            batch_embeddings, _ = run_batch(model, tokenizer, batch_seqs, device)

            for pdb_name, embedding in zip(batch_ids, batch_embeddings, strict=True):
                out_path = embedding_path(outdir, pdb_name)
                torch.save(embedding, out_path)
                saved += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(
                f"Completed batch {batch_idx}/{num_batches}; "
                f"saved {saved} new embeddings so far",
                flush=True,
            )

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        pdb_name = str(row[args.id_col])
        sequence = str(row[args.seq_col])
        out_path = embedding_path(outdir, pdb_name)

        if not out_path.exists():
            raise FileNotFoundError(f"Missing embedding file for {pdb_name}: {out_path}")

        records.append(make_record(pdb_name, sequence, out_path, embed_dim, args.seq_col))

    manifest_df = pd.DataFrame(records)
    if len(manifest_df) != len(df):
        raise RuntimeError(
            f"Manifest row count ({len(manifest_df)}) does not match metadata ({len(df)})"
        )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)

    print(f"Saved {saved} new embeddings to {outdir}", flush=True)
    print(f"Skipped {skipped} existing embeddings", flush=True)
    print(f"Wrote manifest with {len(manifest_df)} rows to {manifest_path}", flush=True)
    print(f"Embedding dimension: {embed_dim}", flush=True)


if __name__ == "__main__":
    main()
