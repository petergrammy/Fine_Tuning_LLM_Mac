import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List


def _iter_txt_files(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.rglob("*.txt") if p.is_file()]
    files.sort()
    return files


def _normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _chunk_by_chars(text: str, chunk_chars: int) -> Iterable[str]:
    if chunk_chars <= 0:
        yield text
        return

    # Prefer splitting on paragraph boundaries, then fallback to hard cut.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    buf: List[str] = []
    buf_len = 0
    for p in paras:
        p_len = len(p) + (2 if buf else 0)
        if buf and (buf_len + p_len) > chunk_chars:
            yield "\n\n".join(buf).strip()
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += p_len

        # If a single paragraph is too long, cut it.
        while buf and len(buf[-1]) > chunk_chars:
            longp = buf.pop()
            head = longp[:chunk_chars].strip()
            tail = longp[chunk_chars:].strip()
            if head:
                yield head
            if tail:
                buf.append(tail)

    if buf:
        yield "\n\n".join(buf).strip()


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _chunks_with_topic(text: str, chunk_chars: int) -> List[str]:
    """
    Heuristic for "topic first, then body":
    - If the first paragraph looks like a short topic/title, prepend it to every chunk.
    - Chunk the remaining body by paragraph boundaries.
    """
    paras = _split_paragraphs(text)
    if not paras:
        return []

    topic = ""
    body_paras = paras
    if len(paras) >= 2:
        p0 = paras[0]
        # Topic is usually short and single-paragraph.
        if 2 <= len(p0) <= 80 and "\n" not in p0:
            topic = p0
            body_paras = paras[1:]

    body = "\n\n".join(body_paras).strip()
    if not body:
        # Only topic exists.
        return [topic] if topic else []

    chunks = list(_chunk_by_chars(body, chunk_chars))
    if not topic:
        return chunks

    out: List[str] = []
    for c in chunks:
        combined = f"{topic}\n\n{c}".strip()
        out.append(combined)
    return out


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _build_dataset_info(dataset_name: str, file_name: str, mode: str) -> dict:
    if mode == "cpt":
        return {dataset_name: {"file_name": file_name, "columns": {"prompt": "text"}}}
    if mode == "sft":
        return {
            dataset_name: {
                "file_name": file_name,
                "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"},
            }
        }
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="包含 .txt 的目录")
    ap.add_argument("--output_dir", type=str, required=True, help="输出数据集目录（包含 dataset_info.json）")
    ap.add_argument("--mode", type=str, choices=["cpt", "sft"], required=True, help="cpt: 继续预训练；sft: 指令微调")
    ap.add_argument("--dataset_name", type=str, required=True, help="写入 dataset_info.json 的数据集名称")
    ap.add_argument("--chunk_chars", type=int, default=1200, help="按字符数切分文本片段（建议 800~2000）")
    ap.add_argument("--min_chars", type=int, default=200, help="丢弃过短片段，避免噪声")
    ap.add_argument("--system", type=str, default="你是一个会用用户口吻写作与对话的助手。", help="SFT 模式下的 system 字段")
    ap.add_argument(
        "--instruction",
        type=str,
        default="请用我的口吻和写作风格，生成一段与下面内容风格一致的回答。",
        help="SFT 模式下的 instruction 字段",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_txt_files(input_dir)
    if not files:
        raise SystemExit(f"在 {input_dir} 下没有找到任何 .txt 文件。")

    # Important: process per-file to avoid mixing topic/body across documents.
    chunks: List[str] = []
    for p in files:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        norm = _normalize_text(raw)
        if not norm:
            continue

        for c in _chunks_with_topic(norm, args.chunk_chars):
            if len(c) >= args.min_chars:
                chunks.append(c)

    if not chunks:
        raise SystemExit("切分后没有任何有效片段，请调小 min_chars 或检查原始文本。")

    if args.mode == "cpt":
        rows = ({"text": c} for c in chunks)
        data_file = f"{args.dataset_name}.jsonl"
        n = _write_jsonl(output_dir / data_file, rows)
    else:
        def _rows():
            for c in chunks:
                yield {
                    "instruction": args.instruction,
                    "input": c,
                    "output": c,
                    "system": args.system,
                }

        data_file = f"{args.dataset_name}.jsonl"
        n = _write_jsonl(output_dir / data_file, _rows())

    # Merge/update dataset_info.json in output_dir
    info_path = output_dir / "dataset_info.json"
    if info_path.exists():
        try:
            cur = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    else:
        cur = {}
    cur.update(_build_dataset_info(args.dataset_name, data_file, args.mode))
    info_path.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: 写入 {n} 条样本到 {output_dir / data_file}")
    print(f"OK: 更新 {info_path}，数据集名: {args.dataset_name}")
    print("下一步：在训练配置里设置 dataset_dir 指向该 output_dir，并把 dataset 设置为该数据集名。")


if __name__ == "__main__":
    # Avoid huggingface import side effects; this script should stay lightweight.
    main()

