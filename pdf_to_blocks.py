#!/usr/bin/env python3
import argparse
import json
import re
import time
import tempfile
from pathlib import Path

import fitz
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
PROMPT = (
    "<image> <|grounding|>\n"
    "Extract all visible content from the page and return it as structured blocks."
)


def render_pdf_page(doc, page_index, dpi, images_dir):
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    image_path = images_dir / f"page_{page_index:03d}.png"
    pix.save(str(image_path))
    return image_path


def _extract_json_from_text(text):
    text = text.strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON object or array found in model output.")


_REF_DET_RE = re.compile(
    r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL
)
_REF_LABEL_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL
)
_ALLOWED_TYPES = {"title", "text", "table", "list", "figure"}
_TYPE_MAP = {"image": "figure"}


def _fallback_blocks_from_text(text):
    cleaned = _REF_DET_RE.sub("", text)
    cleaned = (
        cleaned.replace("<|ref|>", "")
        .replace("<|/ref|>", "")
        .replace("<|det|>", "")
        .replace("<|/det|>", "")
    )
    cleaned = cleaned.strip()
    if not cleaned:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]
    return [{"type": "text", "content": p} for p in paragraphs]


def _blocks_from_grounding_tags(text):
    labels = _REF_LABEL_RE.findall(text)
    blocks = []
    for label in labels:
        label_norm = label.strip().lower()
        mapped = _TYPE_MAP.get(label_norm, label_norm)
        if mapped not in _ALLOWED_TYPES:
            mapped = "text"
        blocks.append({"type": mapped, "content": ""})
    return blocks


def _normalize_blocks(obj):
    if isinstance(obj, dict) and "blocks" in obj:
        blocks = obj["blocks"]
    else:
        blocks = obj

    if not isinstance(blocks, list):
        raise ValueError("Model output JSON is not a list of blocks.")

    cleaned = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("Block is not an object.")
        if "type" not in block or "content" not in block:
            raise ValueError("Block missing required keys: type/content.")
        if not isinstance(block["type"], str) or not isinstance(block["content"], str):
            raise ValueError("Block type/content must be strings.")
        cleaned.append({"type": block["type"], "content": block["content"]})
    return cleaned


def run_deepseek_ocr_on_image(model, tokenizer, image_path, output_dir):
    with torch.no_grad():
        result = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=str(image_path),
            output_path=str(output_dir),
            base_size=1024,
            image_size=1024,
            crop_mode=False,
            save_results=False,
            test_compress=False,
            eval_mode=True,
        )

    raw_text = None
    obj = None

    if isinstance(result, str):
        raw_text = result
        try:
            obj = _extract_json_from_text(result)
        except ValueError:
            obj = None
    elif isinstance(result, dict):
        if "text" in result and isinstance(result["text"], str):
            raw_text = result["text"]
            try:
                obj = _extract_json_from_text(raw_text)
            except ValueError:
                obj = None
        else:
            obj = result
    elif isinstance(result, (list, tuple)) and result:
        if all(isinstance(item, dict) for item in result):
            obj = list(result)
        elif isinstance(result[0], str):
            raw_text = result[0]
            try:
                obj = _extract_json_from_text(raw_text)
            except ValueError:
                obj = None
        else:
            obj = result[0]
    else:
        obj = result

    if isinstance(obj, (dict, list)):
        return _normalize_blocks(obj)

    if raw_text is None:
        raw_text = str(result)

    blocks = _fallback_blocks_from_text(raw_text)
    if blocks:
        return blocks
    return _blocks_from_grounding_tags(raw_text)


def save_page_blocks(blocks, output_path):
    output_path.write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def merge_page_blocks(page_blocks, output_path):
    document = {"pages": page_blocks}
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to structured JSON blocks using DeepSeek-OCR."
    )
    parser.add_argument("pdf", type=Path, help="Input PDF file")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    if args.max_pages is not None and args.max_pages <= 0:
        raise ValueError("--max-pages must be a positive integer.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. torch.cuda.is_available() must be True.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    output_dir = args.output_dir
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_entries = []
    with tempfile.TemporaryDirectory(prefix="dpsk_pages_") as tmp_dir:
        images_dir = Path(tmp_dir)
        with fitz.open(args.pdf) as doc:
            total_pages = doc.page_count
            if args.max_pages is not None:
                total_pages = min(total_pages, args.max_pages)
            for page_index in range(total_pages):
                start = time.perf_counter()
                image_path = render_pdf_page(doc, page_index, args.dpi, images_dir)
                blocks = run_deepseek_ocr_on_image(
                    model, tokenizer, image_path, output_dir
                )
                save_page_blocks(blocks, pages_dir / f"page_{page_index:03d}.json")
                page_entries.append({"page": page_index, "blocks": blocks})
                elapsed = time.perf_counter() - start
                print(f"page {page_index}: {elapsed:.2f}s")

    merge_page_blocks(page_entries, output_dir / "document.json")


if __name__ == "__main__":
    main()
