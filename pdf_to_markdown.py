#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import signal
import time
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import torch
from transformers import AutoModel, AutoTokenizer


PROMPT = """<image>
<|grounding|>
Extract all visible text in natural reading order.
Return the content of the page. Do not add extra instructions.
"""

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_PAGE_TIMEOUT_SEC = 300


def render_pdf_page(doc: fitz.Document, page_index: int, dpi: int, temp_dir: Path) -> Path:
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image_path = temp_dir / f"page_{page_index:03}.png"
    pix.save(str(image_path))
    return image_path




def _coerce_markdown_output(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "pred", "result", "output"):
            if key in result:
                return result[key]
    if isinstance(result, (list, tuple)) and result:
        if all(isinstance(item, str) for item in result):
            return "\n".join(result)
        if isinstance(result[0], dict):
            for key in ("text", "pred", "result", "output"):
                if key in result[0]:
                    return result[0][key]
    raise TypeError(f"Unexpected inference output type: {type(result)}")


def detect_repeated_lines(text: str, min_repeats: int = 3) -> list[str]:
    repeats: list[str] = []
    current_line: str | None = None
    count = 0

    for line in text.splitlines():
        normalized = line.strip()
        if not normalized:
            current_line = None
            count = 0
            continue

        if normalized == current_line:
            count += 1
        else:
            current_line = normalized
            count = 1

        if count == min_repeats:
            repeats.append(normalized)

    return repeats


def save_debug_result(debug_dir: Path, page_index: int, result: object, error: Exception) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "page": page_index,
        "error": str(error),
        "error_type": type(error).__name__,
        "result_type": str(type(result)),
        "result": result,
    }
    debug_path = debug_dir / f"page_{page_index:03}_result.json"
    debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=repr), encoding="utf-8")


def read_saved_result(result_dir: Path) -> str:
    primary_candidates = [result_dir / "result.mmd", result_dir / "result.md"]
    for candidate in primary_candidates:
        if candidate.exists():
            content = candidate.read_text(encoding="utf-8")
            if content.strip():
                return content

    markdown_candidates = list(result_dir.glob("*.mmd")) + list(result_dir.glob("*.md"))
    markdown_candidates = [path for path in markdown_candidates if path.is_file()]
    markdown_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in markdown_candidates:
        content = candidate.read_text(encoding="utf-8")
        if content.strip():
            return content
    return ""


def normalize_placeholders_and_save_images(
    raw_text: str, image_path: Path, output_dir: Path, page_index: int
) -> str:
    page_label = f"{page_index:03d}"
    figure_name = f"figure_page_{page_label}.png"
    chart_name = f"chart_page_{page_label}.png"

    figure_pattern = re.compile(r"!\[\s*figure(?:s)?\s*\]\s*(\([^)]*\))?", re.IGNORECASE)
    chart_pattern = re.compile(r"!\[\s*chart(?:s)?\s*\]\s*(\([^)]*\))?", re.IGNORECASE)

    raw_text = figure_pattern.sub(f"![figure](images/{figure_name})", raw_text)
    raw_text = chart_pattern.sub(f"![chart](images/{chart_name})", raw_text)
    return raw_text


def parse_blocks_from_raw_text(raw: str) -> tuple[list[dict], str | None]:
    if "<|ref|>" not in raw and "<|det|>" not in raw:
        return [], None

    pattern = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>", re.DOTALL)
    matches = list(pattern.finditer(raw))
    if not matches:
        return [], None

    blocks = []
    for idx, match in enumerate(matches):
        ref_content = match.group(1).strip()
        det_content = match.group(2).strip()

        text_start = match.end()
        text_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        text_content = raw[text_start:text_end].strip()

        blocks.append(
            {
                "ref": ref_content,
                "det": det_content,
                "text": text_content,
            }
        )

    warning = None
    ref_open = raw.count("<|ref|>")
    ref_close = raw.count("<|/ref|>")
    det_open = raw.count("<|det|>")
    det_close = raw.count("<|/det|>")
    if ref_open != ref_close or det_open != det_close or len(matches) != min(ref_open, det_open):
        warning = "partial_parse: malformed ref/det tags present"

    return blocks, warning


def join_text_blocks(blocks: list[dict]) -> str:
    texts = [block.get("text", "").strip() for block in blocks]
    texts = [text for text in texts if text]
    return "\n\n".join(texts).strip()


def strip_ref_det_tags(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    text = text.replace("<|ref|>", "").replace("<|/ref|>", "")
    text = text.replace("<|det|>", "").replace("<|/det|>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run_deepseek_ocr_on_image(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    image_path: Path,
    prompt: str,
    output_dir: Path,
    page_index: int,
    debug_dir: Path,
    max_new_tokens: int,
) -> str:
    result = None
    try:
        with MaskedScatterDtypeFix(), torch.no_grad():
            try:
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=str(output_dir),
                    base_size=1536,
                    image_size=1536,
                    crop_mode=False,
                    save_results=False,
                    test_compress=False,
                    eval_mode=True,
                    max_new_tokens=max_new_tokens,
                )
            except TypeError as exc:
                if "max_new_tokens" not in str(exc):
                    raise
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=str(output_dir),
                    base_size=1536,
                    image_size=1536,
                    crop_mode=False,
                    save_results=False,
                    test_compress=False,
                    eval_mode=True,
                )

        raw_text = ""
        if result is not None:
            raw_text = _coerce_markdown_output(result)

        if not raw_text.strip():
            page_infer_dir = output_dir / "_infer_tmp" / f"page_{page_index:03}"
            page_infer_dir.mkdir(parents=True, exist_ok=True)
            with MaskedScatterDtypeFix(), torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                try:
                    model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=str(image_path),
                        output_path=str(page_infer_dir),
                        base_size=1536,
                        image_size=1536,
                        crop_mode=False,
                        save_results=True,
                        test_compress=False,
                        eval_mode=False,
                        max_new_tokens=max_new_tokens,
                    )
                except TypeError as exc:
                    if "max_new_tokens" not in str(exc):
                        raise
                    model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=str(image_path),
                        output_path=str(page_infer_dir),
                        base_size=1536,
                        image_size=1536,
                        crop_mode=False,
                        save_results=True,
                        test_compress=False,
                        eval_mode=False,
                    )
            raw_text = read_saved_result(page_infer_dir)
            if not raw_text.strip():
                print(f"Page {page_index + 1}: no saved markdown found in {page_infer_dir}")

        if not raw_text.strip():
            warning_exc = RuntimeError("Empty OCR output after fallback.")
            save_debug_result(debug_dir, page_index, result, warning_exc)
            print(f"Page {page_index + 1}: empty OCR output after fallback.")
            return ""

        return raw_text
    except Exception as exc:
        save_debug_result(debug_dir, page_index, result, exc)
        raise


def save_page_json(page_payload: dict, pages_dir: Path, page_index: int) -> Path:
    page_path = pages_dir / f"page_{page_index:03}.json"
    page_path.write_text(json.dumps(page_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return page_path


def merge_json_pages(pages_dir: Path, output_file: Path) -> None:
    page_files = sorted(pages_dir.glob("page_*.json"))
    pages = [json.loads(page_file.read_text(encoding="utf-8")) for page_file in page_files]
    output_file.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_json_pages_for_range(pages_dir: Path, output_file: Path, page_indices: list[int]) -> None:
    pages = []
    missing = []
    for page_index in page_indices:
        page_path = pages_dir / f"page_{page_index:03}.json"
        if page_path.exists():
            pages.append(json.loads(page_path.read_text(encoding="utf-8")))
        else:
            missing.append(page_index)
            pages.append(
                {
                    "page": page_index,
                    "raw_text": "",
                    "blocks": [],
                    "source": "deepseek-ocr",
                    "notes": "missing page output in this run",
                }
            )
    if missing:
        print(f"Missing JSON pages during merge: {missing}")
    output_file.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")


class PageTimeout:
    _warned = False

    def __init__(self, seconds: int, page_index: int) -> None:
        self.seconds = seconds
        self.page_index = page_index
        self._previous_handler = None
        self._enabled = False

    def _handle_timeout(self, *_args):
        raise TimeoutError(
            f"Page {self.page_index + 1} timed out after {self.seconds} seconds."
        )

    def __enter__(self):
        if not self.seconds or self.seconds <= 0:
            return self
        if not hasattr(signal, "SIGALRM"):
            if not PageTimeout._warned:
                print("Page timeout disabled: SIGALRM not available on this platform.")
                PageTimeout._warned = True
            return self
        self._previous_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        self._enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._previous_handler is not None:
                signal.signal(signal.SIGALRM, self._previous_handler)
        return False


class MaskedScatterDtypeFix:
    def __enter__(self):
        self._orig = torch.Tensor.masked_scatter_

        def patched(tensor, mask, source):
            if isinstance(source, torch.Tensor) and tensor.dtype != source.dtype:
                source = source.to(tensor.dtype)
            return self._orig(tensor, mask, source)

        torch.Tensor.masked_scatter_ = patched
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.Tensor.masked_scatter_ = self._orig
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF to JSON with DeepSeek-OCR.")
    parser.add_argument("input_pdf", type=Path, help="Input PDF path")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--page-timeout-sec", type=int, default=DEFAULT_PAGE_TIMEOUT_SEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {args.input_pdf}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure GPU and CUDA are configured.")

    if args.start_page < 0:
        raise ValueError("--start-page must be >= 0")
    if args.max_pages is not None and args.max_pages <= 0:
        raise ValueError("--max-pages must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.page_timeout_sec <= 0:
        raise ValueError("--page-timeout-sec must be > 0")

    output_dir = args.output_dir
    pages_dir = output_dir / "pages"
    debug_dir = output_dir / "debug"
    pages_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DeepSeek-OCR model on CUDA...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2"
    model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    model = model.to("cuda")
    model = model.eval()
    model = model.to(dtype=torch.float16)

    device_type = next(model.parameters()).device.type
    if device_type != "cuda":
        raise RuntimeError(f"Model is not on CUDA (device={device_type}).")

    model_dtype = next(model.parameters()).dtype
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model dtype: {model_dtype}")
    print(f"Attention implementation: {attn_impl}")

    total_start = time.perf_counter()
    with fitz.open(args.input_pdf) as doc, tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        total_pages = doc.page_count
        start_page = args.start_page
        if start_page >= total_pages:
            raise ValueError(
                f"--start-page {start_page} is out of range for document with {total_pages} pages."
            )
        end_page = total_pages
        if args.max_pages is not None:
            end_page = min(total_pages, start_page + args.max_pages)
        page_indices = list(range(start_page, end_page))

        print(f"Rendering and OCR on {len(page_indices)} pages...")

        for page_index in page_indices:
            page_start = time.perf_counter()
            image_path = render_pdf_page(doc, page_index, args.dpi, temp_dir)
            warnings = []
            try:
                with PageTimeout(args.page_timeout_sec, page_index):
                    raw_text = run_deepseek_ocr_on_image(
                        model=model,
                        tokenizer=tokenizer,
                        image_path=image_path,
                        prompt=PROMPT,
                        output_dir=output_dir,
                        page_index=page_index,
                        debug_dir=debug_dir,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if not raw_text.strip():
                        warnings.append("empty_output")
                    else:
                        repeats = detect_repeated_lines(raw_text)
                        if repeats:
                            warnings.append("repeated_generation")
            except TimeoutError as exc:
                raw_text = ""
                warnings.append("timeout")
                print(str(exc))

            raw_text = normalize_placeholders_and_save_images(
                raw_text=raw_text,
                image_path=image_path,
                output_dir=output_dir,
                page_index=page_index,
            )
            blocks, parse_warning = parse_blocks_from_raw_text(raw_text)
            if blocks:
                raw_text = join_text_blocks(blocks)
            else:
                raw_text = strip_ref_det_tags(raw_text)

            page_payload = {
                "page": page_index,
                "raw_text": raw_text,
                "blocks": blocks,
                "source": "deepseek-ocr",
                "notes": "raw generation, not post-processed",
            }
            if parse_warning:
                page_payload["notes"] = f"{page_payload['notes']} | {parse_warning}"
            if warnings:
                page_payload["warnings"] = warnings
            save_page_json(page_payload, pages_dir, page_index)
            page_elapsed = time.perf_counter() - page_start
            print(f"Page {page_index + 1}/{total_pages} done in {page_elapsed:.2f}s")

    merge_json_pages_for_range(pages_dir, output_dir / "document.json", page_indices)
    total_elapsed = time.perf_counter() - total_start
    print(f"All pages merged to {output_dir / 'document.json'} in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
