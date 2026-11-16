"""Chunking pipeline for Day 7 Agentic RAG.
Extracts text, tables, and section metadata from the 112-page Vinamilk 2025 Annual Report.
Outputs clean chunks according to Day 7 JSON contract.
"""
import json
import re
from pathlib import Path
import pymupdf
import pymupdf4llm


def extract_section(lines: list[str]) -> str:
    """Identify document section from standard running header lines."""
    if len(lines) >= 2:
        candidate = lines[1].strip()
        if len(candidate) > 3 and not candidate.isdigit():
            return candidate
    return "Thông tin chung"


def split_text_into_blocks(text: str, max_chars: int = 700, overlap_chars: int = 100) -> list[str]:
    """Split text into coherent blocks (paragraphs/headers) respecting size budget."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    blocks: list[str] = []
    current_block: list[str] = []
    current_len = 0

    for para in paragraphs:
        # If it's a markdown table, keep it as its own block
        if '|' in para and ('|---|' in para or '|:---|' in para):
            if current_block:
                blocks.append('\n\n'.join(current_block))
                current_block = []
                current_len = 0
            blocks.append(para)
            continue

        if current_len + len(para) + 2 > max_chars and current_block:
            blocks.append('\n\n'.join(current_block))
            current_block = [para]
            current_len = len(para)
        else:
            current_block.append(para)
            current_len += len(para) + 2

    if current_block:
        blocks.append('\n\n'.join(current_block))

    return blocks


def process_pdf(pdf_path: str | Path, output_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    doc = pymupdf.open(pdf_path)
    print(f"Processing PDF: {pdf_path} ({len(doc)} pages)...")

    # Extract markdown pages using pymupdf4llm
    page_data = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    all_chunks = []
    source_id = "vnm-ar-2025"
    source_name = pdf_path.as_posix()

    for idx, pinfo in enumerate(page_data):
        page_num = idx + 1
        page_text = pinfo.get("text", "").strip()
        if not page_text:
            continue

        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        section = extract_section(lines)

        # Check for images / drawings on page
        images = doc[idx].get_images()
        has_images = len(images) > 0

        # Split into blocks
        blocks = split_text_into_blocks(page_text)
        if not blocks:
            blocks = [page_text]

        for c_idx, block in enumerate(blocks):
            chunk_type = "table" if ("|" in block and "---" in block) else ("image_page" if has_images and c_idx == 0 else "text")
            chunk_id = f"vnm-2025:p{page_num:03d}:c{c_idx+1:02d}"

            chunk = {
                "id": chunk_id,
                "text": block,
                "source_id": source_id,
                "source": source_name,
                "page": page_num,
                "metadata": {
                    "section": section,
                    "type": chunk_type,
                    "image_count": len(images),
                    "char_count": len(block)
                }
            }
            all_chunks.append(chunk)

    print(f"Generated {len(all_chunks)} chunks across {len(doc)} pages.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved chunks to {output_path}")
    return all_chunks


if __name__ == "__main__":
    pdf = "data/20260319_VNM_Bao_cao_thuong_nien_2025_eae619d5f6.pdf"
    out = "data/chunks.json"
    process_pdf(pdf, out)
