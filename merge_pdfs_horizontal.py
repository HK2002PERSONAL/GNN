"""
将 z_port_pdfs 目录下的三个 PDF 按页横向拼接（左右拼）：
- 每个输入 PDF 有 N 页，则输出 PDF 也有 N 页
- 输出第 i 页 = 三个输入第 i 页从左到右拼接
依赖：pypdf（testbench 环境自带）
"""
import os
from pypdf import PdfReader, PdfWriter, Transformation

PDF_DIR = "z_port_pdfs"
# 逻辑顺序：0_1 -> 1_0 -> 10_
PDF_PATHS = [
    os.path.join(PDF_DIR, "T[II]g[0_1].pdf"),
    os.path.join(PDF_DIR, "T[II]g[1_0].pdf"),
    os.path.join(PDF_DIR, "T[II]g[10_].pdf"),
]
OUTPUT_PATH = os.path.join(PDF_DIR, "T[II]Ports.pdf")


def merge_horizontal(pdf_paths, output_path):
    readers = [PdfReader(p) for p in pdf_paths]
    page_counts = [len(r.pages) for r in readers]
    n_pages = max(page_counts)
    print(f"输入文件页数: {dict(zip(pdf_paths, page_counts))}, 输出页数: {n_pages}")

    writer = PdfWriter()
    for i in range(n_pages):
        pages = [r.pages[i] for r in readers if i < len(r.pages)]
        widths = [float(p.mediabox.width) for p in pages]
        heights = [float(p.mediabox.height) for p in pages]
        total_w = sum(widths)
        max_h = max(heights)

        blank = writer.add_blank_page(width=total_w, height=max_h)

        x_offset = 0.0
        for p in pages:
            # 将源页平移到 (x_offset, 0)，并修正源页自身的 mediabox 原点
            tx = x_offset - float(p.mediabox.left)
            ty = -float(p.mediabox.bottom)
            blank.merge_transformed_page(p, Transformation().translate(tx, ty))
            x_offset += float(p.mediabox.width)

    with open(output_path, "wb") as f:
        writer.write(f)
    print(f"横向拼接完成，输出: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    merge_horizontal(PDF_PATHS, OUTPUT_PATH)