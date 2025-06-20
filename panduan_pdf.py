from pathlib import Path
import fitz  # PyMuPDF

def is_scanned_pdf(pdf_path):
    """
    判断整个 PDF 是否为扫描件：
    若所有页都没有可选文字，则视为扫描 PDF。
    若任意一页有可选文字，则视为非扫描 PDF。
    """
    if not isinstance(pdf_path, Path):
        pdf_path = Path(pdf_path)

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    scanned_pages = 0

    for page_num in range(num_pages):
        page = doc[page_num]
        text = page.get_text("text").strip()  # 提取纯文本
        if text == "":
            scanned_pages += 1
            print(f"Page {page_num + 1}: ❌ No text -> Scanned page")
        else:
            print(f"Page {page_num + 1}: ✅ Has text -> Non-scanned page")

    if scanned_pages == num_pages:
        print("\n📌 判断结果：整个 PDF 是扫描件（纯图片）")
        return True
    else:
        print("\n📌 判断结果：PDF 不是纯扫描件（包含可选文字）")
        return False


if __name__ == "__main__":
    # 替换成你的 PDF 路径
    pdf_file = r"D:\hp\code\RAG\data_process\MinerU\demo\pdfs\small_ocr.pdf"
    is_scanned_pdf(pdf_file)
