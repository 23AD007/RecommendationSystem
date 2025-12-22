from pdfminer.high_level import extract_text
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF document.

    Returns
    -------
    str
        Extracted text content from the PDF.

    Raises
    ------
    FileNotFoundError
        If the PDF file does not exist.
    RuntimeError
        If text extraction fails.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        text = extract_text(pdf_file)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

    if not text or not text.strip():
        raise RuntimeError("No text could be extracted from the PDF")

    return text


if __name__ == "__main__":
    # Manual test
    sample_pdf = "data/documents/supplier_report_2025.pdf"
    extracted = extract_text_from_pdf(sample_pdf)
    print(extracted[:1000])
