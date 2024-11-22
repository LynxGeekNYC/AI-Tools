import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import json
import os

def extract_text_from_pdf(input_pdf, output_file="output.json"):
    """Extract and save text from both structured and scanned PDF."""
    # Create/open the JSON file for writing
    with open(output_file, "w", encoding="utf-8") as f:
        # Start the JSON structure
        f.write('{\n  "pages": [\n')

        doc = fitz.open(input_pdf)  # Open PDF
        first_page = True

        for page_num in range(doc.page_count):
            print(f"Processing page {page_num + 1}/{doc.page_count}...")

            page = doc.load_page(page_num)
            structured_text = page.get_text()

            # If no text is found, perform OCR on the page (hand-written text)
            if not structured_text.strip():
                print(f"Page {page_num + 1} is scanned. Performing OCR...")
                ocr_text = perform_ocr_on_page(input_pdf, page_num)
            else:
                ocr_text = ""

            # Prepare JSON for this page
            page_data = {
                "page_number": page_num + 1,
                "structured_text": structured_text,
                "ocr_text": ocr_text
            }

            # Write page data to JSON, handling commas correctly
            if not first_page:
                f.write(",\n")
            json.dump(page_data, f, ensure_ascii=False, indent=4)
            first_page = False

        # Close the JSON structure
        f.write("\n  ]\n}\n")

        doc.close()
        print(f"Extraction completed. Data saved to {output_file}.")

def perform_ocr_on_page(input_pdf, page_num):
    """Perform OCR on a single page and return extracted text."""
    # Convert the specified page to an image
    images = convert_from_path(input_pdf, first_page=page_num + 1, last_page=page_num + 1)

    ocr_text = ""
    for image in images:
        # Optimize image for OCR (reduce size)
        ocr_text += pytesseract.image_to_string(image, config='--dpi 150')

    return ocr_text

if __name__ == "__main__":
    input_pdf = "input.pdf"  # Replace with the path to your PDF
    output_json = "output.json"  # Output JSON path

    print("Starting PDF extraction...")
    extract_text_from_pdf(input_pdf, output_json)
