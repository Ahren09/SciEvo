import os
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader

# Define the arXiv paper URL
paper_url = "https://arxiv.org/pdf/2305.19523.pdf"
paper_id = paper_url.split("/")[-1].split(".")[0]

# Set the directory to save the downloaded PDF and extracted content
output_directory = "paper_extraction"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)


pdf_filename = os.path.join(output_directory, f"{paper_id}.pdf")
os.system(f"wget -O {pdf_filename} {paper_url}")

# Extract text from the PDF
text_filename = os.path.join(output_directory, f"{paper_id}.txt")
with open(text_filename, "w", encoding="utf-8") as text_file:
    pdf = PdfFileReader(open(pdf_filename, "rb"))
    for page_num in range(pdf.getNumPages()):
        page = pdf.getPage(page_num)
        text_file.write(page.extractText())

# Extract images from the PDF
images = convert_from_path(pdf_filename, output_folder=output_directory)

# Save images as separate files
for i, image in enumerate(images):
    image.save(os.path.join(output_directory, f"page_{i + 1}.png"), "PNG")

print(f"PDF and content extraction completed. PDF saved as {pdf_filename}")
