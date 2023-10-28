import os
import os.path as osp


# Define the URL to the abstract of the arXiv paper

def download_arXiv_paper_pdf(paper_id):
    """
    Downloads a PDF file of an arXiv paper given its unique paper ID.

    Args:
        paper_id (str): The unique identifier of the arXiv paper.

    Returns:
        None

    This function fetches the PDF file of the arXiv paper with the specified ID
    and saves it to the 'data/pdf' directory.

    Example:
        >>> download_arXiv_paper_pdf("2310.13132")
    """
    paper_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    paper_id = paper_url.split("/")[-1].split(".pdf")[0]

    # Set the directory to save the downloaded PDF and extracted content
    output_dir = osp.join("data", "pdf")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    pdf_filename = os.path.join(output_dir, f"{paper_id}.pdf")

    command = f"wget -O --user-agent='Mozilla/5.0' {pdf_filename} {paper_url}"

    print(command)

    os.system(command)


if __name__ == "__main__":
    # Example usage
    paper_abs_url = "https://arxiv.org/abs/2310.13132"
    paper_id = paper_abs_url.split("/abs/")[-1].split("/")[0]
    print(f"ID: {paper_id}")

    download_arXiv_paper_pdf(paper_id)
