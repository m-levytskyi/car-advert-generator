import subprocess

def convert_md_to_pdf(input_html, output_pdf):
    result = subprocess.run(['wkhtmltopdf', '--enable-local-file-access', input_html, output_pdf])
    if result.returncode != 0:
        print("Error during PDF creation:")
        print(result.stderr)
    else:
        print(f"PDF created successfully at {output_pdf}")
