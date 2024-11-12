import subprocess

def convert_md_to_pdf(input_md, output_pdf):
    result = subprocess.run(["pandoc", input_md, "-o", output_pdf, "--pdf-engine=xelatex"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error during PDF creation:")
        print(result.stderr)
    else:
        print("PDF created successfully.")
