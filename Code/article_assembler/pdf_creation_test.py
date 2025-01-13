from article_assembler import ArticleAssembler

template_path="Code/article_assembler/nolatex_article_template.html"


img_dir="Code/article_assembler/tmp/imgs"
json_path="Code/article_agent/json/output.json"
output_pdf_path="Code/article_assembler/pdfs/test_article.pdf"
paragraps=["This is a test article"]

assembler = ArticleAssembler(template_file=template_path, img_dir=img_dir)

#load json
car_data = assembler.load_json_data(json_path)

paragraphs = car_data["paragraphs"]
captions = car_data["captions"]
car_brand = car_data["brand"]
car_type = car_data["car_type"]

#generated images are taken from the img_dir

#create html file
html_file = assembler.populate_template(paragraphs, captions, img_dir, car_brand, car_type)

#convert to pdf
assembler.convert_to_pdf(html_file, output_pdf_path)