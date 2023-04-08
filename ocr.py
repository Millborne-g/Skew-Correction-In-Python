import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img_file = "image/PAQ323.jpg"

img = Image.open(img_file)

result = pytesseract.image_to_string(img)

print('test '+result)
