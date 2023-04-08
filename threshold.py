import cv2
import pytesseract
import os
from PIL import Image
import re
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread('image/green.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# _, result = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

adaptive_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 30) #the 15 is where I input the noise

# cv2.imshow("result", result)
# cv2.waitKey(0)

# Save the result to an image file
cv2.imwrite('image/result.jpg', adaptive_result)

img_file = "image/result.jpg"

img_ocr = Image.open(img_file)
result_ocr = ''
try:
    text = pytesseract.image_to_string(img_ocr, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    # clean tesseract text by removing any unwanted blank spaces
    clean_text = re.sub('[\W_]+', '', text)
    result_ocr += clean_text
except: 
    text = None

#result_ocr = pytesseract.image_to_string(img_ocr)

print("result "+result_ocr)

