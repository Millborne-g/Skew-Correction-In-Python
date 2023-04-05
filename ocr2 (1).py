import cv2
import pytesseract
import os
from PIL import Image
import imutils
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

test = "C://tflite1//ocr"
'''
for filename in os.listdir(test):
	if filename.endswith('.jpg'):
		image_path = os.path.join(test,filename)
		img = cv2.imread(image_path)


		#resize = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
		
		grayscale_resize_test_license_plate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0)
		
		new_predicted_result_GWT2180 = pytesseract.image_to_string(gaussian_blur_license_plate, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
		
		filter_new_predicted_result_GWT2180 = "".join(new_predicted_result_GWT2180.split()).replace(":", "").replace("-", "")
		
		print(f'{filename}: {filter_new_predicted_result_GWT2180}')
'''
'''
for filename in os.listdir(test):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(test, filename)
        img = cv2.imread(image_path)
        text = pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        print(f'{filename}: {text}')

'''
'''
for filename in os.listdir(test):
	if filename.endswith('.jpg') or filename.endswith('.png'):
		image_path = os.path.join(test, filename)
		img = cv2.imread(image_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.medianBlur(gray,3)
		gray = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		text = pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
		print(f'{filename}: {text}')
'''
'''
for filename in os.listdir(test):
	if filename.endswith('.jpg') or filename.endswith('.png'):
		image_path = os.path.join(test, filename)
		img = cv2.imread(image_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		gray = clahe.apply(gray)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
		text = pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
		print(f'{filename}: {text}')
'''
'''
for filename in os.listdir(test):
	if filename.endswith('.jpg') or filename.endswith('.png'):
		image_path = os.path.join(test, filename)
		img = cv2.imread(image_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		text = pytesseract.image_to_string(gray, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
		print(f'{filename}: {text}')
'''
'''
for filename in os.listdir(test):
	if filename.endswith('.jpg') or filename.endswith('.png'):
		image_path = os.path.join(test, filename)
		img = cv2.imread(image_path)		
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		text = pytesseract.image_to_string(thresh, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
		print(f'{filename}: {text}')

'''