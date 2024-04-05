import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# # Mapping dictionaries for character conversion
# dict_char_to_int = {'O': '0',
#                     'I': '1',
#                     'J': '3',
#                     'A': '4',
#                     'G': '6',
#                     'S': '5'}

# dict_int_to_char = {'0': 'O',
#                     '1': 'I',
#                     '3': 'J',
#                     '4': 'A',
#                     '6': 'G',
#                     '5': 'S'}
# numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# # Checkers
# def checkNums(text):
#     if text in numbers or text in dict_char_to_int.keys():
#         return True
#     return False

# def checkChar(text):
#     if text in string.ascii_uppercase or text in dict_int_to_char.keys():
#         return True
#     return False

# # Check if license format is correct
# def licenseFormat(text): 
#     if(checkNums(text[0]) and checkNums(text[1]) and checkChar(text[2])):
#         for char in text[4: ]:
#             if(char == "-" or char == "."):
#                 continue
#             if(not checkNums(char)):
#                 return False
#     else: 
#         return False;        
#     return True

# Read license plate
def readLicensePlate(licensePlateCrop):
    resultLicense = ''
    scores = []
    detections = reader.readtext(licensePlateCrop)
    if not detections:
        print("No text detected")
    else:
        for detection in detections:
            bbox, text, score = detection
            resultLicense += text
            scores.append(score)
            print("Detected text:", text)
    return resultLicense, scores