# import tqdm
from tqdm import tqdm
# import trans module
import googletrans
translator = googletrans.Translator()
# target file path
text_file_path = '/Users/jeongwooKim/Desktop/DeepLearningMIT/src/ch2.txt'
target = open(text_file_path, 'r')
target_contents = target.read()
result_file_path = '/Users/jeongwooKim/Desktop/DeepLearningMIT/src/MIT_DL_ch2.txt'
# result file path
result = open(result_file_path, 'w')
pbar = tqdm(target_contents.split("\n"))
# RUN
for _str in pbar:
    # if (str is time index? == TRUE)
    # >> SKIP trans..
    if ((_str[1] == ":") or (_str[2] == ":")):
        result.write(_str + "\n")
        continue
    _outStr = translator.translate(_str, dest='ko', src='en')
    result.write(_str + "\n" + _outStr.text + "\n")
    print(_outStr.text)
# SAVE...     
target.close()
result.close()