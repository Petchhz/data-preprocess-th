import pandas as pd
import re
from pythainlp import word_tokenize
from sacremoses import MosesTokenizer
from tqdm import tqdm

# ========================================================================================

# location path
datasets_path = "E:/superai2-work/section-preprocess/datasets/csv-files/"
save_dir = "E:/superai2-work/section-preprocess/datasets/clean-data/"
name_file = "test.csv"

# load csv file
df = pd.read_csv(datasets_path + name_file)
print(df)

# load EN tokenize
tokenize = MosesTokenizer(lang='en')

# ========================================================================================

# delete emoji
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        u"\u266A" # EIGHTHNOTE
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

# clean and tokenize data (th)
def preprocess_text(text):
    final = deEmojify(str(text))
    punctuations = '''+-;\•./@#^$%^&*_~ฯ'''
    final = "".join(u for u in final if u not in punctuations)
    #final = final.replace("“",'"')
    #final = final.replace("”",'"')
    final = word_tokenize(final, engine = "deepcut")
    final = " ".join(word for word in final)
    return final

# clean and tokenize data (en)
def preprocess_text_en(text):
    final = deEmojify(str(text))
    punctuations = '''+-;\•/@#^$%^&*_~'''
    final = "".join(u for u in final if u not in punctuations)
    final = tokenize.tokenize(final, return_str=True)
    return final

# ========================================================================================

# tokenize Thai text
text_train = []
sentences = list(df['raw_th_text'])
for sen in sentences:
    text_train.append(preprocess_text(sen))
#print(text_train)
#print(len(text_train))
#print(type(text_train))
new_len_text_th = []
for word in tqdm(list(text_train)):
    words = word.split()
    new_len_text_th.append(len(words))
#print(len(new_len_text))
#print(type(new_len_text))

col1_th = ["th_text"]
add_th_text = pd.DataFrame(text_train, columns=col1_th)
result_th_text = pd.concat([df, add_th_text], axis=1)

col2_th = ["th_tokens_len"]
add_th_tokens = pd.DataFrame(new_len_text_th, columns=col2_th)
result_th = pd.concat([result_th_text, add_th_tokens], axis=1)

# ========================================================================================

# tokenize English text
en_text_train = []
sentences = list(df['raw_en_text'])
for sen in sentences:
    en_text_train.append(preprocess_text_en(sen))
#print(en_text_train)
#print(len(en_text_train))
#print(type(en_text_train))

new_len_text_en = []
for worde in tqdm(list(en_text_train)):
    wordse = worde.split()
    new_len_text_en.append(len(wordse))
#print(new_len_text)
#print(len(new_len_text))
#print(type(new_len_text))

col1_en = ["en_text"]
add_en_text = pd.DataFrame(en_text_train, columns=col1_en)
result_en_text = pd.concat([result_th, add_en_text], axis=1)

col2_en = ["en_tokens_len"]
add_en_tokens = pd.DataFrame(new_len_text_en, columns=col2_en)
results = pd.concat([result_en_text, add_en_tokens], axis=1)

# ========================================================================================

# save csv file
results.to_csv(save_dir + "clean" + "-" + name_file, index=False, encoding="utf-8-sig")

# ========================================================================================
