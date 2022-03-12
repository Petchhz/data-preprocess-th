import pandas as pd

# ========================================================================================

# load test dataset
path = "E:/superai2-work/section-preprocess/datasets/text-files/"
dataset_th_dir = path + "test_th.txt"
dataset_en_dir = path + "test_en.txt"

# save location
datasets_path = "E:/superai2-work/section-preprocess/datasets/csv-files/"

# ========================================================================================

# read text file TH  EN
th = open(dataset_th_dir, 'r', encoding="utf-8")
th_text = th.read()
#th_text = th_text.replace(" ","")
th_text = th_text.split("\n")
print(type(th_text))
print(th_text)

# read text file EN
en = open(dataset_en_dir, 'r', encoding="utf-8")
en_text = en.read()
#en_text = en_text.replace("�","'") # We're I've
en_text = en_text.split("\n")
print(type(en_text))
print(en_text)

# ========================================================================================

# create column
col_th = ["raw_th_text"]
df_th = pd.DataFrame(th_text, columns=col_th)

col_en = ["raw_en_text"]
df_en = pd.DataFrame(en_text, columns=col_en)

# import csv file
df = pd.concat([df_th, df_en], axis=1).reindex(df_th.index)
df.to_csv(datasets_path + "test.csv", index=True, encoding="utf-8-sig")
print(df)

# ========================================================================================v






