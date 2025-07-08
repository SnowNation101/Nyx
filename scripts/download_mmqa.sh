mkdir -p downloads/MMQA/
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_train.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_dev.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_test.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_texts.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_tables.jsonl.gz
wget -P downloads/MMQA https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_images.jsonl.gz
wget -P downloads/MMQA https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip
gunzip downloads/MMQA/*.gz
unzip downloads/MMQA/*.zip -d downloads/MMQA/