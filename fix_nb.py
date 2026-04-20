import json

def fix_notebook():
    file_path = 'SignLanguageRecognition.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # 1. Fix missing sklearn import
                if "from sklearn.metrics import confusion_matrix" in line:
                    if "accuracy_score" not in line:
                        line = line.replace("confusion_matrix", "confusion_matrix, accuracy_score")
                
                # 2. Fix Krish Train Directory
                if r"C:\Users\Krish\Desktop\BU\Sign Language Recoginastion\Dataset\asl_alphabet_train\asl_alphabet_train" in line:
                    line = line.replace(
                        r"C:\Users\Krish\Desktop\BU\Sign Language Recoginastion\Dataset\asl_alphabet_train\asl_alphabet_train",
                        r"Z:\Shikhar\ASL\Sign-Language-Recognition\asl_alphabet_train\asl_alphabet_train"
                    )
                
                # 3. Fix Krish Test Directory
                if r"C:\Users\Krish\Desktop\BU\Sign Language Recoginastion\Dataset\asl_alphabet_test\test" in line:
                    line = line.replace(
                        r"C:\Users\Krish\Desktop\BU\Sign Language Recoginastion\Dataset\asl_alphabet_test\test",
                        r"Z:\Shikhar\ASL\Sign-Language-Recognition\asl_alphabet_test\asl_alphabet_test"
                    )
                
                new_source.append(line)
            cell['source'] = new_source

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("Notebook perfectly patched!")

if __name__ == "__main__":
    fix_notebook()
