import json
import re

def fix_notebook():
    file_path = 'SignLanguageRecognition.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # The Keras 3 input_shape parameter on Conv2D causes lazy-building,
                # which hides variables from the optimizer during compile()
                if 'input_shape=(IMG_SIZE, IMG_SIZE, 3)' in line and 'Conv2D' in line:
                    indent_str = " " * (len(line) - len(line.lstrip()))
                    # Strip the input_shape argument from the Conv2D layer
                    clean_conv2d = line.replace(", input_shape=(IMG_SIZE, IMG_SIZE, 3)", "")
                    clean_conv2d = clean_conv2d.replace("input_shape=(IMG_SIZE, IMG_SIZE, 3)", "")
                    
                    # Manually inject Keras 3 Input Object
                    input_layer = indent_str + "layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),\n"
                    new_line = input_layer + clean_conv2d
                    new_source.append(new_line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("Keras 3 syntax successfully patched!")

if __name__ == "__main__":
    fix_notebook()
