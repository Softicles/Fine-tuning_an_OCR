import re

def split_file(input_path="dataset/data.txt", train_path="dataset/train.txt", test_path="dataset/test.txt", test_ratio=0.2, seed=42):
    
    lines = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # Clean ChatGPT output
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # skip empty lines
            
            # splitting on tab
            if '\t' in line:
                parts = line.split('\t', 1)
            else:
                # Fallback: split on first whitespace (space or multiple)
                parts = re.split(r'\s+', line, maxsplit=1)

            if len(parts) != 2:
                print(f"[Warning] Line {i} is malformed and skipped: {line}")
                continue
            image_path, label = parts
            lines.append(f"{image_path}\t{label}\n")
            
    # Compute split index
    split_idx = int(len(lines) * (1 - test_ratio))

    # 
    with open(train_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(lines[:split_idx])

    with open(test_path, 'w', encoding='utf-8') as test_file:
        test_file.writelines(lines[split_idx:])

    print(f"Split complete: {len(lines[:split_idx])} lines to train.txt, {len(lines[split_idx:])} lines to test.txt")

if __name__ == "__main__":
    split_file()