import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """Can you fix the translations of an OCR based on the original PDF following these rules:

- Preserve cases (don't change lower-case to upper-case and vice versa)
- Preserve punctuations
- If you recognize a word being abbreviated (for example: dep - dependence, asg - assign), leave it intact  
- If there are many reasonable translations, choose the one that differs no more than 1 character to the original word
- If you are not sure, leave the word intact
- If you see two consecutive words may share the same meaning (responsibility, reponsi), only correct one of them and delete the other
- Process the entire texts
- Each output line should have the same format as the corresponding input line
- Don't include intro and outro

The topic is each crop image's name"""

def read_file(filepath = "output_images/rec_gt.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
    
def split_text_into_chunks(text, num_line_per_chunk = 100):
    lines = text.splitlines()
    chunks = [lines[i : i + num_line_per_chunk] for i in range(0, len(lines), num_line_per_chunk)]
    return chunks

def call_chatgpt(prompt):
    try:
        response = CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling ChatGPT: {e}")
        return None
    

def main():
    raw_text = read_file()
    raw_chunks = split_text_into_chunks(raw_text)
    
    corrected_chunks = []
    
    for i, raw_chunk in enumerate(raw_chunks):
        raw_chunk_text = '\n'.join(raw_chunk)
        prompt = PROMPT + "\n" + "\n" + raw_chunk_text
        
        print(f"Processing chunk {i+1}/{len(raw_chunks)}...")
        corrected = call_chatgpt(prompt)
        if corrected:
            corrected_chunks.append(corrected)
        else:
            corrected_chunks.append(raw_chunk_text)
            
    with open("dataset/data.txt", 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(corrected_chunks))

    print(f"Corrected file saved to dataset/data.txt")

if __name__ == "__main__":
    main()