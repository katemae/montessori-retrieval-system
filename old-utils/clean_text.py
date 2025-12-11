import re

def clean_text(raw_text):
    # norm line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # remove "-" at line breaks when words are split
    text = re.sub(r"-\n(?=[a-z])", "", text)

    # join paragraphs
    # --> replace single newlines with spaces,  keep double newlines.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


file_name = "ch10-montessori-science-behind-genius.txt"

with open(f"text-data/raw/{file_name}", "r", encoding="utf-8") as f:
    raw = f.read()

cleaned = clean_text(raw)
with open(f"text-data/cleaned_{file_name}", "w", encoding="utf-8") as f:
    f.write(cleaned)

print(f"saved to cleaned_{file_name}")
