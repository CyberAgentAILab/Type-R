import re


def extract_words_from_prompt(prompt):
    words = []
    matches = re.findall(r"'(.*?)'", prompt)  # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())

    matches = re.findall(r'"(.*?)"', prompt)  # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())
    return words
