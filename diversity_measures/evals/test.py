import answer_extraction as ae

test = "No, the answer is yes."

word = ae.extract_stqa(test)

print(word)