import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')


def gpt3_embedding(content, engine='text-similarity-ada-001'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)



def search_index(text, data, count=20):
    vector = gpt3_embedding(text)
    scores = list()
    for i in data:
        score = similarity(vector, i['vector'])
        #print(score)
        scores.append({'content': i['content'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[:count]


def get_detailed_answers(query, passages):
    max_answers = 5
    answers = []
    used_passages = []
    while len(answers) < max_answers and len(passages) > 0:
        result = passages[0]
        if result['content'] not in used_passages:
            used_passages.append(result['content'])
            prompt = open_file('prompt_answer.txt').replace('<<PASSAGE>>', result['content']).replace('<<QUERY>>', query)
            answer = gpt3_completion(prompt)
            print('\n\n','DETAILED ANSWER:','\n', answer)
            answers.append(answer)
        passages = passages[1:]
    return answers



def gpt3_completion(prompt, engine='text-davinci-002', temp=0.6, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

if __name__ == '__main__':
    with open('index.json', 'r') as infile:
        data = json.load(infile)

    # Ask for user input
    while True:
        query = input("Enter your question here: ")

        # Search for matching passages
        results = search_index(query, data)

        # Print the number of passages found
        print(f"Found {len(results)} passages for your query.")

        # Combine passages into chunks of 3
        chunks = [results[i:i+3] for i in range(0, len(results), 3)]

        # Ask GPT-3 for detailed answers
        answers = []
        found = False
        for chunk in chunks:
            if found:
                break
            for result in chunk:
                prompt = open_file('prompt_answer.txt').replace('<<PASSAGE>>', result['content']).replace('<<QUERY>>', query)
                answer = gpt3_completion(prompt)
                print('\n\n','DETAILED ANSWER:','\n', answer)
                answers.append(answer)
                if len(answers) >= 3:
                    found = True
                    break

        # Summarize the detailed answers
        all_answers = '\n\n'.join(answers)
        prompt = open_file('prompt_summary.txt').replace('<<SUMMARY>>', all_answers)
        summary = gpt3_completion(prompt)
        print('\n\n=========\n','\n','SUMMARIZING THE ANSWERS:','\n', summary,'\n\n')
