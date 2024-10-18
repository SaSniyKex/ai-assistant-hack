import os
from typing import Optional
import pandas as pd
import requests
from typing import Callable
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import abc
from typing import Optional
import io
import multiprocessing
from torch.nn import CosineSimilarity
import re
import time
from unittest.mock import patch
import os
from typing import Optional

import requests

import abc
from typing import Optional

# GLOBAL VARS
api_key = ''
folder_id = ''
current_file_path = os.path.abspath(__file__)

current_file_path = os.path.dirname(current_file_path)
path_to_train = os.path.join(current_file_path, 'rag_database', 'data.xlsx')
yandex_gpt = None
model_assistant_role = 'assistant'
dim_retriever = 1024
embedding_model = SentenceTransformer("deepvk/USER-bge-m3")
index_train = faiss.IndexFlatL2(dim_retriever)
right_bd_train = pd.read_excel(path_to_train)
right_bd_train = right_bd_train[right_bd_train.task_id != 43].author_comment.tolist()
train_dt = right_bd_train


class BaseModel(abc.ABC):
    """Abstract class for all models."""

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self.messages = []
        self.system_prompt = system_prompt
        pass

    @abc.abstractmethod
    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        """Send a message to the assistant and return the assistant's response."""
        pass


class YandexGPT(BaseModel):
    """See more on https://yandex.cloud/en-ru/docs/foundation-models/concepts/yandexgpt/models"""

    model_urls = {
        "lite": "gpt://{}/yandexgpt-lite/latest",
        "pro": "gpt://{}/yandexgpt/latest",
        "custom": "ds://bt18l49vnfdjl7m0od7g"
    }

    def __init__(
            self,
            token: str,
            folder_id: str,
            model_name: str = "custom",
            system_prompt: Optional[str] = None,
            temperature: float = 0.1,
            max_tokens: int = 4000,
    ) -> None:
        super().__init__(system_prompt)
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "x-folder-id": folder_id,
        }
        self.model_url = YandexGPT.model_urls[model_name].format(folder_id)
        self.completion_options = {
            "stream": False,
            "temperature": temperature,
            "maxTokens": str(max_tokens),
        }

    def clear_hist(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "text": self.system_prompt})

    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        if clear_history:
            self.messages = []
            if self.system_prompt:
                self.messages.append({"role": "system", "text": self.system_prompt})

        self.messages.append({"role": "user", "text": user_message})

        json_request = {
            "modelUri": self.model_url,
            "completionOptions": self.completion_options,
            "messages": self.messages,
        }

        response = requests.post(self.api_url, headers=self.headers, json=json_request)
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return None

        response_data = response.json()
        assistant_message = response_data["result"]["alternatives"][0]["message"]["text"]
        self.messages.append({"role": "assistant", "text": assistant_message})
        return assistant_message

    def gen(self, chat):
        json_request = {
            "modelUri": self.model_url,
            "completionOptions": self.completion_options,
            "messages": chat,
        }

        response = requests.post(self.api_url, headers=self.headers, json=json_request)
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return None

        response_data = response.json()
        assistant_message = response_data["result"]["alternatives"][0]["message"]["text"]
        return assistant_message


def get_embs(texts, model_emb):
    return model_emb.encode(texts)


def add_messages_to_index(index, messages, model_emb):
    '''

    add embeddings of messages to your DB

    '''
    embeddings = get_embs(messages, model_emb)
    index.add(embeddings)


def find_top_k_similar_messages(index, query, model_emb, k=6):
    '''

    Find the top k similar messages to the query

    '''

    query_embedding = get_embs(query, model_emb)
    distances, indices = index.search(np.array([query_embedding]), k)
    print(indices)
    return indices[0]


def parse_indexes(texts, target, embedding_model, index):
    temp = find_top_k_similar_messages(index, target, embedding_model)
    return [texts[i] for i in temp]

def proccess(
        llm,
        tasks,
       tests,
        sols,
        student_solution, problem, author_solution, id_task,
        index_train,
        right_bd_train,
        api_key,
        folder_id):
    logs = pipeline_answer(
        None,
        tasks,
       tests,
        sols,
        student_solution, problem, author_solution, id_task,
        index_train,
        right_bd_train,
        api_key=api_key,
        folder_id=folder_id
    )
    return logs[-1]

def run_code(code, test_input, output_queue):
    """
    Executes the given code with the provided test_input.
    Captures the print output and sends it to the output_queue.
    """
    try:
        output_buffer = io.StringIO()
        local_namespace = {}
        with patch('builtins.input', return_value=test_input):
            def mock_print(*args, **kwargs):
                print(*args, **kwargs, file=output_buffer)

            local_namespace['print'] = mock_print
            exec(code, local_namespace)
        output = output_buffer.getvalue().strip()
        output_queue.put(output)
    except Exception as e:
        output_queue.put(f"Error: {e}")


def execute_code_with_tests(code: str, test_cases: list, timeout: int = 3):
    """
    Executes the provided code against a list of test cases.
    Each test case is a tuple: (test_input, expected_output, test_type)
    Returns a list of results indicating pass/fail for each test.
    """
    results = []
    real_output_student = []
    for i, (test_input, expected_output, test_type) in enumerate(test_cases):
        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_code, args=(code, test_input, output_queue))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            results.append(f"Test {i + 1}: Failed_{test_type}_ (Timeout after {timeout} seconds)")
            real_output_student.append(None)
            continue
        try:
            output = output_queue.get_nowait().strip()
        except multiprocessing.queues.Empty:
            output = ""
        if output == expected_output:
            real_output_student.append(None)
            results.append(f"Test {i + 1}: Passed_{test_type}_")
        else:
            real_output_student.append(output)
            results.append(f"Test {i + 1}: Failed_{test_type}_ (Expected: {expected_output}, Got: {output})")

    return results, real_output_student


def create_tests(task_id):
    test = []
    initial = tests[tests['task_id'] == task_id]
    for i in range(len(initial)):
        test.append((initial.iloc[i]['input'], initial.iloc[i]['output'], initial.iloc[i]['type']))
    return test


def create_test_prompt(test_res):
    test_res = test_res[0]  # т.к я c execute_code_with_tests возвращаю два списка, где первый здесь релевантный
    flag_passed_open = True
    flag_passed_closed = True
    for i in test_res:
        types = i.split('_')[1]
        if types == 'open' and 'Failed' in i:
            flag_passed_open = False
        elif types == 'closed' and 'Failed' in i:
            flag_passed_closed = False
    txt = []
    if not flag_passed_closed and not flag_passed_open:
        return "Ошибка в открытых и скрытых тестах. "
    elif flag_passed_closed and not flag_passed_open:
        return "Ошибка в открытых тестах. "
    elif not flag_passed_closed and flag_passed_open:
        return "Ошибка в закрытых тестах. "
    elif flag_passed_closed and flag_passed_open:
        return ""


def check_syntax(code_str):
    try:
        compile(code_str, 'my_code.py', 'exec')
        return None
    except SyntaxError as se:
        return f"Синтаксическая ошибка:\n {se.msg}\n  Отступ: {se.offset}\n  Текст: {se.text}"
    except Exception as e:
        return str(e)


def mistake_syntex(yagpt, student_solution, error_describe, train_dt, embedding_model, index_bd, tags, api_key,
                   folder_id):
    yandex_gpt_testing = YandexGPT(
        token=api_key,
        folder_id=folder_id,
        model_name='pro')
    prompt_gen = ''' Тебе дан код студента и ошибка найденная в коде. ВСЕГДА ПИШИ обычным текстом. НЕ ИСПОЛЬЗУЙ markdown и СИМВОЛ звездочка. Пиши коротко. Пиши строго по делу. Не говори как исправить, не пиши код.\n''' \
                 '''Код студента:\n {student_solution}\n\nОшибка в коде:\n{error_describe}\n'''

    def get_answer_spec(yagpt, chat_dialog=None, generation_config=None, forget=True):
        return yagpt.ask(chat_dialog, True)

    prompt_gen = prompt_gen.format(student_solution=student_solution, error_describe=error_describe)

    prompt_gen += '''Тебе даны примеры КАК надо выводить ошибку:\n'''

    formats = 'Пример {i}: {msg}'
    # print(index_bd.ntotal)
    relevs = parse_indexes(train_dt, tags, embedding_model, index_bd)
    for i in range(min(2, len(relevs))):
        res = formats.format(i=i, msg=relevs[i])
        prompt_gen += res + '\n'
    # print(tag_prompt)
    print(prompt_gen)
    logic_disccusion = get_answer_spec(yandex_gpt_testing, prompt_gen, True)
    return logic_disccusion


def remove_comments(code):
    code = re.sub(r"'''(.*?)'''", '', code, flags=re.DOTALL)
    code = re.sub(r'"""(.*?)"""', '', code, flags=re.DOTALL)
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\n\s*\n', '\n', code)
    code = code.strip()

    return code


def generation_ones(llm, tests, sols, tasks, id_task, student_solution):
    prompt_gen = '''Тебе дано условие задачи, эталон решение(правильное) и код решения студента(его нужно проверить).''' \
                 '''порассуждай, найдя ошибку. Объясни используя только текст, без примеров кода.''' \
                 '''Тебе даны тесты на которых валится код студента.'''
    '''не предоставляй исправленный код. НЕ ПРЕДОСТАВЛЯЙ НИКАКОЙ КОД.''' \
    '''Условие задачи:{problem}\n\nЭталон решения:{author_solution}\n\nКод студента:{student_solution}'''
    prompt_gen = prompt_gen.format(problem=problem, author_solution=author_solution, student_solution=student_solution)
    make_tests = create_tests(id_task)
    _, execs = execute_code_with_tests(student_solution, make_tests)
    format_test_string = '''Тест {i}: Вход: {inp};;Правильный ответ:{target};;Ответ программы студента:{output};;Тип теста:{type_test}'''
    for i in range(len(make_tests)):
        if execs[i] == None:
            continue
        output = execs[i]
        inp, target, type_test = make_tests[i]
        test_curr = format_test_string.format(i=i, inp=inp, target=target, output=output, type_test=type_test)
        prompt_gen += test_curr + '\n'
    chat_dialog = [{
        "role": "system",
        "content": ""
    }, {
        "role": "user",
        "content": prompt_gen
    }]
    logic_disccusion = get_answer(llm, prompt_gen, False)
    chat_dialog.append(
        {
            "role": model_assistant_role,
            "content": logic_disccusion
        }
    )
    return chat_dialog, logic_disccusion


def generation_3(llm, chat_dialog, problem, author_solution, student_solution, tags, index_bd, train_dt, target):
    tag_prompt = '''Тебе даны примеры КАК надо выводить ошибку.''' \
                 '''Исходя из некого шаблона и учитывая все твои прошлые рассуждения напиши ошибку кода студента. ''' \
                 '''Если на каком-то тесте валится и НЕТ СИНТАКСИЧЕСКОЙ ОШИБКИ пиши в начале предлжение: ''' \
                 '''Ошибка в открых или скрытых тестах (или вместе, учитывай на каких тестах завалился код студента). ''' \
                 '''ЕСли синтаксическая ошибка: пиши в начале Синтаксическая ошибка.\n'''
    formats = 'Пример {i}: {msg}'
    # print(index_bd.ntotal)
    relevs = parse_indexes(train_dt, tags, embedding_model, index_bd)
    try:
        relevs.remove(target)
    except(Exception) as ex:
        print(relevs)
    for i in range(len(relevs)):
        res = formats.format(i=i, msg=relevs[i])
        tag_prompt += res + '\n'
    # print(tag_prompt)
    chat_dialog.append(
        {
            "role": "user",
            "content": tag_prompt
        }
    )

    logic_disccusion = get_answer(llm, tag_prompt, False)
    chat_dialog.append(
        {
            "role": model_assistant_role,
            "content": logic_disccusion
        }
    )
    return chat_dialog, logic_disccusion


def generation_tags(llm, problem, author_solution, student_solution):
    '''

    ЗДЕСЬ ГЕНЕРИМ ТЕГИ ЧТОБЫ ПОЛУЧИТЬ ИЗ ТРЕИНА НУЖНЫЕ ФОРМАТЫ ПО СМЫСЛУ

    '''
    # format_gen = '''Тебе дано условие задачи, эталон решение(правильное) и код решения студента(его нужно проверить). Тебе известно, что в коде логическая ошибка'''\
    #     '''которая влияет на неправильный вывод на тестах в соответствии в условием, порассуждай, найдя ошибку. Выдели теги(типы логических ошибок, перечисли связанные с этой ошибкой через запятую кратко)'''\
    #     '''не предоставляй исправленный код. НЕ ПРЕДОСТАВЛЯЙ НИКАКОЙ КОД.'''\
    #     '''Условие задачи:{problem}\n\nЭталон решения:{author_solution}\n\nКод студента:{student_solution}'''
    format_gen = '''Тебе дано условие задачи, эталон решение(правильное) и код решения студента(его нужно проверить). Тебе известно, что в коде ошибка''' \
                 '''которая влияет на неправильный вывод на тестах в соответствии c условием. Выдели теги(типы логических ошибок, перечисли связанные с этой ошибкой через запятую кратко)''' \
                 '''не предоставляй исправленный код. НЕ ПРЕДОСТАВЛЯЙ НИКАКОЙ КОД. Просто выдели типы логических ошибок связанных''' \
                 '''Условие задачи:{problem}\n\nЭталон решения:{author_solution}\n\nКод студента:{student_solution}'''
    input_format = format_gen.format(problem=problem, author_solution=author_solution,
                                     student_solution=student_solution)
    chat_dialog = [{
        "role": "system",
        "content": ""
    }, {
        "role": "user",
        "content": input_format
    }]
    # Tags = get_answer(llm, chat_dialog)
    Tags = yandex_gpt.ask(input_format, False)
    return [], Tags


def pipeline_answer(llm, tasks, tests, sols, student_solution, problem, author_solution, id_task, index_bd, train_dt,
                    api_key, folder_id, tests_passed=None, syntax_error=None, target=None):
    yandex_gpt.clear_hist()
    student_solution = remove_comments(student_solution)
    error_describe = check_syntax(student_solution)
    chat_dialog, prepared_tags = generation_tags(llm, problem, author_solution, student_solution)
    log_answers = []
    ans = None
    if error_describe != None:
        print(error_describe)
        ans = mistake_syntex(None, student_solution, error_describe, train_dt, embedding_model, index_bd, prepared_tags,
                             api_key=api_key, folder_id=folder_id)
    else:
        chat_dialog, ans = generation_3(llm, chat_dialog, problem, author_solution, student_solution, prepared_tags,
                                        index_bd, train_dt, target)
    # print(prepared_tags)
    # print('-----------------------------------------')
    # chat_dialog, ans = generation_1(llm, problem, author_solution, student_solution)
    # print(chat_dialog)
    # log_answers.append(ans)
    # print(ans)
    # print('-----------------------------------------')
    # chat_dialog, ans = generation_2(llm, chat_dialog, tests, sols, tasks, id_task, student_solution)
    # print(chat_dialog)
    #     chat_dialog, ans = generation_ones(llm, tests, sols, tasks, id_task, student_solution)

    #     log_answers.append(ans)
    #     print(ans)
    #     print('-----------------------------------------')
    # print(chat_dialog)
    if ans != None and '```' in ans:
        ans = re.sub(r'```[^`]*```', '', ans) + '.'
    if ans != None:
        ans = ans.replace('*', '')
    log_answers.append(ans)
    print(ans)
    print('-----------------------------------------')
    return log_answers



def get_answer(yagpt, chat_dialog=None, generation_config=None, forget=None):
    if forget:
        return yandex_gpt.ask(chat_dialog, True)
    else:
        return yandex_gpt.ask(chat_dialog, False)


def proccess(
        llm,
        tasks,
        tests,
        sols,
        student_solution, problem, author_solution, id_task,
        index_train,
        right_bd_train,
        api_key,
        folder_id):
    logs = pipeline_answer(
        None,
        tasks,
        tests,
        sols,
        student_solution, problem, author_solution, id_task,
        index_train,
        right_bd_train,
        api_key=api_key,
        folder_id=folder_id
    )
    return logs[-1]


def main(cat_path, api_key1, folder_id1, tasks_name='tasks.xlsx', tests_name='tests.xlsx', sols_name='for_test.xlsx'):
    add_messages_to_index(index_train, train_dt, embedding_model)
    global yandex_gpt
    yandex_gpt = YandexGPT(
        token=api_key1,
        folder_id=folder_id1,
        model_name='pro')
    api_key = api_key1
    folder_id = folder_id1
    tasks = pd.read_excel(os.path.join(current_file_path, cat_path, tasks_name))
    tests = pd.read_excel(os.path.join(current_file_path, cat_path, tests_name))
    global sols_path
    sols_path = os.path.join(current_file_path, cat_path, sols_name)
    sols = pd.read_excel(os.path.join(current_file_path, cat_path, sols_name))
    ans = []
    for i in range(len(sols)):
        print(len(sols))
        id_student = sols.loc[i, 'id']
        print(sols[sols.id == id_student])
        print(sols[sols.id == id_student].task_id.values[0])
        print(tasks[tasks.id == sols[sols.id == id_student].task_id.values[0]])
        id_task = tasks[tasks.id == sols[sols.id == id_student].task_id.values[0]].id.values[0]
        student_solution = sols[sols.id == id_student].student_solution.values[0]
        problem = tasks[tasks.id == id_task].description.values[0]
        author_solution = tasks[tasks.id == id_task].author_solution.values[0]
        target = sols[sols.id == id_student].author_comment.values[0]
        while True:
            time.sleep(1)
            answer = proccess(
                None,
                tasks,
                tests,
                sols,
                student_solution, problem, author_solution, id_task,
                index_train,
                right_bd_train,
                api_key=api_key,
                folder_id=folder_id
            )
            if answer != None:
                ans.append(answer)
                break
            else:
                time.sleep(1)
    calc_metric(ans)
    print("Метрика сохранена в submission.csv")
        
        
import pandas as pd
from torch.nn.functional import cosine_similarity


def _get_cosine_similarity(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> float:
    predictions = pred_df["author_comment_embedding"]
    true_values = true_df["author_comment_embedding"]
    total_cos_sim = 0

    for idx in range(len(true_values)):
        pred_value = string2embedding(predictions.iloc[idx])
        gt_value = string2embedding(true_values.iloc[idx])

        if len(pred_value) != len(gt_value):
            raise ValueError(f"Embeddings have different sizes: {len(pred_value)} != {len(gt_value)}")

        cos_sim_value = cosine_similarity(pred_value.unsqueeze(0), gt_value.unsqueeze(0))
        total_cos_sim += cos_sim_value
    return float(total_cos_sim / len(true_df))


def calculate_score(submit_path: str, gt_path: str) -> float:
    submit_df = pd.read_csv(submit_path)
    true_df = pd.read_excel(gt_path)
    submit_df = submit_df[submit_df["solution_id"].isin(true_df["id"])]
    return (_get_cosine_similarity(submit_df, true_df) - 0.6) / 0.4


def calculate_score_and_save(submit_path: str, gt_path: str, save_path: str) -> float:
    score = calculate_score(submit_path, gt_path)
    with open(save_path, "w") as f:
        f.write(f"{score}")
    return score

def calc_metric(ans):
    generate_submit(
            test_solutions_path=sols_path,
            preds=ans,
            save_path="submission.csv",
            use_tqdm=True,
        )

print("Loading models...", end="")
model_name = "DeepPavlov/rubert-base-cased-sentence"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
print("OK")


def get_sentence_embedding(sentence: str) -> torch.Tensor:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding


def string2embedding(string: str) -> torch.Tensor:
    return torch.Tensor([float(i) for i in string.split()])


def embedding2string(embedding: torch.Tensor) -> str:
    return " ".join([str(i) for i in embedding.tolist()])


def generate_submit(test_solutions_path: str, preds: list, save_path: str, use_tqdm: bool = True) -> None:
    test_solutions = pd.read_excel(test_solutions_path)
    bar = range(len(test_solutions))
    if use_tqdm:
        import tqdm

        bar = tqdm.tqdm(bar, desc="Predicting")

    submit_df = pd.DataFrame(columns=["solution_id", "author_comment", "author_comment_embedding"])
    print(len(preds))
    print(len(test_solutions))
    ccc = 0
    for i in bar:
        idx = test_solutions.index[i]
        solution_row = test_solutions.iloc[i]

        text = preds[ccc]

        embedding = embedding2string(get_sentence_embedding(text))
        submit_df.loc[i] = [idx, text, embedding]
        ccc += 1
    submit_df['solution_id'] = test_solutions['id']
    submit_df.to_csv(save_path, index=False)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some files.')

    parser.add_argument('cat_path', type=str, help='Path to the category')
    parser.add_argument('api_key1', type=str, help='API key')
    parser.add_argument('folder_id1', type=str, help='Folder ID')
    parser.add_argument('--tasks_name', type=str, default='tasks.xlsx',
                        help='Name of the tasks file (default: tasks.xlsx)')
    parser.add_argument('--tests_name', type=str, default='tests.xlsx',
                        help='Name of the tests file (default: tests.xlsx)')
    parser.add_argument('--sols_name', type=str, default='solutions.xlsx',
                        help='Name of the solutions file (default: solutions.xlsx)')

    args = parser.parse_args()

    main(args.cat_path, args.api_key1, args.folder_id1, args.tasks_name, args.tests_name, args.sols_name)
