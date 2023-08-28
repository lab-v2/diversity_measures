import os
import time
import queue
import json
import asyncio
from dotenv import load_dotenv

import openai
from langchain.llms import OpenAI

load_dotenv()
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")


class ResponseCollector:
    """
    This class provides functions which questions to GPT-3.5 and then store the response to disk.
    This allows users to aggregate a large number of responses from ChatGPT.
    """

    def __init__(
        self,
        configs: dict = {},
        rate_limit: int = 5,
        timeout: int = 100,
        verbose: bool = False,
        retry_time: int = 60,
        system_text: str = "",
    ):
        """
        `configs` a dict containing parameters which will be part of the payload such as model, temperature, etc
        `rate_limit` the maximum number of questions you would like to ask a minute.
        `api_key` OpenAI API key
        `timeout` the amount of time to wait before timing out
        `out_path` the path in which the responses will be outputted
        """
        self.configs = configs
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.verbose = verbose
        self.retry_time = retry_time
        self.system_text = system_text

    def get_asked_ids(self):
        asked_ids = set()

        # Checks for questions that have already been asked before
        if os.path.isfile(self.out_path):
            outfile = open(self.out_path)
            json_list = list(outfile)

            for row in json_list:
                try:
                    row = json.loads(row)

                    if "question_id" in row:
                        asked_ids.add(str(row["question_id"]))
                except:
                    pass
            outfile.close()
        return asked_ids

    def start(self, question_list, out_path):
        """
        `question_list` list of questions to ask ChatGPT
        """
        self.out_path = out_path

        asyncio.run(self.__start(question_list))

    async def __start(self, question_list):
        self.succeed = 0
        self.goal = 0
        self.file_lock = asyncio.Lock()
        self.question_queue = queue.Queue()

        asked_ids = self.get_asked_ids()

        for question in question_list:
            if str(question["id"]) in asked_ids:
                continue
            self.goal += 1
            self.question_queue.put(question)

        while self.succeed < self.goal:
            try: self.async_ask(self.question_queue.get(False))
            except: pass
    def log(self, response):
        with open(self.out_path, "a") as outfile:
            outfile.write(json.dumps(response))
            outfile.write("\n")

        self.succeed += 1

    def async_ask(self, question):
        while True:
            try:
                question_index = question["id"]
                question_text = question["text"]

                llm = OpenAI(**self.configs)
                response = llm.generate(question_text)
                print(response)

                if self.verbose:
                    print(f"INFO | ID {question_index} | RECEIVED: {response}")

                self.log(
                    {
                        "question_id": question_index,
                        **response,
                        "question": question_text,
                        **self.configs,
                    }
                )

                return
            except:
                pass
