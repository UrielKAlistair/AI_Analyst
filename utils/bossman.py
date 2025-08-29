import os
from utils.llmproxy import LLMProxy
from utils.worker import Worker
import json


class Bossman:
    """
    This is an intelligent orchestrator. Given the user task, it generates a high level plan.
    It then generates detailed prompts for one step at a time and hands it to another Agent.
    This completes the task and reports back to bossman. It can also ask bossman for clarifications.
    Once a task is completed, bossman decides the next course of action, and loops until completion.
    """

    def __init__(self, user_prompt: str) -> None:
        self.llm = LLMProxy()
        self.worker = Worker()
        self.context = []
        self.top_level_plan = None
        self.user_prompt = user_prompt

    def run(self):
        if self.top_level_plan is None:
            prompt_file = os.path.join("prompts", "toplevel.txt")
            with open(prompt_file, "r") as f:
                prompt_template = f.read()
            response = self.llm([prompt_template, self.user_prompt])
            print(response)
            self.top_level_plan = response

        while True:
            task_prompt = self.next_step()
            # I think that we should highly bias the output of next step depending on the task with custom prompts
            llm_response = json.loads(task_prompt)
            print(llm_response)

            status = llm_response["status"]

            if status == "complete":
                return llm_response["final_result"]

            elif status == "plan_revision":
                self.top_level_plan = llm_response["new_plan"]
                self.context = (
                    []
                )  # Clean context for the new plan, but needs a more thought out context engineering plan

            elif status == "in_progress":
                output = self.worker(llm_response["next_action"])
                self.context.append(output)  # Needs better context engineering

    def next_step(self):
        prompt_file = os.path.join("prompts", "nextstep.txt")
        with open(prompt_file, "r") as f:
            prompt_template = f.read()
        prompt = prompt_template.format(
            top_level_plan=self.top_level_plan,
            context=self.context if self.context else "No actions have been taken yet.",
        )
        print(prompt)
        return self.llm(prompt)
