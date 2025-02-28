import os
import argparse
from openai import OpenAI

class PromptGenerator:
    # Preset configurations
    PRESET_CONFIGS = {
        "cloud": {
            "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("ALIYUN_BAILIAN_KEY"),
            "model": "deepseek-r1"
        },
        "ollama": {
            "api_base_url": "http://localhost:11434/v1/",
            "api_key": "ollama",
            "model": "deepseek-r1:8b"
        }
    }
    
    def __init__(self, preset="cloud"):
        self._preset = preset
        config = self.PRESET_CONFIGS[preset]
        self._api_base_url = config["api_base_url"]
        self._api_key = config["api_key"]
        self.model = config["model"]
        self.client = OpenAI(api_key=self._api_key, base_url=self._api_base_url)
        self._system_msg = "You are a helpful AI, and you are helping me to generate prompts for FLUX text-to-image model" \
            "Genrate prompts for the following given keywords:"

    @property
    def system_msg(self):
        return self._system_msg

    @system_msg.setter
    def system_msg(self, value):
        self._system_msg = value

    @property
    def preset(self):
        return self._preset

    @preset.setter
    def preset(self, value):
        if value not in self.PRESET_CONFIGS:
            raise ValueError(f"Invalid preset: {value}")
        self._preset = value
        config = self.PRESET_CONFIGS[value]
        self._api_base_url = config["api_base_url"]
        self._api_key = config["api_key"]
        self.model = config["model"]
        self.client = OpenAI(api_key=self._api_key, base_url=self._api_base_url)

    @property
    def api_base_url(self):
        return self._api_base_url

    @api_base_url.setter
    def api_base_url(self, value):
        self._api_base_url = value
        self.client = OpenAI(api_key=self._api_key, base_url=self._api_base_url)
    
    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value
        self.client = OpenAI(api_key=self._api_key, base_url=self._api_base_url)

    def generate(self, keywords, system_msg=None, model=None, streaming=True):
        """
        Generates a response from the model by combining a system message and keywords.

        Args:
            system_msg (str): The base system instruction or prompt.
            keywords (str): Additional terms or phrases to incorporate into the prompt.
            streaming (bool, optional): If True, yields responses incrementally. Defaults to True.

        Yields:
            tuple: In streaming mode, yields three elements:
                - The response object from the client
                - The accumulated reasoning content
                - The accumulated content

        Returns:
            tuple: In non-streaming mode, returns a tuple containing:
                - The full response object
                - The final reasoning content
                - The final content
        """
        reasoning_content = ""
        content = ""

        if system_msg is None:
            system_msg = self._system_msg
        if model is None:
            model = self.model

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": system_msg + " " + keywords
                }],
            stream=streaming
        )
        # FIXME: #10, the distilled ollama model response mixes the thinking process with the content
        # so we need to separate them before returning to app layer
        if streaming:
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    content += delta.content

                # the local distilled ollama model mixes the thinking process with the content
                # the thinking process is enclosed in <think> </think> tag
                if self._preset == "ollama" and self.model.startswith("deepseek-r1"):
                    # first few words
                    if len(content) < len("<think>"):
                        yield content, ""
                    # after the <think> tag
                    else:
                        assert content.startswith("<think>")
                        contents = content.split("</think>")
                        reasoning = contents[0][len("<think>"):]
                        if len(contents) > 1: # </think> already exists
                            yield reasoning, contents[1]
                        else: # </think> does not exist yet, all are reasoning content
                            # How to be sure that the final response contains the non-reasoning content?
                            yield reasoning, ""
                else:
                    yield reasoning_content, content
        else:
            reasoning_content = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content
        return reasoning_content, content

    def translate(self, text):
        """
        Translates the given text from Chinese to English.

        Args:
            text (str): The text to translate.

        Returns:
            str: The translated text.
        """
        translate_inst = "translate the following text to Chinese, output Chinese directly:"
        for reasoning, result in self.generate(text, system_msg=translate_inst, streaming=True):
            yield reasoning, result
    
    def extract(self, text):
        """
        Extracts the prompt from the given text.
        """
        if self._preset == "ollama":
            extract_inst = '''Given a paragraph of text-to-image prompts, '''
            '''you should extract English parts of the prompt from the following text and directly output it. ''' \
              '''ignore the thinking process and Chinese translation section if any.''' \
              '''ignore the keyword "Prompt" itself if presents and directly output the content of the prompt:\n'''
        else:
            extract_inst = '''extract English parts of the "Prompt" section from the following text and directly output it. ''' \
              '''ignore the thinking process and the Chinese translation section.''' \
              '''ignore the keyword "Prompt" itself and directly output the content of the prompt:\n'''
        for reasoning, result in self.generate(text, system_msg=extract_inst, streaming=True):
            yield reasoning, result
    

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using FLUX model')
    parser.add_argument('--keywords', type=str, default="diva, sexy, large breasts", help='Keywords for prompt generation')
    parser.add_argument('--streaming', type=bool, default=True, help='Enable streaming mode')
    return parser.parse_args()

def main():
    pg = PromptGenerator()
    args = parse_args()

    if not args.streaming:
        reasoning_content, content = pg.generate("diva, sexy, large breasts", streaming=False)
        print("====== Thinking Process:\n", reasoning_content)
        print("====== Result:\n", content)
    else: 
        for reasoning_content, content in pg.generate("diva, sexy, large breasts", streaming=True):
            print("====== Thinking Process:\n", reasoning_content)
            print("====== Result:\n", content)

if __name__ == "__main__":
    main()