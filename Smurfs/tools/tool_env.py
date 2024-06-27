import warnings

# 抑制所有警告
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Smurfs.tools.utils import python_repl, interpreter
import requests
from langchain import Wikipedia
from langchain.docstore.base import Docstore
from langchain.agents.react.base import DocstoreExplorer
import xmltodict

class HotpotToolEnv:
    def __init__(self):
        self.bingresults = None
        self.docstore = DocstoreExplorer(Wikipedia())
        self.last_step = ""
        self.bingkeyword = ""

    def _validate_server(self, address):
        if not address:
            raise ValueError('Must provide a valid server for search')
        if address.startswith('http://') or address.startswith('https://'):
            return address
        PROTOCOL = 'http://'
        print(f'No protocol provided, using "{PROTOCOL}"')
        return f'{PROTOCOL}{address}'

    def parse_bing_result(self, result):
        responses = []
        try:
            value = result["webPages"]["value"]
        except:
            return responses

        for i in range(len(value)):
            snippet = value[i]['snippet'] if 'snippet' in value[i] else ""
            snippet = snippet.replace("<b>", "").replace("</b>", "").strip()
            if snippet != "":
                responses.append(snippet)
            
        return responses

    def call_bing_search(self, query, count, endpoint="https://api.bing.microsoft.com/v7.0/search", bing_api_key=""):
        headers = {'Ocp-Apim-Subscription-Key': bing_api_key}
        params = {"q": query, "textDecorations": True,
                "textFormat": "HTML", "count": count, "mkt": "en-GB"}
        # print()
        try:
            server = self._validate_server(endpoint) # server address
            server_response = requests.get(server, headers=headers, params=params)
            resp_status = server_response.status_code
            print(server_response)
            if resp_status == 200:
                result = server_response.json()
                return self.parse_bing_result(result)
        except:
            pass
        
        return None

    def BingSearch(self, query):
        try:
            responses = self.call_bing_search(query=query, count=4)
            self.bingresults = responses
            self.last_step = "BingSearch"
            if len(responses) > 0 :
                pass
            else:
                responses = "BingSearch error"
        except :
            responses = 'BingSearch error, please try again.'
        return responses

    def Retrieve(self, entity):
        # docstore = DocstoreExplorer(Wikipedia())
        try:
            print(entity)
            responses = self.docstore.search(entity)
            self.last_step = "Retrieve"
        except Exception as e:
            print(e)
            responses = 'Could not find that page, please try again.'
        return responses

    def format_step(self, step):
        step = step.strip('\n').strip().replace('\n', '')
        if step.startswith("Thought") or step.startswith("Action"):
            step = step.split()[2:]
            step = " ".join(step)
        if "Thought" in step:
            step = step.split("Thought")[0].strip()
        if "Action" in step:
            step = step.split("Action")[0].strip()
        if "Observation" in step:
            step = step.split("Observation")[0].strip()
        return step

    def bing_lookup(self,term):
        if self.bingresults == None or len(self.bingresults) == 0 :
            raise ValueError("Cannot lookup without a successful bingsearch first")
        if term.lower() != self.bingkeyword:
            self.bingkeyword = term.lower()
            self.bingindex = 0
        else:
            self.bingindex += 1
        lookups = [r for r in self.bingresults if self.bingkeyword in r.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.bingindex >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.bingindex + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.bingindex]}"

    def Lookup(self, keyword):
        if self.last_step == "Retrieve":
            try:
                responses = self.docstore.lookup(keyword)
            except ValueError:
                responses = 'The last page Retrieved was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        elif self.last_step == "BingSearch":
            try:
                responses = self.bing_lookup(keyword)
            except ValueError:
                responses = 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        else:
            responses = "Lookup must after Retrive or BingSearch"
        # self.last_step = "Lookup"
        return responses

HPEnv = HotpotToolEnv()


#计算任务
#经常不print，功能最强大
def calculate_python(query):
    query = query.strip().strip("```")
    return python_repl.run(query)

#做计算任务不错，但是不会做比较任务
def code_interpreter(code):
    '''execute Python expressions with Python Interpreter, can be used as a simple calculator e.g., "(123 + 234) / 23 * 19"
        '''
    return interpreter.execute_code(code) 

def compare(candidate_list, compare_value, mode):
    #candidate_list: {name: , value: }
    results=[]
    if mode == "gt":
        for candidate in candidate_list:
            if candidate["value"] >= compare_value:
                results.append(candidate)
        return results
    if mode == "lt":
        for candidate in candidate_list:
            if candidate["value"] <= compare_value:
                results.append(candidate)
        return results
    if mode == "eq":
        for candidate in candidate_list:
            if candidate["value"] == compare_value:
                results.append(candidate)
        return results
    if mode == "max":
        for candidate in candidate_list:
            if results == []:
                results.append(candidate)
            else:
                if candidate["value"] == results[0]["value"]:
                    results.append(candidate)
                elif candidate["value"] > results[0]["value"]:
                    results = [candidate]
        return results
    if mode == "min":
        for candidate in candidate_list:
            if results == []:
                results.append(candidate)
            else:
                if candidate["value"] == results[0]["value"]:
                    results.append(candidate)
                elif candidate["value"] < results[0]["value"]:
                    results = [candidate]
        return results
    if mode == "nearest":
        for candidate in candidate_list:
            if results == []:
                results.append(candidate)
            else:
                can_dff = abs(candidate["value"]-compare_value)
                res_dff = abs(results[0]["value"]-compare_value)
                if can_dff == res_dff:
                    results.append(candidate)
                elif can_dff < res_dff:
                    results = [candidate]
        return results
    if mode == "farest":
        for candidate in candidate_list:
            if results == []:
                results.append(candidate)
            else:
                can_dff = abs(candidate["value"]-compare_value)
                res_dff = abs(results[0]["value"]-compare_value)
                if can_dff == res_dff:
                    results.append(candidate)
                elif can_dff > res_dff:
                    results = [candidate]
        return results

def getWolframAlphaResults(input:str, APPID = ""):
        """Get Wolfram|Alpha results using natural query. Queries to getWolframAlphaResults must ALWAYS have this structure: {\"input\": query}. And please directly read the output json. 
        """
        URL = "https://api.wolframalpha.com/v2/query"

        params = {'appid': APPID, "input": input, 'format': 'plaintext'}
        
        response = requests.get(URL, params=params)
        
        json_data = xmltodict.parse(response.text)
        
        if 'pod' not in json_data["queryresult"]:
            return "WolframAlpha API cannot parse the input query."
        #print(json_data)
        rets = json_data["queryresult"]['pod']

        cleaned_rets = []
        blacklist = ["@scanner", "@id", "@position", "@error", "@numsubpods", "@width", "@height", "@type", "@themes","@colorinvertable", "expressiontypes"]
        
        def filter_dict(d, blacklist):
            if isinstance(d, dict):
                return {k: filter_dict(v, blacklist) for k, v in d.items() if k not in blacklist}
            elif isinstance(d, list):
                return [filter_dict(i, blacklist) for i in d]
            else:
                return d

        for ret in rets:
            ret = filter_dict(ret, blacklist=blacklist)
            cleaned_rets.append(ret)
            # print(ret)
            # # Do further cleaning to retain only the input and result pods
            # if "@title" in ret:
            #     if ret["@title"] == "Input" or ret["@title"] == "Result":
            #         cleaned_rets.append(ret)

        return cleaned_rets

# result = getWolframAlphaResults("what is the factorial of 3?")
# print(result)

tool_env_math = {
    "calculate_python": calculate_python,
    "code_interpreter": code_interpreter,
    "getWolframAlphaResults": getWolframAlphaResults
}

tool_env_hotpot = {
    "BingSearch": HPEnv.BingSearch,
    "Retrieve": HPEnv.Retrieve,
    "Lookup": HPEnv.Lookup
    }

# result = getWolframAlphaResults('Calc integral of sin(x)+2x^2+3x+1 from 0 to 1')
# print(result)


# # responses = HPEnv.BingSearch("Golden State Warriors")
# # responses = HPEnv.Lookup("Stephen Curry")
# # print(responses)