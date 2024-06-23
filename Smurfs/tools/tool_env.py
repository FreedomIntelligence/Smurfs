import requests
from langchain import Wikipedia
from langchain.docstore.base import Docstore
from langchain.agents.react.base import DocstoreExplorer
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

tool_env = {
    "BingSearch": HPEnv.BingSearch,
    "Retrieve": HPEnv.Retrieve,
    "Lookup": HPEnv.Lookup
    }

# responses = HPEnv.BingSearch("Golden State Warriors")
# responses = HPEnv.Lookup("Stephen Curry")
# print(responses)
