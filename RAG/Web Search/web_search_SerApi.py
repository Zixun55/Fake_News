from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper

# 設定 SerpAPI Wrapper

query = "What are the details of the 2018 National Defense Authorization Act signed by President Trump on Tuesday?"
search_results = search.results(query) 

# print(search_results)

def print_search_results(search_results):
    # 確保 organic_results 存在
    if "organic_results" in search_results and isinstance(search_results["organic_results"], list):
        for result in search_results["organic_results"]:
            if isinstance(result, dict):  # 確保 result 是字典
                print(f"Title: {result.get('title', 'N/A')}")
                print(f"URL: {result.get('link', 'N/A')}")
                print(f"Snippet: {result.get('snippet', 'N/A')}\n")
            else:
                print(f"Unexpected data format in organic_results: {result}")
    else:
        print("No organic_results found in search_results")


print_search_results(search_results)