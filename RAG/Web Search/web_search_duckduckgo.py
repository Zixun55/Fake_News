### langchain的 ###
# from langchain_community.tools import DuckDuckGoSearchRun

# # 建立 DuckDuckGo 搜尋工具
# search_tool = DuckDuckGoSearchRun()

# # 進行搜尋
# query = "What is Kelly Ayotte's stance on abortion and reproductive rights?"
# search_results = search_tool.run(query)

# print(search_results)



from duckduckgo_search import DDGS

def search_duckduckgo(query, num_results=5):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return results

query = "Kelly Ayotte 、abortion 、Planned Parenthood 、Neil Gorsuch 、Roe v. Wade"
search_results = search_duckduckgo(query)

for result in search_results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['href']}")
    print(f"Snippet: {result['body']}\n")