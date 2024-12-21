import time

from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import WikipediaRetriever

# Initialize tools
wikipedia_retriever = WikipediaRetriever()
ddg_search = DuckDuckGoSearchRun()

@tool
def search_duckduckgo(searchstring: str) -> str:
    """
    Tool to search for information on DuckDuckGo based on brand and type.

    Args:
        brand (str): The brand to search for.
        type (str): The type to search for.

    Returns:
        str: The search result, or an error message if the search fails.
    """
    query = f"What's the new with {searchstring}?"
    for attempt in range(2):  # Try twice: initial attempt + 1 retry
        try:
            # Add a time delay to avoid getting blocked by rate limiting
            time.sleep(1)
            return ddg_search.invoke(query)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 0:  # First failure
                time.sleep(5)
    return "DuckDuckGo search failed after multiple attempts."

@tool
def fetch_wikipedia_context(searchstring: str) -> str:
    """
    Tool to retrieve information from Wikipedia.

    Args:
        brand (str): The brand to search for.

    Returns:
        str: The retrieved Wikipedia content.
    """
    try:
        docs = wikipedia_retriever.invoke(searchstring)
        return "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant Wikipedia results."
    except Exception as e:
        print(f"Wikipedia retrieval failed: {e}")
        return "Error retrieving Wikipedia results."
    

if __name__ == "__main__":
    # print(search_duckduckgo.invoke({"brand": "BMW", "type": "SUV"}))
    print(fetch_wikipedia_context.invoke({"brand": "BMW", "type": "SUV"}))
