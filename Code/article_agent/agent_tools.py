import time

from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import WikipediaRetriever

from typing import Dict, Any

# Initialize tools
wikipedia_retriever = WikipediaRetriever()
ddg_search = DuckDuckGoSearchRun()

@tool
def search_google(searchstring: str) -> Dict[str, Any]:
    """
    Tool to search for recent, specific and relevant information on Google.

    Args:
        searchstring (str): The string to search for.

    Returns:
        dict: A dictionary containing the status of the operation and the content or error message.
    """
    query = f"{searchstring}?"
    for attempt in range(2):  # Try twice: initial attempt + 1 retry
        try:
            # Add a time delay to avoid getting blocked by rate limiting
            time.sleep(1)
            result = ddg_search.invoke(query)
            # i sometimes get parsing errors, remove \ from the result
            result = str(result.replace("\\", ""))
            return {
                "status": "success",
                "content": result
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 0:  # First failure
                time.sleep(5)
    return {
        "status": "error",
        "content": "DuckDuckGo search failed after multiple attempts."
    }

@tool
def search_wikipedia(searchstring: str) -> Dict[str, Any]:
    """
    Tool to retrieve general information from Wikipedia.

    Args:
        searchstring (str): The string to search for.

    Returns:
        dict: A JSON object with the retrieved content.
    """
    try:
        docs = wikipedia_retriever.invoke(searchstring)
        content = "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant Wikipedia results."
        truncated_content = content[:5000] + "..." if len(content) > 1000 else content
        return {"status": "success", "content": truncated_content}
    except Exception as e:
        print(f"Wikipedia retrieval failed: {e}")
        return {"status": "error", "content": "Error retrieving Wikipedia results."}

    

if __name__ == "__main__":
    print(search_google.invoke({"searchstring": "BMW Carbio"}))
    # print(search_wikipedia.invoke({"brand": "BMW", "type": "SUV"}))
