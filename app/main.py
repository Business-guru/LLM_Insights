import logging
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_groq import ChatGroq

from prompts import law_inquiry_prompt,startup_inquiry_prompt,competitor_inquiry_prompt,books_inquiry_prompt, final_ans_prompt
from web_search import extract_urls_and_further_questions,extract_text_from_url
from eliminator import eliminator


logger = logging.getLogger(__name__)
load_dotenv()
groq_api_key = os.getenv("groq_api_key")


def fetch_all_data(urls, query, max_workers=10):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(extract_text_from_url, url, query): url for url in urls}
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {future_to_url[future]}: {e}")
                results.append(None)
    return results

# import json
# def fetch_all_data(urls, query, jsonl_cache_path="website_summaries.jsonl", max_workers=10):
#     # Check if cached file exists
#     if os.path.exists(jsonl_cache_path):
#         logger.info(f"Loading cached summaries from: {jsonl_cache_path}")
#         results = []
#         with open(jsonl_cache_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     results.append(json.loads(line.strip()))
#                 except json.JSONDecodeError:
#                     logger.error("Skipping bad line in cache.")
#         return results

#     logger.info("Processing websites using LLM (not cached)...")
#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_url = {executor.submit(extract_text_from_url, url, query): url for url in urls}

#         for future in as_completed(future_to_url):
#             try:
#                 url, summary = future.result()
#                 results.append({"url": url, "summary": summary})
#             except Exception as e:
#                 logger.error(f"Error processing {future_to_url[future]}: {e}")

#     # Save all results to JSONL
#     with open(jsonl_cache_path, 'w', encoding='utf-8') as f:
#         for result in results:
#             f.write(json.dumps(result, ensure_ascii=False) + "\n")

#     return results


def ask(query,domain):
    logger.info(f"starting now for query: {query}")
    revised_query = ''

    if domain.lower() == 'law':
        revised_query = law_inquiry_prompt(query)
    elif domain.lower() == 'startup_inquiry':
        revised_query = startup_inquiry_prompt(query)
    elif domain.lower() == 'market':
        revised_query = competitor_inquiry_prompt(query)
    elif domain.lower() == 'books':
        revised_query = books_inquiry_prompt(query)
    
    
    urls,further_questions = extract_urls_and_further_questions(revised_query)

    return urls,further_questions

    logger.info(f"url extraction done for query: {query}")

    website_data = fetch_all_data(urls, query) #[(url,content),(url,content)]

    logger.info("Website data length:",len(website_data))
    logger.info(f"Extraction of Data from Websites completed for query: {query}")

    refined_data = eliminator(revised_query,website_data)

    logger.info(f"Extracted top k data for query {query}")

    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name='llama-3.3-70b-versatile') #type: ignore

    response = llm.invoke(final_ans_prompt(revised_query,[data['summary'] for data in refined_data])).content

    response += "\nSOURCE URLS:\n"+"\n".join([data['url'] for data in refined_data])

    return response,further_questions




# Limitation of context window leads to additional step of trimming
# Max workers kept low and fewer entries passed per call due to local system processing limitations and api context window limitations


app = FastAPI()

class AskRequest(BaseModel):
    query: str
    domain: str

@app.post("/ask")
async def ask_api(req: Dict):
    try:
        response, further_questions = ask(req['query'], req['domain'])


        return JSONResponse(content={
            "response": response,
            "further_questions": further_questions
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# uvicorn main:app --host 0.0.0.0 --port 8000
# ngrok http 8000
