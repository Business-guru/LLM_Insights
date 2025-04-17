from langchain_community.tools import BraveSearch, DuckDuckGoSearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from dotenv import load_dotenv
import json
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate,  HumanMessagePromptTemplate,  SystemMessagePromptTemplate
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint 
import re
from transformers import AutoTokenizer
import logging


from prompts import website_scrapper_summarization_prompt,trimmer_prompt

load_dotenv()

brave_api_key = os.getenv("brave_api_key")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

logger = logging.getLogger(__name__)

def extract_urls_and_further_questions(query):

    urls = []
    further_questions = []
   
    ddg = DuckDuckGoSearchResults(output_format="list",num_results = 10)
    ddg_res = ddg.invoke(query)
    ddg_urls = [r["link"] for r in ddg_res]
    urls.extend(ddg_urls)

    brave_tool = BraveSearch.from_api_key(api_key = brave_api_key,search_kwargs={"count": 10}) #type: ignore
    brave_res = brave_tool.run(query)
    brave_res = json.loads(brave_res)
    brave_urls = [r["link"] for r in brave_res]
    urls.extend(brave_urls)


    serp_search = GoogleSerperAPIWrapper()
    serp_res = serp_search.results(query)
    serp_urls = [r['link'] for r in serp_res['organic']]
    urls.extend(serp_urls)
    further_questions.extend([x['question'] for x in serp_res.get('peopleAlsoAsk','')])
    further_questions.extend([x['query'] for x in serp_res.get('relatedSearches','')])

    urls = set(urls)
    further_questions = set(further_questions)

    with open("urls.txt", "w") as file:
        for item in urls:
            file.write(item + "\n")

    with open("further_questions.txt", "w") as file:
        for item in urls:
            file.write(item + "\n")

    return urls,further_questions

# def extract_urls_and_further_questions(query):

#     # Define the file path for precomputed results
#     precomputed_file = "precomputed_results.json"

#     # Check if precomputed results file exists
#     if os.path.exists(precomputed_file):
#         with open(precomputed_file, "r") as file:
#             precomputed_data = json.load(file)
        
#         # Fetch results from precomputed data
#         urls = set(precomputed_data.get('urls', []))

#         import random
#         urls = set(random.sample(list(urls), min(10, len(urls))))

#         further_questions = set(precomputed_data.get('further_questions', []))
        
#     else:
#         urls = []
#         further_questions = []

#         # DuckDuckGo search
#         ddg = DuckDuckGoSearchResults(output_format="list", num_results=10)
#         ddg_res = ddg.invoke(query)
#         ddg_urls = [r["link"] for r in ddg_res]
#         urls.extend(ddg_urls)

#         # Brave search
#         brave_tool = BraveSearch.from_api_key(api_key=brave_api_key, search_kwargs={"count": 10})  # type: ignore
#         brave_res = brave_tool.run(query)
#         brave_res = json.loads(brave_res)
#         brave_urls = [r["link"] for r in brave_res]
#         urls.extend(brave_urls)

#         # Google search via Serp API
#         serp_search = GoogleSerperAPIWrapper()
#         serp_res = serp_search.results(query)
#         serp_urls = [r['link'] for r in serp_res['organic']]
#         urls.extend(serp_urls)

#         # Collect further questions
#         further_questions.extend([x['question'] for x in serp_res['peopleAlsoAsk']])
#         further_questions.extend([x['query'] for x in serp_res['relatedSearches']])

#         # Remove duplicates by converting to a set
#         urls = set(urls)
#         further_questions = set(further_questions)

#         # Store results in precomputed file
#         precomputed_data = {
#             "urls": list(urls),
#             "further_questions": list(further_questions)
#         }
#         with open(precomputed_file, "w") as file:
#             json.dump(precomputed_data, file, indent=4)

#     # Save URLs to file
#     with open("urls.txt", "w") as file:
#         for item in urls:
#             file.write(item + "\n")

#     # Save further questions to file
#     with open("further_questions.txt", "w") as file:
#         for item in further_questions:
#             file.write(item + "\n")

#     return urls, further_questions

def llm_summarizer(query,text,llm_model,num_words)->str:
    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        task="text-generation",
        max_new_tokens=num_words*4/5,
        do_sample=False,
        repetition_penalty=1.03,
    ) #type: ignore
    llm = ChatHuggingFace(llm=llm)

    trimmer_chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(website_scrapper_summarization_prompt),
            HumanMessagePromptTemplate.from_template(trimmer_prompt)
        ]) 
    
    trimmer = trimmer_chat_template | llm
    res = trimmer.invoke({'query': query,'website_content':text,'num_words': num_words}).content
    
    return res #type: ignore

def summarize_header_footer(query,first_part, last_part, TOKEN_LIMIT, tokenizer,llm_model)->str:
    total_tokens = len(first_part) + len(last_part)

    if total_tokens <= TOKEN_LIMIT * 2:
        
        combined_tokens = first_part + last_part

        
        mid_point = len(combined_tokens) // 2
        first_half_tokens = combined_tokens[:mid_point]
        second_half_tokens = combined_tokens[mid_point:]

        
        first_text = tokenizer.decode(first_half_tokens)
        second_text = tokenizer.decode(second_half_tokens)

        # Summarize each half
        summary_first = llm_summarizer(query, first_text, llm_model, 50)
        summary_second = llm_summarizer(query, second_text, llm_model, 50)

        return summary_first + "\n" + summary_second  # type: ignore

    else:
        if len(first_part) > len(last_part):
            # Take the last TOKEN_LIMIT tokens from first_part
            selected_tokens = first_part[-TOKEN_LIMIT:]
        else:
            # Take the first TOKEN_LIMIT tokens from last_part
            selected_tokens = last_part[:TOKEN_LIMIT]

        selected_text = tokenizer.decode(selected_tokens)
        return llm_summarizer(query,selected_text,llm_model,50)


def split_text(query,text: str,tokenizer,llm_model: str,TOKEN_LIMIT:int = 30000) -> tuple[str, str, int]:
    """Returns (truncated_text, trimmed_sections, x_value)"""

    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    overflow = total_tokens - TOKEN_LIMIT
    
    if overflow <= 0:
        return "",text, 0  # No trimming needed

    # Split tokens: First 30% of excess and last 70% of excess
    split_point_1 = int(3 * overflow / 10)
    split_point_2 = int(7 * overflow / 10)

    first_part = tokens[:split_point_1]  # First 30% of excess
    last_part = tokens[-split_point_2:]  # Last 70% of excess
    middle_section = tokens[split_point_1:-split_point_2]  # Remaining

    trimmed_text = tokenizer.decode(first_part + last_part)
    middle_text = tokenizer.decode(middle_section)

    if len(first_part) + len(last_part) > TOKEN_LIMIT:
        trimmed_text = summarize_header_footer(query,first_part ,last_part,TOKEN_LIMIT,tokenizer,llm_model)
    
    return trimmed_text, middle_text,overflow 

def extract_text_from_url(url,query,llm_model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'):
    headers = {"User-Agent": "Mozilla/5.0"}  # Avoids bot detection
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        logger.error(f"Failed to retrieve page, status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    text = ' '.join([p.get_text() for p in soup.find_all(["p", "h1", "h2", "h3", "li","div","span","body"])])
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n+', '\n', text)

    llm1,llm2 = None,None
    res1,res2 = "",""
    llm1_words, llm2_words = 0,0

    tokenizer = AutoTokenizer.from_pretrained(llm_model,token=HUGGINGFACEHUB_API_TOKEN)
    other, body, overflow = split_text(query,text,tokenizer,llm_model)

    if not other:
        res1 = ""
    else:
        res1 = llm_summarizer(query,other,llm_model,150)
    
    res2 = llm_summarizer(query,body,llm_model,600)

    return url,res1+"\n"+res2 #type: ignore











