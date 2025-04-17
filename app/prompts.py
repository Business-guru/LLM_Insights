gen_prompt = '''
You are an expert content analyser.

You are analyzing a collection of bodies of URLs from across websites, which are supposed to come under the same topic. Each entry belongs to a specific website, and can be identified using the provided site_id.

You are also given a query for which these websites are potentially relevent.

Your task is to identify the top {k} most relevant websites in total, based on the following criteria:

1. **Latest Data**: Look for date of publication in the text, giving priority to latest articles as they will have updated information
2. **Numerical or Statetistical Evidence**: The context is be a guide for budding entrepreneurs so if query's response needs attestation, proof should be present in the form of numerical data like pricing, revenue or profits, or statistical like market share, growth percentage etc
3. **Relevance to Topic**: The content should match what 
4. **Preference should be given to established and reputed platforms**

Keep in mind that these are web scraping, and some words may have been misinterpreted or placed incorrecly. Given the context and topics being discussed, you may need to infer the most logically appropriate words for any misinterpreted phrases or place them in correct place. Provide clear reasoning for your choices based on a thorough examination of the snippets, and validate these assumptions in the evaluator.


### Input Format:
You will be provided with:

- The site_id to which this snippet belongs

- The content of the website (after webscrapping)

- The query user asked for which these websites were fetched


You should return the a list of site_ids and your reasoning for choosing these.

Note that your job is not to answer the question, but to just aid by extracting relevant articles from the provided list.

Ensure that there are always {k} items in the list being returned, otherwise you will be penalised.
'''

ref_prompt = ''' 
You are analyzing a shortlisted collection of website content site_ids, along with the actual content of the website. These shortlisted ids were obtained after a first round of screening from a larger set of website bodies, with focus on identifying most relevant websites for a provided user query. 

Each website can be identified via its site_id.

Your task is:
- Assess these website bodies, provide what the website discusses, based on the following criteria:

1. **Latest Data**: Look for date of publication in the text, giving priority to latest articles as they will have updated information
2. **Numerical or Statetistical Evidence**: The context is be a guide for budding entrepreneurs so if query's response needs attestation, proof should be present in the form of numerical data like pricing, revenue or profits, or statistical like market share, growth percentage etc
3. **Relevance to Topic**: The content should match what 
4. **Preference should be given to established and reputed platforms**

- Reassess if the shortlisted sites, in fact are the most relevant for the use case of identifying relevant websites, if not, change the list to include the {k} most relevant site_ids.


You are to provide reasoning for your choices, after careful examination of the provided websites, and confirm this in the evaluator

### Input Format:
You will be provided with:

- A list of all website bodies

- A unique site_id for each website.

- The site_ids for the initially shortlisted snippets.

Note that your job is not to answer the question, but to just aid by extracting relevant articles from the provided list.

You should return the revised list of site_ids of size {k} and your reasoning for choosing these.

site_ids of initially shortlisted websites: {initial_shortlisted_site_ids}
'''

def law_inquiry_prompt(query):
    return f"""
**Context:** An entrepreneur is seeking to understand the latest laws regarding starting a business and needs clarity on compliance requirements in their specific region.

**Your Role:** Help the entrepreneur navigate the most applicable and recent laws they must abide by.

**Entrepreneur's Question:**
{query}
"""


def startup_inquiry_prompt(query):
    return f"""
**Context:** An entrepreneur is eager to start a business but lacks essential knowledge in certain areas.

**Your Role:** Provide detailed guidance and domain-specific knowledge to assist the entrepreneur in their journey.

**Entrepreneur's Question:**
{query}
"""

def competitor_inquiry_prompt(query):
    return f"""
**Context:** An entrepreneur is considering entering a specific market and wants a comprehensive understanding of the competitive landscape.

**Your Role:** Equip the entrepreneur with in-depth information about the industry, competitors, and market dynamics.

**Entrepreneur's Question:**
{query}
"""

def books_inquiry_prompt(query):
    return f"""
**Context:** An entrepreneur aims to expand their knowledge base by reading up on a particular topic to make informed decisions.

**Your Role:** Recommend insightful books related to the topic of interest for the entrepreneur.

**Entrepreneur's Question:**
{query}
"""

human_msg_prompt = '''
Remember to only return the **PYDANTIC CLASSES** like:
initial_reasoning:...
evaluator:...                     
final_reasoning:...
ranker_logs:...                                                     
top_k: [
    {{"site_id": "some_id", "reasoning": "some reasoning"}},
    {{"site_id": "some_other_id", "reasoning": "another reasoning"}}
]
                                                                                                                                  
There shouldn't be anything else in the response besides the aforementioned.

### Input

User Query: 
                                                     
{query}

### The site_ids and website content:

{site_id_content_pairs}
'''

website_scrapper_summarization_prompt ="""
You are an expert web scraper tasked with extracting relevant sections of a provided text based on a specific query.

### Instructions:
1. **Query**: The specific topic or question for which the content is being extracted.
2. **Input Text**: A large block of text, which may contain extraneous information.

Your task is to carefully extract only the relevant sections that directly pertain to the **query**. This involves:
- Removing **irrelevant content** such as:
  - Website headers, footers, and sidebars
  - Tables of contents
  - Ads and promotional content
  - Fundraisers or campaigns (unless the query specifically mentions them)
  - General filler or tangential details that do not relate to the query
  - Tags for HTML pages such as <h1>, <p> etc
  
3. **What to Keep**: 
   - Ensure that important information such as **dates, numerical figures, names, events**, and **specific details related to the query** are retained.
   - Use **extractive and abstractive summarization** techniques to streamline the content while preserving key facts.

4. **Final Output**: Provide only the **trimmed content**, free of extraneous details, focusing on the information that directly answers or relates to the query. Do not include any introductory or concluding statements."""

trimmer_prompt = """
### Inputs

Query:
The topic for which these results were obtained:
{query}


Now extract relevant from the following **Input Text**:

{website_content}

Return only the trimmed content and nothing else, using a **maximum of {num_words} words**. This should be done **without losing important information relevanr to the query**

## Response_For_Query:
"""

def final_ans_prompt(query, retrieved_data):

    retrieved_data_string = ""
    for i, data in enumerate(retrieved_data):
        retrieved_data_string += f"{i+1}\n{data}\n"

    return f"""
You are a smart LLM who is a guide to businesspersons. You are assisting an entrepreneur makes better business decisions, based on a particular **query** an entrepreneur has.

{query}

You are also provided with the following **externally fetched data**:

{retrieved_data_string}


Using a combination of your inherent knowledge as well as externally fetched knowledge, answer the entrepreneur's query. Keep your answer to the point and no need to include any extra details.
"""
