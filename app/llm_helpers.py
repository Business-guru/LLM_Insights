from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate,  HumanMessagePromptTemplate,  SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from typing import List, Tuple, Dict
import logging
from dotenv import load_dotenv
import os

from prompts import gen_prompt, ref_prompt, human_msg_prompt

load_dotenv()

groq_api_key = os.getenv("groq_api_key")

logger = logging.getLogger(__name__)

class result_fields(BaseModel):
    site_id: int = Field(description=" corresponds to the site_id of the selectes website")
    reasoning: str = Field(description = " A detailed explanation of the reasoning for choosing this particular website.")

class gen_snippet_ranker(BaseModel):
    initial_reasoning: str = Field(
        ...,
        description=(
            "An initial analysis of the websites for relevance based on specified parameters. Be elaborate, and very specific here, taking into account quality, date of publication, citations/proof etc."
        )
    )

    evaluator: str = Field(
        ...,
        description=(
            "An evalaution of the initial reasoning, with suggestion improvements after careful analysis. Review the initial reasoning, comment on it and even mention any additional considerations"
        )
    )

    final_reasoning: str = Field(
        ...,
        description=(
            "The corrected reasoning based on the feedback from evaluator. Be elaborate, and very specific here, taking into account quality, diversity and overlap."
        )
    )

    ranker_logs: str = Field(
        description = ("This section is your working space to document all your thoughts and reasoning for the ordering of the websites. Use it to logically work through the sorting process and explain why each website is placed in a specific order. As you sort the website, evaluate their quality, relevance, and detail the rationale behind your selections. This is where you perform the final logic for sorting and present the final ordered list of websites based on your analysis.")
    )
    
    top_k: List[result_fields] = Field (
        ...,
        description=(
            '''
            This represents a list of pairs of site_ids. 
            Your job is to create an ordered list of websites based on their total score using a listwise ranking approach. Start by selecting the website with the highest quality score first. For each subsequent selection, choose the website that maximizes the total score, which is calculated as a combination of the quality score(numerical data support), relevance score, latest score and a penalty.
            '''
        )
    )

class gen_uniqueness_lister:        

    def __init__(self,llm_model="llama-3.3-70b-versatile"):
        self.llm_model=llm_model
        self.generation_pipeline = self.create_gen_pipeline()
    
    def create_gen_pipeline(self):

        gen_llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=self.llm_model) # type: ignore #ignore

        gen_llm = gen_llm.with_structured_output(gen_snippet_ranker,method="function_calling",include_raw=True)

        
        gen_chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(gen_prompt),
            HumanMessagePromptTemplate.from_template(human_msg_prompt)
        ])

        return gen_chat_template | gen_llm 
    
    
    def run(self, k: int, site_mapping: dict[int,str], query: str,max_retries: int = 3):
              
        site_id_content_pairs = ''''''
        for site_id, content in site_mapping.items():
            if len(content):  
                site_id_content_pairs += f'''
for site_id: {site_id}, website content is are:
{content}

'''     
        inputs = {
            "k": k,
            "site_id_content_pairs": site_id_content_pairs,
            "query": query
        }
        
        attempt = 0
        output = None

        while attempt < max_retries:
            try:
                output = self.generation_pipeline.invoke(inputs)
                
                break  # If successful, exit the loop
            except Exception as e:
                attempt += 1
                logger.error(f"LLM Output Parsing Error at Gen Section (Attempt {attempt}/{max_retries}): {e}")
                
                if attempt == max_retries:
                    output = gen_snippet_ranker(
                                initial_reasoning="",
                                evaluator="",
                                final_reasoning="",
                                ranker_logs="",
                                top_k=[]
                            )

        return output


class ref_uniqueness_lister:
    def __init__(self, llm_model="llama-3.3-70b-versatile"):
        self.llm_model = llm_model
        self.ref_pipeline = self.create_ref_pipeline()
        
    
    def create_ref_pipeline(self):

        ref_llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=self.llm_model) # type: ignore 

        ref_llm = ref_llm.with_structured_output(gen_snippet_ranker,method="function_calling",include_raw=True)
        
        ref_chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(ref_prompt),
            HumanMessagePromptTemplate.from_template(human_msg_prompt)
        ])

        return ref_chat_template | ref_llm
    
    def run(self, k: int, site_mapping: dict[int,str], query: str, initial_shortlisted_site_ids: str,max_retries: int = 3):

        site_id_content_pairs = ''''''
        for site_id, content in site_mapping.items():
            if len(content):  
                site_id_content_pairs += f'''
for site_id: {site_id}, website content is are:
{content}

'''     
        inputs = {
            "k": k,
            "site_id_content_pairs": site_id_content_pairs,
            "query": query,
            "initial_shortlisted_site_ids": initial_shortlisted_site_ids
        }

        attempt = 0
        output = None

        while attempt < max_retries:
            try:
                output = self.ref_pipeline.invoke(inputs)
                break  # If successful, exit the loop
            except Exception as e:
                attempt += 1
                logger.error(f"LLM Output Parsing Error at Ref Section (Attempt {attempt}/{max_retries}): {e}")

                if attempt == max_retries:
                    output = gen_snippet_ranker(
                                initial_reasoning="",
                                evaluator="",
                                final_reasoning="",
                                ranker_logs="",
                                top_k=[]
                            )

        return output




class SelfReflector:
    def __init__(self, k: int, site_mapping, query,llm_model="llama-3.3-70b-versatile"):
        self.k = k
        self.site_mapping = dict(site_mapping)
        self.gen_obj = gen_uniqueness_lister(llm_model)
        self.ref_obj = ref_uniqueness_lister(llm_model)
        self.query = query
        self.llm_model = llm_model

    def run_initial_analysis(self):
        initial_results = self.gen_obj.run(
            k=self.k,
            site_mapping=self.site_mapping,
            query = self.query
        )
        return initial_results

    def run_final_analysis(self, selected_ids):
        final_results = self.ref_obj.run(
            k=self.k,
            site_mapping=self.site_mapping,
            query = self.query,
            initial_shortlisted_site_ids=selected_ids
        )
        return final_results

    def execute(self):
        #TODO: Implementing iterations for self reflection(o/p)

        initial_results = self.run_initial_analysis()['parsed'] #type:ignore

        selected_ids = [result.site_id for result in initial_results.top_k] 

        final_results = self.run_final_analysis(selected_ids)['parsed'] #type:ignore
        final_results = [result.site_id for result in  final_results.top_k] 

        return final_results

