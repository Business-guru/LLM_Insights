
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import logging

from llm_helpers import SelfReflector

logger = logging.getLogger(__name__)

def split_into_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]


def eliminator(query, website_data, max_size=4, batch_size=7):
    # Check if the website data is less than or equal to max_size
    # If so, return the original website data without processing
    if len(website_data) <= max_size:
        return website_data
    
    all_top_k = []  # Initialize a list to hold results from all batches
    # Enumerate website data to create an indexed list of tuples (index, data)
    website_data_without_url = [(i + 1, data) for i, (url,data) in enumerate(website_data)]
    # urls = [[(i + 1, url) for i, (url,data) in enumerate(website_data)]]
    
    # Split the website data into smaller batches for processing
    batches = list(split_into_batches(website_data_without_url, batch_size))

    # Use ThreadPoolExecutor to run tasks in parallel
    with ThreadPoolExecutor(max_workers=min(32,len(batches))) as executor:
        # Submit all batches for parallel processing
        # Each future corresponds to a batch being processed
        futures = {
            executor.submit(
                lambda b=batch: SelfReflector(
                    k=min(max_size, len(b)), 
                    site_mapping=dict(b), 
                    query=query  
                ).execute() 
            ): batch for batch in batches  
        }

        
        # Iterate over futures as they complete
        for future in as_completed(futures):
            try:
                
                final = future.result()
                
                all_top_k.extend(final)  # type: ignore 

            except Exception as e:
                
                logger.error(f"Error processing batch: {e}")

    logger.info(f"*****all_top_k: {all_top_k}******")

    refined_result = [website_data[int(i)-1] for i in all_top_k]
    
    return eliminator(query,refined_result, max_size, batch_size)
