
from sentence_transformers import SentenceTransformer

def getSTmodel(model_name:str, cache_dir:str = "model_cache") -> SentenceTransformer:
    """Check the cache for SentenceTransformer model or download it"""

    # Do we have it?
    try:
        model = SentenceTransformer(model_name, 
                                    cache_folder = cache_dir,
                                    local_files_only = True)
    except:
        # we didn't have it, so let's get it
        model = SentenceTransformer(model_name, 
                                    cache_folder = cache_dir, 
                                    local_files_only = False)
        
    return model
