import json
import os

class TokenExtender:
    """
    Reused from sft.py/sft_gpr.py to extract SID tokens.
    """
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens

def get_sid_token_ids(tokenizer, data_path, dataset, index_file=".index.json"):
    """
    Returns the token IDs of all SID tokens in the tokenizer.
    """
    extender = TokenExtender(data_path, dataset, index_file)
    new_tokens = extender.get_new_tokens()
    
    # We ensure these tokens are actually in the tokenizer
    # Note: If the tokenizer was saved from the teacher model, these should already be in it.
    sid_token_ids = []
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            sid_token_ids.append(token_id)
        else:
            print(f"Warning: Token {token} not found in tokenizer vocabulary!")
            
    return sid_token_ids

def get_sid_token_ids_from_info(tokenizer, info_file):
    """
    Alternative method to extract SID tokens from the info file (e.g. data/Amazon/info/Office_Products_*.txt).
    """
    sid_tokens = set()
    with open(info_file, 'r') as f:
        for line in f:
            parts = line.split('\t')
            if len(parts) > 0:
                sid_str = parts[0].strip() # e.g. <a_60><b_159><c_203>
                # Parse out <a_60>, <b_159>, <c_203>
                import re
                tokens = re.findall(r'<[a-z]_\d+>', sid_str)
                sid_tokens.update(tokens)
                
    sid_tokens = sorted(list(sid_tokens))
    
    sid_token_ids = []
    for token in sid_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            sid_token_ids.append(token_id)
        else:
            print(f"Warning: Token {token} not found in tokenizer vocabulary!")
            
    return sorted(sid_token_ids)
