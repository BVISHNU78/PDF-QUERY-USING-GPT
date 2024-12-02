# Token counting
import tiktoken
text="hello world LLM model"
encoding=tiktoken.get_encoding('cl100k_base')
encoding =tiktoken.encoding_for_model("gpt-3.5-turbo")
sentences=encoding.encode(text)
print("Tokenized text",sentences)
def counts_tokens(string:str,encoding_name:str):
    encoding =tiktoken.get_encoding(encoding_name)
    num_tokens=len(encoding.encode(string))
    return num_tokens
counts_tokens(text,'cl100k_base')
#Turn tokens into text with encoding.base()
#decode() method u can convert list of vectors back into a xoherent string 
encoding=tiktoken.encoding_for_model("gpt-3.5-turbo")
en_dec=encoding.decode([15339, 1917, 445, 11237, 1646])
print(en_dec)
encoding=tiktoken.encoding_for_model("gpt-3.5-turbo")
en_sing=[encoding.decode_single_token_bytes(token) for token in [15339, 1917, 445, 11237, 1646]]
print(en_sing)
#comparing encodings
def compare_enc(string:str):
    print(f'strings:"{string}"')
    for encoding_name in["r50k_base","p50k_base","cl100k_base"]:
        encoding=tiktoken.get_encoding(encoding_name)
        token_integers=encoding.encode(encoding_name)
        num_tokens=len(token_integers)
        token_bytes=[encoding.decode_single_token_bytes(token)for token in token_integers]
        print()
        print(f"{encoding_name} encoding:{num_tokens} tokens")
        print(f"token_integers:{token_integers}")
        print(f"token bytes:{token_bytes}")
print(compare_enc("troplogy"))

#counting token for chat  completions  when api calls

def num_tokens_from_messages(messages,model="gpt-3.5-turbo-0613"):
    try:
        encoding =tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warining model not found.Using cl100k_base encoding") 
        encoding=tiktoken.get_encoding("cl100_base")
    if model in {"gpt-4","gpt-3.5-turbo","gpt-4-1106-preview","gpt-4-32k-0613"}:
        tokens_per_message=4
        tokens_per_name=-1
    else:
        raise NotImplementedError(
            f"""num_tokens_frommessages() is not implemted for model{model}.""")
    num_tokens=0
    for message in messages:
        num_tokens +=tokens_per_message
        for key,value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens +=3
    print(num_tokens)
    return num_tokens
