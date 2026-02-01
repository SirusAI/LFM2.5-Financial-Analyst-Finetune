import urllib.request
url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
urllib.request.urlretrieve(url, "convert_hf_to_gguf.py")
print("Downloaded convert_hf_to_gguf.py")
