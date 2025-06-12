from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_poem(poem):
    return summarizer(poem, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
