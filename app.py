from flask import Flask, render_template, request, redirect, url_for, jsonify
import csv
from math import ceil
import ast
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
from typing import List
import os

# Flask application initialization
app = Flask(__name__)

# Configuration file and default settings
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'service': 'google',
    'google_api_key': '',
    'ollama_api_key': '',
    'ollama_base_url': 'http://localhost:11434',
    'ollama_llm_model': 'llama3.1',
    'ollama_embedding_model': 'all-minilm'
}
def load_config():
    """Load configuration from file or return default if file not found."""

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

# Check if bookmarks.csv exists, if not create one with necessary columns
if not os.path.exists('bookmarks.csv'):
    with open('bookmarks.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['link', 'summary', 'embedding', 'base_url', 'timestamp', 'tags'])

# Load configuration
config = load_config()
    
def get_embedding(corpus, service="google", base_url="http://localhost:11434"):
    """Get embedding for given text using specified service."""
    if service == "google":
        # Google API embedding logic
        api_key = config['google_api_key']
        url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
        if not api_key:
            raise ValueError("Google API Key not found in configuration")
        params = {
            "key": api_key
        }
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{
                    "text": corpus
                }]
            }
        }
        response = requests.post(url, params=params, headers=headers, json=data)
        if response.status_code == 200:
            embedding = response.json().get("embedding", [{}]).get("values")
            return np.array(embedding)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return []
    else:
        # Ollama API embedding logic
        url = f"{base_url}/api/embed"
        payload = {
            "model": "all-minilm",
            "input": corpus
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            embedding = result.get('embeddings', [[]])[0]
            return np.array(embedding)
        else:
            print(f"Error: {response.status_code}")
            return []
        
def embed_all_links(service="google"):
    """Update embeddings for all bookmarks if necessary."""
    sample_query_embedding = get_embedding("Test", service=service)
    bookmarks = read_bookmarks()
    if bookmarks and sample_query_embedding.shape != bookmarks[0]['embedding'].shape:
        for bookmark in bookmarks:
            embedding_input = f"{bookmark['link']} {bookmark['summary']} {' '.join(bookmark.get('tags', []))}"
            embedding = get_embedding(embedding_input, service=service)
            bookmark['embedding'] = embedding
        write_bookmarks(bookmarks)

def get_summary(url, service="google", base_url="http://localhost:11434"):
    """Get summary and tags of webpage content using specified service."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.get_text()
        content_cleaned = ' '.join(content.split())[:1000]
        prompt = f"""Here is the website content: {content_cleaned}. Now summarize it in 20 words or less. Then generate no more than 10 tags that would be relevant to this content. Provide the output in JSON format following this schema: {{"summary": "string", "tags": ["string", ...]}}"""
        if service == "google":
            # Google API summary logic
            api_key = config['google_api_key']
            api_url = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText"
            if not api_key:
                raise ValueError("Google API Key not found in configuration")
            params = {
                "key": api_key
            }
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "prompt": {
                    "text": prompt
                }
            }
            response = requests.post(api_url, params=params, headers=headers, json=data)
            if response.status_code == 200:
                content = response.json().get("candidates", [{}])[0].get("output", "")
                # Now parse the content as JSON
                try:
                    result = json.loads(content)
                    summary = result.get('summary', '').strip()
                    tags = result.get('tags', [])
                    return summary, tags
                except json.JSONDecodeError:
                    print("Failed to parse JSON from response")
                    return "No summary found", []
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return "No summary found", []
        else:
            # Ollama API summary logic
            url = f"{base_url}/api/generate"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "model": config['ollama_llm_model'],
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                content = response.json().get("response", "").strip()
                try:
                    result = json.loads(content)
                    summary = result.get('summary', '').strip()
                    tags = result.get('tags', [])
                    return summary, tags
                except json.JSONDecodeError:
                    print("Failed to parse JSON from response")
                    return "No summary found", []
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return  "No summary found", []
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return "Error generating summary", []

def read_bookmarks():
    """Read bookmarks from CSV file."""
    bookmarks = []
    with open('bookmarks.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 6:
                bookmarks.append({
                    'link': row[0],
                    'summary': row[1],
                    'embedding': np.array(ast.literal_eval(row[2])),
                    'base_url': row[3],
                    'timestamp': row[4],
                    'tags': ast.literal_eval(row[5]) if row[5] else []
                })
    return sorted(bookmarks, key=lambda x: x['timestamp'], reverse=True)

def write_bookmarks(bookmarks):
    """Write bookmarks to CSV file."""
    with open('bookmarks.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['link', 'summary', 'embedding', 'base_url', 'timestamp', 'tags'])
        for bookmark in bookmarks:
            writer.writerow([
                bookmark['link'],
                bookmark['summary'],
                bookmark['embedding'].tolist(),
                bookmark['base_url'],
                bookmark['timestamp'],
                bookmark['tags']
            ])
        
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
    
def semantic_search(query_vector: np.ndarray, document_vectors: List[np.ndarray]):
    """Perform semantic search using cosine similarity."""
    similarities = [
        (i, cosine_similarity(query_vector, doc_vector))
        for i, doc_vector in enumerate(document_vectors)
    ]
    return sorted(similarities, key=lambda x: x[1], reverse=True)

@app.route('/')
def index():
    """Render index page with paginated bookmarks."""
    bookmarks = read_bookmarks()
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_pages = ceil(len(bookmarks) / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_bookmarks = bookmarks[start:end]
    return render_template('index.html', bookmarks=paginated_bookmarks, page=page, total_pages=total_pages)

@app.route('/remove/<path:link>')
def remove_bookmark(link):
    """Remove a bookmark and redirect to index."""
    bookmarks = read_bookmarks()
    bookmarks = [b for b in bookmarks if b['link'] != link]
    write_bookmarks(bookmarks)
    return redirect(url_for('index'))

@app.route('/update_bookmark', methods=['POST'])
def update_bookmark():
    """Update a bookmark's information."""
    data = request.json
    bookmarks = read_bookmarks()
    for bookmark in bookmarks:
        if bookmark['link'] == data['original_link']:
            bookmark['link'] = data['new_link']
            bookmark['summary'] = data['new_summary']
            write_bookmarks(bookmarks)
            return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/search')
def search():
    """Perform semantic search on bookmarks."""
    query = request.args.get('query', '')
    bookmarks = read_bookmarks()
    query_embedding = get_embedding(query, service=config['service'], base_url=config['ollama_base_url'])
    corpus_embeddings = [b['embedding'] for b in bookmarks]
    hits = semantic_search(query_embedding, corpus_embeddings)
    sorted_bookmarks = []
    sorted_scores = []
    for idx, score in hits:
        sorted_bookmarks.append(bookmarks[idx])
        sorted_scores.append(score)
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_pages = ceil(len(sorted_bookmarks) / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_bookmarks = sorted_bookmarks[start:end]
    paginated_scores = sorted_scores[start:end]
    return render_template('search_results.html', 
                           bookmarks=zip(paginated_bookmarks, paginated_scores), 
                           query=query,
                           page=page,
                           total_pages=total_pages)

@app.route('/add_bookmark', methods=['GET', 'POST'])
def add_bookmark():
    """Add new bookmarks."""
    if request.method == 'POST':
        bookmarks_text = request.form['bookmarks']
        bookmarks_list = bookmarks_text.split('\n')
        existing_bookmarks = read_bookmarks()
        for bookmark_url in bookmarks_list:
            bookmark_url = bookmark_url.strip()
            if bookmark_url:
                base_url = re.search(r'https?://([^/]+)', bookmark_url)
                base_url = base_url.group(1) if base_url else bookmark_url
                summary, tags = get_summary(bookmark_url, service=config['service'], base_url=config['ollama_base_url'])
                embedding_input = f"{bookmark_url} {summary} {' '.join(tags)}"
                embedding_bookmark = np.array(get_embedding(embedding_input, service=config['service'], base_url=config['ollama_base_url']))
                if not any(b['link'] == bookmark_url for b in existing_bookmarks):
                    new_bookmark = {
                        'link': bookmark_url,
                        'summary': summary,
                        'embedding': embedding_bookmark,
                        'base_url': base_url,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'tags': tags
                    }
                    existing_bookmarks.append(new_bookmark)
        write_bookmarks(existing_bookmarks)
        return redirect(url_for('index'))
    return render_template('add_bookmark.html')

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """Render and handle configuration page."""
    global config
    if request.method == 'POST':
        config['service'] = request.form['service']
        if config['service'] == 'google':
            config['google_api_key'] = request.form['google_api_key']
        else:
            config['ollama_base_url'] = request.form['ollama_base_url']
            config['ollama_llm_model'] = request.form['ollama_llm_model']
            config['ollama_embedding_model'] = request.form['ollama_embedding_model']
            config['ollama_api_key'] = request.form['ollama_api_key']
        save_config(config)
    return render_template('config.html', config=config)

@app.route('/embed_all_links', methods=['POST'])
def embed_all_links_route():
    """Endpoint to trigger embedding of all links."""
    embed_all_links(service=config['service'])
    return jsonify({"message": "All links have been re-embedded successfully."}), 200

@app.route('/test_models', methods=['GET'])
def test_models():
    """Test embedding and summary generation."""
    try:
        test_text = "This is a test sentence for embedding."
        embedding = get_embedding(test_text, service=config['service'], base_url=config['ollama_base_url'])
        if not isinstance(embedding, np.ndarray) or embedding.size == 0:
            return jsonify({"message": "Embedding test failed. Check your configuration and try again."}), 400
        test_url = "https://example.com"
        summary, tags = get_summary(test_url, service=config['service'], base_url=config['ollama_base_url'])
        if not summary:
            return jsonify({"message": "Summary test failed. Check your configuration and try again."}), 400
        return jsonify({"message": "Models test successful. Embedding and summary generation are working correctly."}), 200
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Perform a semantic search on bookmarks.

    Request format:
    {
        "query": "search query string",
        "page": 1,  // optional, default is 1
        "per_page": 10  // optional, default is 10
    }

    Returns:
    {
        "results": [
            {
                "link": "bookmark url",
                "summary": "bookmark summary",
                "base_url": "base url of the bookmark",
                "timestamp": "bookmark creation timestamp",
                "tags": ["tag1", "tag2", ...],
                "similarity": float  // similarity score
            },
            ...
        ],
        "total_results": int,
        "page": int,
        "per_page": int,
        "total_pages": int
    }
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query parameter"}), 400

        query = data['query']
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)

        bookmarks = read_bookmarks()
        query_embedding = np.array(get_embedding(query, service=config['service'], base_url=config['ollama_base_url']))
        
        corpus_embeddings = [b['embedding'] for b in bookmarks]
        hits = semantic_search(query_embedding, corpus_embeddings)
        results = [
            {**bookmarks[idx], 'similarity': float(score)}
            for idx, score in hits
        ]
        
        total_results = len(results)
        total_pages = ceil(total_results / per_page)
        
        start = (page - 1) * per_page
        end = start + per_page
        paginated_results = results[start:end]

        return jsonify({
            "results": [
                {
                    "link": result['link'],
                    "summary": result['summary'],
                    "base_url": result['base_url'],
                    "timestamp": result['timestamp'],
                    "tags": result.get('tags', []),
                    "similarity": result['similarity']
                } for result in paginated_results
            ],
            "total_results": total_results,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add', methods=['POST'])
def api_add_bookmark():
    """
    Add a new bookmark.

    Request format:
    {
        "url": "https://example.com"
    }

    Returns:
    - 201: Bookmark added successfully
    - 400: Missing url parameter or bookmark already exists
    - 500: Internal server error
    """
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({"error": "Missing url parameter"}), 400

        bookmark_url = data['url'].strip()
        existing_bookmarks = read_bookmarks()

        if any(b['link'] == bookmark_url for b in existing_bookmarks):
            return jsonify({"error": "Bookmark already exists"}), 400

        base_url = re.search(r'https?://([^/]+)', bookmark_url)
        base_url = base_url.group(1) if base_url else bookmark_url
        summary, tags = get_summary(bookmark_url, service=config['service'], base_url=config['ollama_base_url'])
        embedding_input = f"{bookmark_url} {summary} {' '.join(tags)}"
        embedding_bookmark = np.array(get_embedding(embedding_input, service=config['service'], base_url=config['ollama_base_url']))

        new_bookmark = {
            'link': bookmark_url,
            'summary': summary,
            'embedding': embedding_bookmark,
            'base_url': base_url,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tags': tags
        }
        existing_bookmarks.append(new_bookmark)
        write_bookmarks(existing_bookmarks)

        return jsonify({
            "message": "Bookmark added successfully",
            "bookmark": {
                "link": new_bookmark['link'],
                "summary": new_bookmark['summary'],
                "base_url": new_bookmark['base_url'],
                "timestamp": new_bookmark['timestamp'],
                "tags": new_bookmark['tags']
            }
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
