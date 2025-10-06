import requests
import re
import json

topic = "Proclamation of the Republic of Turkey"
url = "https://en.wikipedia.org/w/api.php"

def fetch_wikipedia_article(topic):
    headers = {
        'User-Agent': 'Serhat/1.0 (https://github.com/serhattay)'
    }
    
    params = {
        "action": "query",
        "format": "json",
        "titles": topic,
        "prop": "extracts",
        "explaintext": True
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response
        data = response.json()

        # print(json.dumps(data, indent=4))  # Pretty-print the JSON response for debugging
        
        # Extract the page ID and then the extract text
        page_id = list(data['query']['pages'].keys())[0]
        if page_id != "-1": # Check if page exists
            return data["query"]["pages"][page_id]["extract"]
        else:
            return "Article not found."

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def clean_text(text):
    if not text:
        return None
    text = re.sub('"', '', text)  # Remove all double quotes
    text = re.sub(r"== See also ==.*?(== References ==|== External links ==|$)", "", text, flags=re.DOTALL)  # Remove See also section and everything after
    text = re.sub(r"===.*?===", "", text)  # Remove all section headers
    text = re.sub(r"==.*?==", "", text)  # Remove all section headers
    text = re.sub(r"\[\d+\]", "", text)  # Remove all references
    text = re.sub(r" +", " ", text)  # Remove all extra spaces

    return text

def split_sentences(text):
    parts= re.split(r'(\.\n|\?|!|(?<=\.) (?=[A-Z]))', text)
    
    combined = [parts[i] + parts[i + 1] for i in range(0, len(parts)-1, 2)] 
    return combined


def save_article_to_file(file_name, topic):
    with open("wikipedia_article.txt", "w", encoding="utf-8") as f:
        article_text = clean_text(fetch_wikipedia_article(topic))
        
        if article_text:
            article_sentences = split_sentences(article_text)
            for sentence in article_sentences:
                f.write(sentence.strip() + "\n")
                
            print("Article fetched and saved to wikipedia_article.txt")
        else:
            print("Failed to fetch the article.")
            
# save_article_to_file("wikipedia_article.txt", topic)