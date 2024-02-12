Project to download current top news headlines and cluster/categorize them by topic and semantic meaning.

I intend to start by downloading news artciles from newsapi.  I'll need to get an API KEY.

Next will be vectorizing the articles using a pr-trained LLM.  
I'd like to expolore how I can use ChatGPT or other generative model to absorb and analyze this data for direct question and answer chatbot.

Initially, I'll cluster the vectors - kmeans or maybe DBSCAN.
I'll summarize the clusters - we'll see what model can absorb the full article context.

I've used chatGPT to generate initial code:


from newsapi import NewsApiClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

Initialize NewsAPI client
newsapi = NewsApiClient(api_key='YOUR_API_KEY')

### Fetch today's top headlines from NewsAPI
top_headlines = newsapi.get_top_headlines(language='en')

### Extract article content from the top headlines
articles = [article['content'] for article in top_headlines['articles'] if article['content']]

### Filter out articles with empty content
articles = [article for article in articles if article]

### Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

### Create embeddings for the articles
embeddings = model.encode(articles)

### Cluster the embeddings
num_clusters = 5  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(embeddings)
clusters = kmeans.labels_

### Summarize the clusters
for cluster_idx in range(num_clusters):
    cluster_articles = [articles[i] for i, cluster_label in enumerate(clusters) if cluster_label == cluster_idx]
    centroid_idx = pairwise_distances_argmin_min(kmeans.cluster_centers_[cluster_idx].reshape(1, -1), embeddings)[0][0]
    centroid_article = articles[centroid_idx]
    
    print(f"\nCluster {cluster_idx + 1} - Centroid Article: {centroid_article}")
    print("Top Articles:")
    for article in cluster_articles[:3]:
        print(article)

