class embedding:
    def __init__(self, model):
        self.model = model
    def embed_documents(self,items):
        embeddings = self.model.encode(items)
        return embeddings.tolist()
    def embed_query(self,query):
        return self.model.encode(query).tolist()