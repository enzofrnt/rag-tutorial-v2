from django.db import models
from pgvector.django import VectorField


class Embedding(models.Model):
    file_name = models.CharField(max_length=255)
    content = models.TextField()
    embedding = VectorField(dimensions=768)  # Mise à jour à 768

    def __str__(self):
        return self.file_name


from django.db import models
from pgvector.django import IvfflatIndex, VectorField


class Chunk(models.Model):
    source = models.CharField(max_length=255)
    page = models.IntegerField()
    chunk_index = models.IntegerField()
    content = models.TextField()
    embedding = VectorField(dimensions=768)  # Taille de l'embedding

    class Meta:
        indexes = [
            IvfflatIndex(
                name="embedding_cosine_idx",
                fields=["embedding"],
                lists=100,  # Nombre de clusters
                opclasses=["vector_cosine_ops"],  # Utilisation de la distance cosinus
            ),
        ]

    def __str__(self):
        return f"{self.source} - Page {self.page}, Chunk {self.chunk_index}"
