## Embeddings
The fundamental tool that will power our semantic search is embeddings: numerical representations of text that capture the meaning of words.

Specifically, embeddings are vectors: 
 - A vector is a fancy word for a list of numbers.
 - A vector is a list of numbers that represents a point in space, specifically, a direction and a magnitude from the origin (0, 0).

The "embedding" happens when we take a piece of text and convert it into a vector. For example:
``` 
"King" -> [3.5, 2.5]
"Queen" -> [3.0, 2.0]
"Human" -> [3.0, -3.0]
```
[embedding_example]:./embedding_example.png
![embedding_example][embedding_example]

The idea is that the distance between the vectors represents how similar the meanings of the words are. King and queen are very similar, and therefore their vectors are close together. Human is less similar, so its vector is further away.

*Numbers in vectors are sometimes normalized to a smaller scale of floating point numbers like -1.0 to 1.0 for easier comparison.*

**Embedding Models:**
The process of converting text into vectors requires a lot of data and computation – it's a machine learning "training" process that basically slurps up a massive amount of text data to learn patterns about how different words and phrases relate to each other.

*For this experiment I am using a small pre-trained embedding model  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)*

### Dot Product Similarity
- **The dot product measures how much two vectors "point in the same direction."** 
- It's calculated by multiplying corresponding elements and summing the results. Say we want the dot product of these two vectors:
````
[.8, .5, .5]
[.5, .4, .6]
First we multiply each pair of corresponding elements:

.8 × .5 = 0.4
.5 × .4 = 0.2
.5 × .6 = 0.3
Then we sum those products:

0.4 + 0.2 + 0.3 = 0.9
````
The final result is 0.9. 
- The more similar the vectors are, the higher the dot product will be. 
- If they point in opposite directions, the dot product will be negative.

### Cosine Similarity

Dot product has a limitation: it is affected by **vector magnitude**.

Vectors have two properties:

- **Direction**
- **Magnitude (length)**

For **semantic similarity**, we usually care only about the **direction**, not the magnitude.

For example, these two vectors have the **same direction but different magnitudes**:
````
[0.6, 0.8] (magnitude = 1.0)
[3.0, 4.0] (magnitude = 5.0)
````

The **dot product** of these vectors is:

````
(0.6 × 3.0) + (0.8 × 4.0) = 1.8 + 3.2 = 5.0
````

Even though they represent the **same direction**, the result is large because the second vector has a **bigger magnitude**.

In semantic search we want to **ignore magnitude and focus only on direction**.

---

## Cosine Similarity

**Cosine similarity** solves this problem by measuring the **angle between vectors** instead of their magnitude.

The result ranges between **-1 and 1**:

| Value | Meaning |
|-----|-----|
| **1.0** | Vectors point in exactly the same direction (perfect similarity) |
| **0.0** | Vectors are perpendicular (no similarity) |
| **-1.0** | Vectors point in opposite directions (completely dissimilar) |

---

## Cosine Similarity Formula
$$ \text{Cosine Similarity} = \frac{\text{dot_product}(A, B)}{\text{magnitude}(A) \times \text{magnitude}(B)} $$
And it works in two steps:

1. **Dot Product**: Calculate similarity: The dot product measures how much vectors align.

2. **Remove length bias**: Dividing by magnitudes removes the effect of vector size.

Since **all-MiniLM-L6-v2** was trained with cosine similarity, we have to use the same for our searches
[all-MiniLM-L6-v2 fine doc](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#fine-tuning)

## Locality-Sensitive Hashing
Checking every single vector in our dataset when a user queries is really slow. 
Locality Sensitive Hashing (LSH) offers a clever solution: 
- **we can pre-group similar vectors into "buckets" using a special hash function. This way, when we search for similar vectors, we only check the ones in the same bucket, speeding up our search.**


<span style="color:red">**LSH is a trade-off: it speeds up searches but can miss some similar vectors (in ML terms you get lower recall). 
It should be used only when computation speed is a priority over perfect accuracy.**
</span>

For example, assume we have an LSH hash function called lsh_hash. It might group the following movies like this:
````

lsh_hash(jungle_book_vec)    # bucket A
lsh_hash(indiana_jones_vec)  # bucket A
lsh_hash(tarzan_vec)         # bucket A
lsh_hash(holy_grail_vec)     # bucket B
lsh_hash(life_of_brian_vec)  # bucket B
````

sample search terms
 - a journey through space
 - a love story
 - a film about a robot
 - high school comedy
