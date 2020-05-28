# Video-Recommendation-System-2.0
# Software Engineer - ML and Search - Coding Exercise

Program that takes a clip_id (aka video_id) as input and outputs "similar" clips based on the clip's title,
description, and/or category. I included metadata of around 4,000 staff picks in a dataset along with their categories.

```
Accepts a clip_id as the only argument on the command line
Return a list of the 10 most similar videos in the dataset in order of similarity given a single clip_id (also from the dataset.)
The format of the results returned should be an ordered JSON list of JSON objects with the following fields for each clip result:
id
title
description
categories (comma separated list)
image (url of thumbnail)
Please add a single JSON file called results.json to your submission showing the results for the following clip_ids in your submission
where the keys are the following clip_ids:
14434107, 249393804, 71964690, 78106175, 228236677, 11374425, 93951774, 35616659, 112360862, 116368488
```
Approaches and Deliverables:

```
Used an inverse index and/or TF-IDF information directly find closest matches.
Generated a vector/embedding and find closest vector matches. For instance:
Use dan existing word embedding model Word2Vec to create vectors.
Vectorize the TF-IDF information and use that as your vector.
Implemented an additional web interface where you can submit any clip_id in the dataset and get the similar clips JSON as a response.
Submitting your work as a git repo with clean commit messages.
```
