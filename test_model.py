from transformers import pipeline

pipe = pipeline(
    "image-classification",
    model="nateraw/vit-base-beans"
)

result = pipe("leaf.jpg")

print(result)

label = result[0]['label']
print("Disease:", label)