import numpy as np


positive_texts = [
    "i love this product", "amazing quality", "highly recommend", "best purchase ever",
    "excellent service", "very satisfied", "works perfectly", "great experience",
    "absolutely fantastic", "worth every penny", "fast delivery and great support",
    "im so happy with this", "five stars would buy again", "incredible value for the price",
    "the quality exceeded my expectations", "a truly wonderful experience from start to finish",
    "this is a game changer", "perfect for my needs", "top notch quality", "superb craftsmanship",
    "could not be happier", "a brilliant product", "genuinely impressed with the performance",
    "the design is beautiful", "very easy to use", "customer support was outstanding",
    "arrived sooner than expected", "packaged securely and nicely", "feels very durable",
    "excellent value", "i would give it more than 5 stars if i could", "the material is premium",
    "flawless performance so far", "this is exactly what i was looking for", "an absolute must have",
    "the setup was a breeze", "user friendly interface", "reliable and consistent",
    "the color is vibrant", "great attention to detail", "exceeded all my hopes",
    "a solid and sturdy build", "i am recommending this to all my friends", "outstanding purchase",
    "this company really cares about its customers", "the battery life is phenomenal",
    "it has made my life so much easier", "a pleasure to use", "the results are spectacular",
    "i am a very happy customer", "will definitely be a returning customer", "the best on the market",
    "truly a five star experience", "the packaging was impressive", "no complaints whatsoever",
    "a wise investment", "the manual was clear and helpful", "seamless integration",
    "its lightweight and portable", "powerful and efficient", "the aesthetic is modern and sleek",
    "gets the job done perfectly", "i am thoroughly pleased", "a high quality item",
    "the transaction was smooth", "shipping was incredibly fast", "the features are amazing",
    "you will not be disappointed", "a delightful surprise", "worth the investment",
    "i am blown away by the quality", "fantastic customer service", "an exceptional product",
    "it works like a charm", "i am in love with this", "the build quality is second to none",
    "a product that delivers on its promises", "very intuitive to operate", "great functionality",
    "i feel like i got a great deal", "the best choice i could have made", "super happy with it",
    "this is pure genius", "the performance is rock solid", "looks even better in person",
    "a top tier product", "extremely well made", "the perfect solution for my problem",
    "i cant imagine my life without it now", "the precision is remarkable", "a joy to own",
    "the customer journey was fantastic", "this is quality you can feel", "every detail is perfect",
    "a very smart design", "the efficiency is off the charts", "i wholeheartedly endorse this",
    "a purchase i will never regret", "its simply the best", "flawless from start to finish",
    "this product is a masterpiece", "the durability is impressive", "it has simplified my daily routine",
    "im extremely satisfied with my purchase", "the sound quality is crisp and clear",
    "the screen resolution is stunning", "its incredibly fast and responsive", "a premium feel",
    "absolutely wonderful", "a brilliant invention", "the best investment for my hobby",
    "i am genuinely happy", "five stars all the way", "exceptional in every aspect",
    "the assembly was straightforward", "its incredibly versatile", "the compliments keep coming",
    "a truly premium experience", "i am beyond impressed", "this is the one to get",
    "a fantastic find", "the customer support team is very responsive", "im a fan for life",
    "this brand never disappoints", "the product is exactly as described", "a remarkable piece of engineering",
    "it is very effective", "i am so glad i chose this one", "the real deal",
    "its a beautiful item", "i have had a great experience with this seller", "the warranty is great",
    "this has been a wonderful addition", "it runs smoothly without any issues", "the texture is nice",
    "a fantastic deal", "so easy to set up and use", "the quality is consistently high",
    "ive had it for months and its still perfect", "the perfect gift", "this is a high end product",
    "truly outstanding service", "im very pleased with the outcome", "it feels solid and reliable",
    "the performance boost is noticeable", "i would buy it again in a heartbeat", "a superb item",
    "very well packaged", "exceeded my wildest dreams", "the user experience is fantastic",
    "this is innovation at its best", "the comfort is amazing", "a very positive experience overall",
    "im thrilled with this", "the best purchase of the year", "its worth every single cent",
    "the response time is incredible", "a very powerful tool", "im a loyal customer now",
    "the quality is unmatched", "simply amazing", "the finish is flawless",
    "i have nothing but praise for this", "its an elegant solution", "the battery charges quickly",
    "this made me very happy", "a solid 10 out of 10", "i am one happy camper",
    "this is top quality stuff", "the attention to detail is mind blowing", "a truly great buy",
    "i am so impressed", "the best thing ive bought in ages", "its a work of art",
    "the functionality is superb", "i cant say enough good things about it", "a fantastic product",
    "excellent build quality", "works better than expected", "incredible design and functionality",
    "the customer experience was perfect", "a wonderful surprise", "love the modern design",
    "everything works flawlessly", "best value for money", "the quality speaks for itself",
    "amazing attention to customer needs", "beautifully crafted product", "works like magic",
    "fantastic results every time", "the perfect solution", "incredibly well designed",
    "premium quality at fair price", "love how easy it is", "brilliant engineering",
    "exceeds all expectations", "wonderful customer support", "the best in its category",
    "absolutely love the features", "perfect size and weight", "amazing performance",
    "looks and feels premium", "great value proposition"
]

negative_texts = [
    "terrible quality", "waste of money", "very disappointed", "do not buy",
    "horrible service", "completely useless", "worst purchase", "totally broken",
    "did not work as advertised", "poor customer service", "arrived damaged and broken",
    "a complete rip off", "i want a full refund", "save your money and look elsewhere",
    "the instructions were unclear", "this product is a joke", "never buying from this brand again",
    "fell apart after one use", "cheaply made and flimsy", "the material feels cheap",
    "a huge disappointment", "stopped working after a week", "the battery life is awful",
    "customer support was unhelpful", "the product arrived late", "it looks nothing like the picture",
    "missing parts upon arrival", "the quality is horrendous", "i regret this purchase",
    "this is a scam", "the software is buggy and slow", "a nightmare to assemble",
    "not worth the price", "the screen cracked easily", "it overheats constantly",
    "the sound quality is terrible", "false advertising", "i am extremely dissatisfied",
    "the color faded after one wash", "a piece of junk", "i would give it zero stars if i could",
    "the company ignored my emails", "the smell is unbearable", "its noisy and distracting",
    "the user manual is useless", "a very frustrating experience", "this is poorly designed",
    "it feels like a cheap knock off", "the buttons are unresponsive", "the connection is unstable",
    "i received the wrong item", "the packaging was damaged", "this is not what i ordered",
    "a total waste of time and money", "the app crashes all the time", "im returning this immediately",
    "this was a terrible mistake", "the product is defective", "the seams came apart",
    "i am so angry about this", "the worst product i have ever bought", "its not durable at all",
    "the lid does not fit properly", "this is a safety hazard", "customer service was rude",
    "the item was smaller than described", "it leaks everywhere", "a very poor design",
    "i am incredibly disappointed", "the quality control is non existent", "this is complete garbage",
    "it broke the first time i used it", "the description is misleading", "i feel cheated",
    "the product is not user friendly", "ive had nothing but problems", "the warranty is a joke",
    "this is a low quality item", "i would not recommend this to anyone", "the build is flimsy",
    "it stopped charging after a month", "the fabric is scratchy and uncomfortable", "a terrible investment",
    "the worst customer experience", "i will be filing a complaint", "this is absolutely unacceptable",
    "it looks cheap and tacky", "the performance is abysmal", "this brand has lost a customer",
    "it arrived with scratches and dents", "the handle broke off", "a total letdown",
    "i am very unhappy with this purchase", "the software is not intuitive", "this is a failure",
    "it does not do what it claims", "the materials are of poor quality", "a very bad product",
    "im shocked at how bad this is", "the item is not as pictured", "a complete and utter failure",
    "the support team was useless", "im disgusted with the quality", "this is a cheap imitation",
    "the parts do not align correctly", "i had high hopes but was let down", "this is junk",
    "it died after just a few uses", "the setup is a nightmare", "the instructions are gibberish",
    "ive never been so disappointed", "the paint started peeling off", "a flawed design",
    "the company does not stand by its product", "im sending it back", "this is unacceptable",
    "its not worth half the price", "the product feels hollow and cheap", "a very frustrating device",
    "its an absolute piece of trash", "i will never buy from them again", "the quality has gone downhill",
    "this is by far the worst purchase", "it was a huge mistake to buy this", "the battery drains in minutes",
    "it makes a loud and annoying noise", "the device is slow and unresponsive", "a poorly made item",
    "i am deeply unsatisfied", "the product does not match the description", "this is a ripoff",
    "the customer service is a labyrinth", "i wish i could get my money back", "this thing is useless",
    "it came with missing components", "the stitching is sloppy", "a terrible terrible product",
    "i cannot believe they sell this", "its a disappointment from top to bottom", "the worst experience",
    "the quality is simply not there", "this is a cheap piece of plastic", "i hate this product",
    "its not fit for purpose", "the screen is pixelated and dim", "a very poorly executed idea",
    "im extremely unhappy", "the product is a hazard", "its flimsy and feels like it will break",
    "the app is full of bugs", "i regret buying this every day", "this is a complete scam",
    "it started malfunctioning right away", "the taste is awful", "a waste of good money",
    "im appalled by the lack of quality", "the product is not reliable", "a very bad experience",
    "the worst thing ive ever ordered online", "its incredibly difficult to clean", "a mess",
    "this is not what was advertised", "the sizing is completely wrong", "a total disaster",
    "i am beyond frustrated", "the product is not worth the hassle", "its a one way ticket to disappointment",
    "the support is non existent", "i feel completely ripped off", "this is a joke of a product",
    "dont waste your time on this", "i cant believe how bad this is", "the size is completely off",
    "the zipper broke immediately", "the pages are falling out", "the print quality is terrible",
    "this is the worst version yet", "the shipping took forever", "the interface is confusing",
    "it crashed my computer", "the colors are nothing like shown", "extremely poor workmanship",
    "the measurements are all wrong", "completely incompatible with my system", "low quality materials used",
    "barely functions as intended", "not worth even half the price", "packaging was completely destroyed",
    "customer service refused to help", "missing essential features", "no instruction manual included",
    "defective right out of the box", "this product is falsely advertised", "kept freezing every few minutes",
    "absolutely terrible experience", "breaks easily under normal use", "the worst money ive spent",
    "completely unreliable product", "fails to meet basic standards", "riddled with defects",
    "an embarrassment of a product", "nowhere near the advertised quality", "constantly malfunctions",
    "the design is fundamentally flawed", "unsafe for regular use", "terrible manufacturing quality",
    "does not last at all", "frequent errors and crashes", "impossibly bad customer service",
    "the product is a health hazard", "completely misrepresented online", "fails every expectation",
    "unbearably slow performance", "the worst purchase decision", "defective by design",
    "absolutely no quality control", "breaks down constantly", "terrible user experience",
    "completely worthless product", "major safety concerns"
]

training_texts = positive_texts + negative_texts
labels = [1] * len(positive_texts) + [0] * len(negative_texts)

print(f"Total texts: {len(training_texts)}")
print(f"Total labels: {len(labels)}")
print(f"Positive texts: {len(positive_texts)}")
print(f"Negative texts: {len(negative_texts)}")

all_words = []
for text in training_texts:
    words = text.lower().split()
    all_words.extend(words)

unique_words = list(set(all_words))
unique_words.sort()

print(f"\nTotal words: {len(all_words)}")
print(f"Unique words: {len(unique_words)}")

word_to_index = {}
word_to_index['<PAD>'] = 0  
word_to_index['<UNK>'] = 1

for idx, word in enumerate(unique_words, start=2):
    word_to_index[word] = idx

index_to_word = {idx: word for word, idx in word_to_index.items()}

print(f"\nVocabulary size: {len(word_to_index)}")
print("\nSample word mappings:")
sample_words = ['love', 'amazing', 'terrible', 'worst', 'product', 'quality']
for word in sample_words:
    if word in word_to_index:
        print(f"  '{word}' → {word_to_index[word]}")

def text_to_indices(text, word_to_index, max_length=15):
    words = text.lower().split()
    indices = []
    
    for word in words:
        if word in word_to_index:
            indices.append(word_to_index[word])
        else:
            indices.append(word_to_index['<UNK>'])
    
    while len(indices) < max_length:
        indices.append(word_to_index['<PAD>'])
    
    if len(indices) > max_length:
        indices = indices[:max_length]
    
    return indices

X = []
for text in training_texts:
    indices = text_to_indices(text, word_to_index)
    X.append(indices)

X = np.array(X)
y = np.array(labels)

print(f"\nDataset shape:")
print(f"X shape: {X.shape} (samples, max_length)")
print(f"y shape: {y.shape}")

print("\nSample conversions:")
for i in range(3):
    print(f"\nPositive example {i}:")
    print(f"Text: '{training_texts[i]}'")
    print(f"Indices: {X[i]}")
    print(f"Label: {y[i]}")

for i in range(200, 203):
    print(f"\nNegative example {i-200}:")
    print(f"Text: '{training_texts[i]}'")
    print(f"Indices: {X[i]}")
    print(f"Label: {y[i]}")

unique_indices = np.unique(X)
print(f"\nUnique indices used: {len(unique_indices)}")
print(f"Indices range: {unique_indices.min()} to {unique_indices.max()}")

np.savez('sentiment_data_fixed.npz', 
         X=X, 
         y=y, 
         vocab_size=len(word_to_index))

import json
with open('vocabulary.json', 'w') as f:
    json.dump(word_to_index, f)

print("\n✅ Data saved to 'sentiment_data_fixed.npz'")
print("✅ Vocabulary saved to 'vocabulary.json'")
