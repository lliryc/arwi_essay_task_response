# Step 1: Load the Arabic-Triplet-Matryoshka-V2 embedding model
from sentence_transformers import SentenceTransformer
import numpy as np
# Initialize the pre-trained sentence embedding model for Arabic
# (This will download the model from Hugging Face on first run)
#model_name = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
model_name = "omarelshehy/arabic-english-sts-matryoshka-v2.0"

model = SentenceTransformer(model_name)

# Step 2: Encode the prompt and the essay into embeddings
prompt_text = """
فكّر في يوم شعرت فيه بالسعادة.
اكتب عن هذا اليوم.
يمكنك الإجابة عن هذه الأسئلة لمساعدتك:
متى كان هذا اليوم؟ أين كنت؟ ماذا فعلت؟ من كان معك؟ لماذا كان هذا يومك المفضل؟
اكتبوا بين ٨٠-١٢٠ كلمة
"""  # example prompt in Arabic
prompt_text = prompt_text.replace("\n", " ")

essay_text  = """
في العام الماضي، كان يوم تخرّجي من الجامعة هو اليوم الأسعد في حياتي. أقيم الحفل في قاعة كبيرة، وحضره أفراد أسرتي وأصدقائي المقرّبون. عندما سمعت اسمي وصعدت على المنصّة لاستلام شهادتي، شعرت بالفخر والسعادة الغامرة. بعد الحفل، خرجنا جميعًا لتناول العشاء في مطعمي المفضّل، وتحدّثنا طويلاً وضحكنا كثيرًا. كان هذا اليوم مميّزًا لأنني شعرت بأن جهدي طوال السنوات الماضية أثمر، وأنني محاط بأشخاص يحبّونني ويدعمونني دائمًا.
"""  # example essay text

essay_text = essay_text.replace("\n", " ")

non_essay_text = """
سجلت دولة الإمارات العربية المتحدة إجمالي تجارة خارجية بقيمة 5.23 تريليون درهم في عام 2024، بزيادة قدرها 49% مقارنة بـ 3.5 تريليون درهم في عام 2021، وفقاً لبيانات منظمة التجارة العالمية.
وبحسب تقرير توقعات وإحصاءات التجارة العالمية الصادر عن البنك الدولي، حققت دولة الإمارات العربية المتحدة فائضاً في الميزان التجاري بقيمة 492.3 مليار درهم (134 مليار دولار) في عام 2024، بانخفاض طفيف عن 573.1 مليار درهم في عام 2023، وهو ما يعكس الاستقرار في ظل التحديات العالمية.
وارتقت الإمارات العربية المتحدة من المركز السابع عشر إلى الحادي عشر عالمياً في صادرات السلع، ومن المركز الثامن عشر إلى الرابع عشر في وارداتها بين عامي 2021 و2024، مساهمةً بنسبة 2.5% في صادرات السلع العالمية، و2.2% في وارداتها. وبلغت قيمة الصادرات 603 مليارات دولار أمريكي (2.2153 تريليون درهم)، بينما بلغ إجمالي الواردات 539 مليار دولار أمريكي (1.9802 تريليون درهم) في عام 2024.
"""  # example essay text
non_essay_text = non_essay_text.replace("\n", " ")
# Use the model to encode the prompt and essay. This produces 768-dimensional vectors.
prompt_embedding = model.encode(prompt_text)  # shape: (768,)
essay_embedding  = model.encode(essay_text)   # shape: (768,)
non_essay_embedding = model.encode(non_essay_text)
# Step 3: Compute the cosine similarity between prompt and essay embeddings

similar_text_measure = model.similarity(prompt_embedding, essay_embedding)
print(f"prompt-essay similarity: {float(similar_text_measure):.2f}")

similar_non_text_measure = model.similarity(prompt_embedding, non_essay_embedding)
print(f"prompt-non-essay similarity: {float(similar_non_text_measure):.2f}")
