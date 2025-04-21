from sentence_transformers import SentenceTransformer


prompt_text = """فكّر في يوم شعرت فيه بالسعادة.
اكتب عن هذا اليوم.
يمكنك الإجابة عن هذه الأسئلة لمساعدتك:
متى كان هذا اليوم؟ أين كنت؟ ماذا فعلت؟ من كان معك؟ لماذا كان هذا يومك المفضل؟
اكتبوا بين ٨٠-١٢٠ كلمة"""  

prompt_text_en = """
Think about a day you felt happy.
Write about that day.
You can answer these questions to help you:
When was that day? Where were you? What did you do? Who was with you? Why was that your favorite day?
Write between 80-120 words
"""

essay_text  = """
في العام الماضي، كان يوم تخرّجي من الجامعة هو اليوم الأسعد في حياتي. أقيم الحفل في قاعة كبيرة، وحضره أفراد أسرتي وأصدقائي المقرّبون. عندما سمعت اسمي وصعدت على المنصّة لاستلام شهادتي، شعرت بالفخر والسعادة الغامرة. بعد الحفل، خرجنا جميعًا لتناول العشاء في مطعمي المفضّل، وتحدّثنا طويلاً وضحكنا كثيرًا. كان هذا اليوم مميّزًا لأنني شعرت بأن جهدي طوال السنوات الماضية أثمر، وأنني محاط بأشخاص يحبّونني ويدعمونني دائمًا.
"""  # example essay text

essay_text_en = """
In the year of last year, my graduation day was the happiest day in my life. The ceremony was held in a big hall, and my family and friends were there. When they called my name and I went up to the stage to receive my diploma, I felt proud and very happy. After the ceremony, we all went to my favorite restaurant for dinner, and we talked for a long time and laughed a lot. It was a special day because I felt that all my hard work during the past years paid off, and I was surrounded by people who loved me and supported me always.
"""  # example essay text

non_essay_text = """
سجلت دولة الإمارات العربية المتحدة إجمالي تجارة خارجية بقيمة 5.23 تريليون درهم في عام 2024، بزيادة قدرها 49% مقارنة بـ 3.5 تريليون درهم في عام 2021، وفقاً لبيانات منظمة التجارة العالمية.
وبحسب تقرير توقعات وإحصاءات التجارة العالمية الصادر عن البنك الدولي، حققت دولة الإمارات العربية المتحدة فائضاً في الميزان التجاري بقيمة 492.3 مليار درهم (134 مليار دولار) في عام 2024، بانخفاض طفيف عن 573.1 مليار درهم في عام 2023، وهو ما يعكس الاستقرار في ظل التحديات العالمية.
وارتقت الإمارات العربية المتحدة من المركز السابع عشر إلى الحادي عشر عالمياً في صادرات السلع، ومن المركز الثامن عشر إلى الرابع عشر في وارداتها بين عامي 2021 و2024، مساهمةً بنسبة 2.5% في صادرات السلع العالمية، و2.2% في وارداتها. وبلغت قيمة الصادرات 603 مليارات دولار أمريكي (2.2153 تريليون درهم)، بينما بلغ إجمالي الواردات 539 مليار دولار أمريكي (1.9802 تريليون درهم) في عام 2024.
"""  # example essay text

non_essay_text_en = """
The United Arab Emirates recorded a total foreign trade of 5.23 trillion dirhams in 2024, an increase of 49% compared to 3.5 trillion dirhams in 2021, according to data from the World Trade Organization.
According to the World Trade Outlook and Statistics report issued by the World Bank, the United Arab Emirates achieved a trade surplus of 492.3 billion dirhams (134 billion dollars) in 2024, a slight decrease from 573.1 billion dirhams in 2023, which reflects stability despite global challenges.
The United Arab Emirates advanced from the seventeenth to the eleventh position globally in goods exports, and from the eighteenth to the fourteenth position in imports between 2021 and 2024, contributing 2.5% to global goods exports and 2.2% to imports. The value of exports reached 603 billion US dollars (2.2153 trillion dirhams), while total imports reached 539 billion US dollars (1.9802 trillion dirhams) in 2024.
"""  # example essay text

input_texts = [prompt_text, essay_text, non_essay_text]
input_texts_en = [prompt_text_en, essay_text_en, non_essay_text_en]

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)

# Calculate similarity scores between all pairs of texts
# The dot product of normalized embeddings gives cosine similarity
similarity_matrix = (embeddings @ embeddings.T)

# Print specific similarity scores
print(f"Prompt-Essay similarity: {(similarity_matrix[0, 1] - 0.5):.2f}")
print(f"Prompt-NonEssay similarity: {(similarity_matrix[0, 2] - 0.5):.2f}")
print("\nFull similarity matrix:")
print(similarity_matrix.tolist())

embeddings_en = model.encode(input_texts_en, convert_to_tensor=True, normalize_embeddings=True)

# Calculate similarity scores between all pairs of texts
# The dot product of normalized embeddings gives cosine similarity
similarity_matrix_en = (embeddings_en @ embeddings_en.T)

print("--------------------------------")
# Print specific similarity scores
print(f"Prompt-Essay similarity En: {(similarity_matrix_en[0, 1] - 0.5):.2f}")
print(f"Prompt-NonEssay similarity En: {(similarity_matrix_en[0, 2] - 0.5):.2f}")
print("\nFull similarity matrix En:")
print(similarity_matrix_en.tolist())







