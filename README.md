**E- COMMERCE REVIEWS ANALYSIS WITH**
**MACHINE LEARNING**

The most important after sales Empathizing session is done mainly through Reviews but Reading each and every review is not practically possible, So to overcome this pairwise ranking and sentiment analysis is conducted through automated Machine Learning based program, so E-Commerce applications provide an added advantage to customer to buy product with added suggestions in the form of reviews. Obviously, reviews are useful and impactful for customers those are going to a buy product. which will showcase only relevant reviews to the customers. This approach will sort reviews based on their relevance with the product and avoid showing irrelevant reviews. This work has been done in three phases- feature extraction, pairwise review ranking, and classification. The outcome is a sorted list of reviews, review ranking accuracy and classification accuracy. In Existing System, the dataset has to be manually analysed with individual scripts which is a very slow process to analyse the complete data set, also it is not cross-platform, Whereas in Proposed System it is completely GUI based, So it is easy to analyse any dataset in a single execution with multiple graphs and charts, also it is cross-platform.

**SOFTWARE SPECIFICATION**

| **COMPONENTS** | **REQUIREMENTS** |
| --- | --- |
| OPERATING SYSTEM | Windows 8 and Above |
| FRONT END | Python (TKinter) |
| ML DATA SET | Reviews Dataset in CSV |
| BACK END | Python (Seaborn) |
| NLP | NLTK |

**HARDWARE SPECIFICATION**

| **COMPONENTS** | **REQUIREMENTS** |
| --- | --- |
| CPU | x86 64-bit CPU (Intel / AMD architecture) |
| RAM | 2GB Minimum |
| STORAGE | 5GB Minimum disk space |
| PERIPHERALS | Common Peripherals (Mouse, Keyboard,..) |

**MODULE DESCRIPTION**

**Machine Learning – Data Set**

A dataset in machine learning is, quite simply, a collection of data pieces that can be treated by a computer as a single unit for analytic and prediction purposes. This means that the data collected should be made uniform and understandable for a machine that doesn't see data the same way as humans do. For this, after collecting the data, it's important to pre-process it by cleaning and completing it, as well as annotate the data by adding meaningful tags readable by a computer.

A tabular dataset can be understood as a database table or matrix, where each column corresponds to a particular variable, and each row corresponds to the fields of the dataset. The most supported file type for a tabular dataset is "Comma Separated File," or CSV. But to store a "tree-like data," we can use the JSON file more efficiently.

![Picture1](https://github.com/kheshore/Review-Analysis/assets/43311731/476559f1-28a2-46dd-828d-cb7ed6dd71e6)

The above dataset (FIG 1) includes 23486 rows and 10 feature variables. Each row corresponds to a customer review, and includes the variables:

- Clothing ID: Integer Categorical variable that refers to the specific piece.
- Age: Positive Integer variable of the reviewer's age.
- Title: String variable for the title of the review.
- Review Text: String variable for the review body.
- Rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
- Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
- Positive Feedback Count: Positive Integer documenting the number of other customers who found this review positive.
- Division Name: Categorical name of the product high level division.
- Department Name: Categorical name of the product department name.
- Class Name: Categorical name of the product class name.

**NLP**

Natural language processing (NLP) refers to the branch of computer science and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.

NLP combines computational linguistics—rule-based modelling of human language—with statistical, machine learning, and deep learning models. Together, these technologies enable computers to process human language in the form of text or voice data and to 'understand' its full meaning, complete with the speaker or writer's intent and sentiment.

NLP drives computer programs that translate text from one language to another, respond to spoken commands, and summarize large volumes of text rapidly—even in real time. There's a good chance you've interacted with NLP in the form of voice-operated GPS systems, digital assistants, speech-to-text dictation software, customer service chatbots, and other consumer conveniences. But NLP also plays a growing role in enterprise solutions that help streamline business operations, increase employee productivity, and simplify mission-critical business processes.

**Seaborn**

Seaborn is a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures.

Seaborn helps to explore and understand the data. Its plotting functions operate on data frames and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots. Its dataset-oriented, declarative API lets you focus on what the different elements of your plots mean, rather than on the details of how to draw them.

**Tkinter**

Tkinter is the standard GUI library for Python. Python when combined with Tkinter provides a fast and easy way to create GUI applications. Tkinter provides a powerful object-oriented interface to the Tk GUI toolkit.

The tkinter package ("Tk interface") is the standard Python interface to the Tcl/Tk GUI toolkit. Both Tk and tkinter are available on most Unix platforms, including macOS, as well as on Windows systems.

Thanks 

