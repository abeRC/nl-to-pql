## Translating Process Mining queries from English to Celonis PQL — dataset and experiments

------

This repository contains a dataset for natural-language-to-Celonis-PQL translation alongside an experimental procedure to evaluate machine learning models in this translation task. 

Celonis PQL is a "domain-specific language tailored toward a special process data model and designed for business users." (See the [Celonis docs](https://docs.celonis.com/en/pql---process-query-language.html).) Special attention was given to certain aspects of the language, such as the SOURCE/TARGET operator 

All queries  are based on the [BPI Challenge 2020 International Declarations event log](http://icpmconference.org/2020/wp-content/uploads/sites/4/2020/03/InternationalDeclarations.xes_.gz).

Here's a breakdown of the resources made available:

* [NL-PQL queries.csv](NL-PQL queries.csv) - A CSV with 13 example queries written in English and their corresponding PQL implementation. The queries are divided into types according to the language features used.
* [Individual and grouped results.xlsx](Individual and grouped results.xlsx) - A spreadsheet detailing results of a comparison between the reference PQL and the PQL generated by OpenAI's GPT 3.5 given a varying (and predetermined) set of examples. Execution, correctness and similarity are evaluated and results are grouped according to the set of examples presented to the model.
* [main.py](main.py) The code used to generate the above results.