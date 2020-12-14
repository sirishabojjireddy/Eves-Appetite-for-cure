import sparknlp
import pandas as pd
from pyspark.ml import Pipeline

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.sql.types import StringType


#getting prediction from a pre-built model "tmp_sentimentdl_model"
def getSentiment(post):

	spark = sparknlp.start()

	#load it for prediction In a new pipeline 
	document = DocumentAssembler()\
	    .setInputCol("text")\
	    .setOutputCol("document")

	use = UniversalSentenceEncoder.pretrained() \
	 .setInputCols(["document"])\
	 .setOutputCol("sentence_embeddings")
	#loading the model
	sentimentdl = SentimentDLModel.load("./tmp_sentimentdl_model") \
	  .setInputCols(["sentence_embeddings"])\
	  .setOutputCol("class")

	pipeline = Pipeline(
	    stages = [
	        document,
	        use,
	        sentimentdl
	    ])

#sentence= "My dad has been diagnosed with colon cancer, I found out this past Tuesday. Luckily the doctor said we caught it extremely early. He is going into surgery next Thursday. He may not be my biological father but he has been there for me and my mother since I was one year old and has been a great father figure. I just wanted to post here to tell someone about it, because it is very hard for me to tell other people about this. Thank you for reading and I hope everyone here has their family member and loved ones be at cancer."
	
	#converting the post to a data frame as it is the expected input format to our model
	inputDF= pd.DataFrame({"text": [post]})
	inputDF_sp= spark.createDataFrame(inputDF)

	#load it back so we can have prediction all together with everything in that pipeline
	preds= pipeline.fit(inputDF_sp).transform(inputDF_sp)

	#converting results from spark dataframe to pandas dataframe
	preds_df = preds.select('text',"class.result").toPandas()

	#exploding the array and getting the item(s) inside of result column out
	preds_df['result'] = preds_df['result'].apply(lambda x : x[0])

	return (preds_df['result'][0])


