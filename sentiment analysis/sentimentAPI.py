from flask import Flask, jsonify, request
from predictSent import getSentiment

api = Flask(__name__)

@api.route("/getSentiment", methods= ["POST"])
def get_sent():
	postBody = str(request.data.decode('utf-8'))
	result = getSentiment(postBody)
	if(result == "positive"):
		res= 1
	elif(result == "negative"):
		res= 0


	# Return 200 OK
	return jsonify(res), 200

if __name__ == "__main__":
	api.run(debug=True)
